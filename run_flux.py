import time

import torch
from einops import rearrange
from PIL import Image
import cv2

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long


def get_models(name: str, device: torch.device, offload: bool, fp8: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool, args):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            fp8=args.fp8,
        )
        self.pulid_model = PuLIDPipeline(self.model, device="cpu" if offload else device, weight_dtype=torch.bfloat16,
                                         onnx_provider=args.onnx_provider)
        lora_local_path = "/home/zhengkai/PuLID/models/flux-lora-collection/realism_lora.safetensors"
        
        import pdb; pdb.set_trace()

        self.pulid_model.set_lora(lora_local_path, lora_weight=1.0)

        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(args.pretrained_model, version=args.version)

    @torch.inference_mode()
    def generate_image(
            self,
            width,
            height,
            num_steps,
            start_step,
            guidance,
            seed,
            prompt,
            id_image=None,
            id_weight=1.0,
            neg_prompt="",
            true_cfg=1.0,
            timestep_to_start_cfg=1,
            max_sequence_length=128,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed), self.pulid_model.debug_img_list




def run(args, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu",
                offload: bool = False, aggressive_offload: bool = False):
    generator = FluxGenerator(model_name, device, offload, aggressive_offload, args)
    width, height = 896, 1152
    num_steps = 20
   
    max_sequence_length = 128
    prompt =  'a women holding a sign that says \'I LOVE WAVESPEED!\',  standing in front of a beach',
    id_image = 'example_inputs/gehuijie.jpg'
    #id_image = Image.open(id_image).convert("RGB")
    id_image = cv2.imread(id_image)
    #id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)

    start_step = 4
    guidance = 4.0
    seed = 2680261499100305976
    true_cfg = 1.0

    outputs = generator.generate_image(width, height, num_steps, start_step, guidance, seed, prompt, id_image, id_weight=1.0,
                             neg_prompt="", true_cfg=true_cfg, timestep_to_start_cfg=1, max_sequence_length=max_sequence_length)
    
    return outputs[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PuLID for FLUX.1-dev")
    parser.add_argument('--version', type=str, default='v0.9.1', help='version of the model', choices=['v0.9.0', 'v0.9.1'])
    parser.add_argument("--name", type=str, default="flux-dev", choices=list('flux-dev'),
                        help="currently only support flux-dev")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use, for 24G GPUs")
    parser.add_argument("--fp8", action="store_true", help="use flux-dev-fp8 model")
    parser.add_argument("--onnx_provider", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="set onnx_provider to cpu (default gpu) can help reduce RAM usage, and when combined with"
                             "fp8 option, the peak RAM is under 15GB")
    parser.add_argument("--port", type=int, default=8080, help="Port to use")
    parser.add_argument("--dev", action='store_true', help="Development mode")
    parser.add_argument("--pretrained_model", type=str, help='for development')
    args = parser.parse_args()

    if args.aggressive_offload:
        args.offload = True

    image = run(args, args.name, args.device, args.offload, args.aggressive_offload)
    image.save("output_1.png")
