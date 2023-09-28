import re
import tempfile
import zipfile
import os
import time
import subprocess
from cog import BasePredictor, Input, Path
import importlib  
from sd_scripts.train_network import setup_parser, train
from diffusers import DiffusionPipeline
import torch

BASE_MODEL_CACHE = "./base-model-cache"
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)
    
class Predictor(BasePredictor):
    def setup(self):
        start = time.time()
        
        print("Ended to download", time.time() - start)
    
    def predict(
        self,
        # Train data path 
        pretrained_model_name_or_path: str = Input(
            description="base model name or path",
            default="stabilityai/stable-diffusion-xl-base-1.0"
        ),
        train_data_zip: Path = Input(
            description="Upload image dataset in zip format using this naming convention: XX_token className.zip"),
        network_weights: Path = Input(
            description=
            "Pretrained LoRA weights",
            default=None),
        output_name: str = Input(description="Model name", default="new_model_name"),
        save_model_as: str = Input(
            description="model save extension | ckpt, pt, safetensors",
            default="safetensors",
            choices=["ckpt", "pt", "safetensors"]),
        # Train related params
        resolution: str = Input(
            description=
            "image resolution must be 'size' or 'width,height'.",
            default="1024"),
        batch_size: int = Input(
            description="batch size", default=1, ge=1),
        max_train_epoches: int = Input(
            description="max train epoches", default=20, ge=1),
        save_every_n_epochs: int = Input(
            description="save every n epochs",
            default=5,
            ge=1),
        train_unet_only: bool = Input(
            description=
            "train U-Net only",
            default=False),
        train_text_encoder_only: bool = Input(
            description="train Text Encoder only", default=False),
        seed: int = Input(
            description=
            "reproducable seed",
            default=98796,
            ge=1),
        noise_offset: float = Input(
            description=
            "noise offset",
            default=0,
            ge=0,
            le=1),
        keep_tokens: int = Input(
            description=
            "keep heading N tokens when shuffling caption tokens",
            default=0,
            ge=0),

        # Learning rate
        learning_rate: float = Input(description="Learning rate. It means 0.0001 or 0.0009",
                                default=4,
                                ge=1,
                                le=9),
        unet_lr: float = Input(description="UNet learning rate. It means 0.0001 or 0.0009",
                                default=1,
                                ge=1,
                                le=9),
        text_encoder_lr: float = Input(
            description="Text Encoder learning rate. It means 0.0001 or 0.0009",
            default=1,
            ge=1,
            le=9),
        lr_scheduler: str = Input(
            description=
            """"linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup""",
            default="cosine",
            choices=[
                "linear", "cosine", "cosine_with_restarts", "polynomial",
                "constant", "constant_with_warmup"
            ]),
        lr_warmup_steps: int = Input(
            description=
            "warmup steps",
            default=0,
            ge=0),
        lr_scheduler_num_cycles: int = Input(
            description=
            "cosine_with_restarts restart cycles",
            default=1,
            ge=1),

        # Bucket size
        min_bucket_reso: int = Input(
            description="arb min resolution", default=256, ge=1),
        max_bucket_reso: int = Input(
            description="arb max resolution", default=1024, ge=1),
        persistent_data_loader_workers: bool = Input(
            description=
            "makes workers persistent, further reduces/eliminates the lag in between epochs. however it may increase memory usage",
            default=True),
        clip_skip: int = Input(description="clip skip",
                               default=1,
                               ge=0),
        # Optimizer
        optimizer_type: str = Input(
            description=
            """adaFactor","AdamW","AdamW8bit","Lion","SGDNesterov","SGDNesterov8bit","DAdaptation", "Lion", "Prodigy""",
            default="Lion",
            choices=["adaFactor", "AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "Prodigy"]),
        network_module: str = Input(description="Network module",
                                    default="networks.lora",
                                    choices=["networks.lora", "networks.dylora", "lycoris.kohya"]),
        network_dim: int = Input(description="network dimension",
                                 default=32,
                                 ge=1),
        network_alpha: int = Input(
            description=
            "network alpha",
            default=16,
            ge=1)
    ) -> Path:
        # Unzip the dataset 
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        # Setup the output parameters
        output_dir = Path(tempfile.mkdtemp())
        if not output_name:
            output_name = Path(
                re.sub("[^-a-zA-Z0-9_]", "", train_data_zip.name)).name
        
        parser = setup_parser()
        args = parser.parse_args()
        args.enable_bucket = True
        args.pretrained_model_name_or_path = BASE_MODEL_ID
        args.train_data_dir = train_data_dir
        args.output_dir = output_dir
        args.output_name = output_name
        args.logging_dir = "./logs"
        args.resolution = resolution
        args.max_train_epochs = max_train_epoches
        args.learning_rate = learning_rate / 10000
        args.unet_lr = unet_lr
        args.text_encoder_lr = text_encoder_lr
        args.lr_scheduler = lr_scheduler
        args.lr_warmup_steps = lr_warmup_steps
        args.lr_scheduler_num_cycles = lr_scheduler_num_cycles
        args.network_module = network_module
        args.network_dim = network_dim
        args.network_alpha = network_alpha
        args.output_name = output_name
        args.train_batch_size = batch_size
        args.save_every_n_epochs = save_every_n_epochs
        args.mixed_precision = "bf16" 
        args.save_precision = "fp16"
        args.seed = seed
        args.cache_latents = True
        args.clip_skip = clip_skip
        args.prior_loss_weight = 1
        args.max_token_length = 225
        args.caption_extension = ".txt"
        args.save_model_as = save_model_as
        args.min_bucket_reso = min_bucket_reso
        args.max_bucket_reso = max_bucket_reso
        args.keep_tokens = keep_tokens
        if noise_offset > 0:
            args.noise_offset = noise_offset
        args.xformers = True
        args.shuffle_caption = True

        # Fine tune parameters

        if train_unet_only:
            args.train_unet_only = True

        if train_text_encoder_only:
            args.train_text_encoder_only = True

        if persistent_data_loader_workers:
            args.persistent_data_loader_workers = True

        if optimizer_type == "adafactor":
            args.optimizer_type = optimizer_type
            args.optimizer_args = ["scale_parameter=True", "warmup_init=True"]
        elif optimizer_type == "DAdaptation":
            args.optimizer_type = optimizer_type
            args.optimizer_args = ["decouple=True"]
            args.learning_rate = 1
            args.unet_lr = 1
            args.text_encoder_lr = 0.5
        elif optimizer_type == "Lion":
            args.use_lion_optimizer = True
        elif optimizer_type == "AdamW8bit":
            args.AdamW8bit = True
        elif optimizer_type == "Prodigy":
            args.optimizer_type = "Prodigy"
            args.optimizer_args = ["weight_decay=0.01", "decouple=True", "use_bias_correction=True"] # TO TEST from: https://civitai.com/articles/1022/update-sdxl-scriptbdsqlsz-lora-training-advanced-tutorial2prodigy-is-all-you-need

        if network_weights:
            args.network_weights = network_weights

        train(args)
        return output_dir