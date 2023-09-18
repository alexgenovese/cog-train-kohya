# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import re
import tempfile
import zipfile
from cog import BasePredictor, Input, Path
from sd_scripts.train_network import setup_parser, train


class Predictor(BasePredictor):

    def setup(self):
        pass

    def predict(
        self,

        # Train data path | 设置训练用模型、图片
        pretrained_model_name_or_path: str = Input(
            description="base model name or path | 底模名称或路径",
            default="CompVis/stable-diffusion-v1-4"),
        train_data_zip: Path = Input(
            description="train dataset zip file | 训练数据集zip压缩包"),
        network_weights: Path = Input(
            description=
            "pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请上传文件",
            default=None),
        training_comment: str = Input(
            description="training_comment | 训练介绍，可以写作者名或者使用触发关键词",
            default="this LoRA model credit from replicate-sd-scripts"),

        # Output settings | 输出设置
        output_name: str = Input(description="output model name | 模型保存名称",
                                 default=None),
        save_model_as: str = Input(
            description="model save ext | 模型保存格式 ckpt, pt, safetensors",
            default="safetensors",
            choices=["ckpt", "pt", "safetensors"]),

        # Train related params | 训练相关参数
        resolution: str = Input(
            description=
            "image resolution must be 'size' or 'width,height'. 图片分辨率，正方形边长 或 宽,高。支持非正方形，但必须是 64 倍数",
            default="512"),
        batch_size: int = Input(
            description="batch size 一次性训练图片批处理数量，根据显卡质量对应调高", default=1, ge=1),
        max_train_epoches: int = Input(
            description="max train epoches | 最大训练 epoch", default=10, ge=1),
        save_every_n_epochs: int = Input(
            description="save every n epochs | 每 N 个 epoch 保存一次",
            default=2,
            ge=1),
        network_dim: int = Input(description="network dim | 常用 4~128，不是越大越好",
                                 default=32,
                                 ge=1),
        network_alpha: int = Input(
            description=
            "network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率",
            default=32,
            ge=1),
        train_unet_only: bool = Input(
            description=
            "train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启",
            default=False),
        train_text_encoder_only: bool = Input(
            description="train Text Encoder only | 仅训练 文本编码器", default=False),
        seed: int = Input(
            description=
            "reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词",
            default=1337,
            ge=1),
        noise_offset: float = Input(
            description=
            "noise offset | 在训练中添加噪声偏移来改良生成非常暗或者非常亮的图像，如果启用，推荐参数为 0.1",
            default=0,
            ge=0,
            le=1),
        keep_tokens: int = Input(
            description=
            "keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变",
            default=0,
            ge=0),

        # Learning rate | 学习率
        learning_rate: float = Input(description="Learning rate | 学习率",
                                     default=6e-5,
                                     ge=0),
        unet_lr: float = Input(description="UNet learning rate | UNet 学习率",
                               default=6e-5,
                               ge=0),
        text_encoder_lr: float = Input(
            description="Text Encoder learning rate | Text Encoder 学习率",
            default=7e-6,
            ge=0),
        lr_scheduler: str = Input(
            description=
            """"linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
推荐默认cosine_with_restarts或者polynomial，配合输出多个epoch结果更玄学""",
            default="cosine_with_restarts",
            choices=[
                "linear", "cosine", "cosine_with_restarts", "polynomial",
                "constant", "constant_with_warmup"
            ]),
        lr_warmup_steps: int = Input(
            description=
            "warmup steps | 仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值",
            default=0,
            ge=0),
        lr_scheduler_num_cycles: int = Input(
            description=
            "cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效",
            default=1,
            ge=1),

        # 其他设置
        min_bucket_reso: int = Input(
            description="arb min resolution | arb 最小分辨率", default=256, ge=1),
        max_bucket_reso: int = Input(
            description="arb max resolution | arb 最大分辨率", default=1024, ge=1),
        persistent_data_loader_workers: bool = Input(
            description=
            "makes workers persistent, further reduces/eliminates the lag in between epochs. however it may increase memory usage | 跑的更快，吃内存。大概能提速2.5倍，容易爆内存，保留加载训练集的worker，减少每个 epoch 之间的停顿",
            default=True),
        clip_skip: int = Input(description="clip skip | 玄学 一般用 2",
                               default=2,
                               ge=0),
        # 优化器
        optimizer_type: str = Input(
            description=
            """优化器，"adaFactor","AdamW","AdamW8bit","Lion","SGDNesterov","SGDNesterov8bit","DAdaptation",  推荐 新优化器Lion。推荐学习率unetlr=lr=6e-5,tenclr=7e-6""",
            default="Lion",
            choices=[
                "adaFactor", "AdamW", "AdamW8bit", "Lion", "SGDNesterov",
                "SGDNesterov8bit", "DAdaptation"
            ]),
        network_module: str = Input(description="Network module",
                                    default="networks.lora",
                                    choices=["networks.lora"])
    ) -> Path:

        # 解压训练集
        train_data_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(train_data_zip, 'r') as zip_ref:
            zip_ref.extractall(train_data_dir)

        # 创建输出文件
        output_dir = Path(tempfile.mkdtemp())
        if not output_name:
            output_name = Path(
                re.sub("[^-a-zA-Z0-9_]", "", train_data_zip.name)).name

        parser = setup_parser()
        args = parser.parse_args()
        args.enable_bucket = True
        args.pretrained_model_name_or_path = pretrained_model_name_or_path
        args.training_comment = training_comment
        args.train_data_dir = train_data_dir
        args.output_dir = output_dir
        args.output_name = output_name
        args.logging_dir = "./logs"
        args.resolution = resolution
        args.network_module = network_module
        args.max_train_epochs = max_train_epoches
        args.learning_rate = learning_rate
        args.unet_lr = unet_lr
        args.text_encoder_lr = text_encoder_lr
        args.lr_scheduler = lr_scheduler
        args.lr_warmup_steps = lr_warmup_steps
        args.lr_scheduler_num_cycles = lr_scheduler_num_cycles
        args.network_dim = network_dim
        args.network_alpha = network_alpha
        args.output_name = output_name
        args.train_batch_size = batch_size
        args.save_every_n_epochs = save_every_n_epochs
        args.mixed_precision = "fp16"
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

        # 设置优化器

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

        if network_weights:
            args.network_weights = network_weights

        train(args)
        return output_dir