# training script.

import os
from importlib.resources import files

import hydra

from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.config import TTSConfig

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(cfg):
    tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    exp_name = f"{cfg.model.name}_{mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"

    # set text tokenizer

    train_dataset = load_dataset(cfg.datasets.name, 
                                 tokenizer, 
                                 dataset_type = cfg.datasets.dataset_type, 
                                 mel_spec_kwargs=cfg.model.mel_spec,
                                 max_spec_lengths = cfg.datasets.max_spec_lengths,
                                 testing = cfg.datasets.testing)
    if hasattr(train_dataset, 'getTokenizer'):
        vocab_char_map, vocab_size = train_dataset.getTokenizer()

    else: 
        if tokenizer != "custom":
            tokenizer_path = cfg.datasets.name
        else:
            tokenizer_path = cfg.model.tokenizer_path

        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    wandb_resume_id = None

    tts_config = TTSConfig(vocab_size = vocab_size)

    model = CFM(
        tts_config,
        vocab_char_map=vocab_char_map
    )

    # init trainer
    trainer = Trainer(
        model,
        cfg=cfg,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{cfg.ckpts.save_dir}")),
        batch_size=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        weight_decay=cfg.optim.weight_decay,
        logger=cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=cfg.ckpts.last_per_steps,
        log_samples=True,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path=cfg.model.vocoder.local_path,
    )

    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
