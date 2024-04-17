
class BaseConfig:
    """ base Encoder Decoder config """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class NMTConfig(BaseConfig):
    # Data
    src_lang = 'en'
    tgt_lang = 'vi'
    max_len = 75
    add_special_tokens = True

    # Model
    model_name = "facebook/mbart-large-50-many-to-many-mmt"

    # Training
    device = "cuda"
    learning_rate = 5e-5
    train_batch_size = 16
    eval_batch_size = 16
    num_train_epochs = 2
    save_total_limit = 1
    ckpt_dir = f'./mbart50-{src_lang}-{tgt_lang}'
    eval_steps = 1000

    # Inference
    beam_size = 5

