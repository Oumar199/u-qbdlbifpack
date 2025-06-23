from translate_package import (
    pl,
    evaluate,
    LoraConfig,
    TaskType,
    torch,
    get_linear_schedule_with_warmup,
    wandb,
    get_peft_model,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    Adafactor,
    AdamWeightDecay
)

from translate_package.models.lstm import LSTMSequenceToSequence


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

class MachineTranslationTransformer(pl.LightningModule):

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    def __init__(
        self,
        model_name,
        tokenizer,
        model_generation="t5",
        model=None,
        lr=1e-4,
        weight_decay=1e-2,
        num_warmup_steps=0,
        num_training_steps=20000,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        max_new_tokens=200,
        predict_with_generate=True,
        num_beams=1,
        use_peft=False,
        embedding_size=128,
        num_layers=6,
        hidden_size=128,
        dropout=0.1,
        bidirectional=False
    ):

        super().__init__()

        if model is None:
            if model_generation in ["t5"]:
                
                self.original_model = T5ForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.float32
                )
                
            elif model_generation in ["mt5"]:
                
                self.original_model = MT5ForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.float32
                )
                
            elif model_generation in ["nllb"]:
                
                self.original_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch.float32
                )
                
            elif model_generation in ["bart"]:
                
                self.original_model = BartForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.float32
                )
            
            elif model_generation in ["lstm"]:
                
                self.original_model = LSTMSequenceToSequence(tokenizer, embedding_size, num_layers, hidden_size, dropout, bidirectional)
                
            # resize the token embeddings
            if not model_generation in ["lstm"]: self.original_model.resize_token_embeddings(len(tokenizer))
            
            if use_peft and not model_generation in ["lstm"]:
                
                self.lora_config = LoraConfig(
                    r=r,  # Rank
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=bias,
                    task_type=TaskType.SEQ_2_SEQ_LM, 
                )
                
                self.model = get_peft_model(self.original_model, self.lora_config)          
            
            else:
                
                self.model = self.original_model
            
        else:

            self.model = model

        print(print_number_of_trainable_model_parameters(self.model))

        self.tokenizer = tokenizer
        
        self.lr = lr

        self.weight_decay = weight_decay

        self.num_warmup_steps = num_warmup_steps

        self.num_training_steps = num_training_steps

        self.predict_with_generate = predict_with_generate

        self.max_new_tokens = max_new_tokens

        self.num_beams = num_beams
        
        self.model_generation = model_generation

        self.predictions = {
            "Source references": [],
            "Predictions": [],
            "Target references": [],
        }

    def forward(self, input):

        output = self.model(**input) if not self.model_generation in ["lstm"] else self.model(input['input_ids'], input['labels'])

        return output.loss, output.logits

    def configure_optimizers(self):

        if self.model_generation in ["t5", "mt5"]:
            
            optimizer = Adafactor(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay, relative_step = False,
                warmup_init = False
            )
        
        elif self.model_generation in ["bart", "mbart"]:
            
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        
        elif self.model_generation in ["lstm"]:
            
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        if self.model_generation in ["t5", "lstm", "mt5"]:
            
            return [optimizer]

        elif self.model_generation in ["bart"]:

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            
            return {'optimizer': optimizer, 'lr_scheduler': {"scheduler": scheduler}}
            
    def training_step(self, batch, batch_idx=None):

        loss, y_pred = self(batch)

        self.log_dict(
            {"train_loss": loss, "global_step": float(self.global_step)},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        wandb.log({"train_loss": loss, "trainer/global_step": self.global_step})

        return loss

    def validation_step(self, batch, batch_idx=None):

        loss, y_pred = self(batch)

        metrics = {}

        if self.predict_with_generate:

            # generate predictions
            predictions = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.max_new_tokens,
            ) if not self.model_generation in ["lstm"] else self.model.generate(
                input=batch["input_ids"],
                max_new_tokens=self.max_new_tokens,
            ) 

            # decode the labels
            predictions = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            labels = self.tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )

            # get bleu metric
            bleu = self.bleu.compute(
                predictions=predictions,
                references=[[label.strip()] for label in labels],
            )

            metrics["bleu"] = bleu["score"]

            # get rouge metrics
            rouge = self.rouge.compute(
                predictions=predictions, references=[label.strip() for label in labels]
            )

            metrics.update({k: v for k, v in rouge.items() if "rouge" in k})

        # get the loss
        metrics.update(
            {"eval_loss": loss.item(), "global_step": float(self.global_step)}
        )

        self.log_dict(
            metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        metrics.update({"trainer/global_step": self.global_step})

        wandb.log(metrics)

        return loss

    def test_step(self, batch, batch_idx):

        loss, y_pred = self(batch)

        references = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )

        # generate predictions
        predictions = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            num_beams=self.num_beams
        ) if not self.model_generation in ["lstm"] else self.model.generate(
            input=batch["input_ids"],
            max_new_tokens=self.max_new_tokens,
            use_sampling=True
        )

        # decode the labels
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        self.predictions["Source references"].extend(references)
        self.predictions["Predictions"].extend(predictions)
        self.predictions["Target references"].extend(labels)

        # get bleu metric
        bleu = self.bleu.compute(
            predictions=predictions, references=[[label.strip()] for label in labels]
        )

        metrics = {}

        metrics["bleu"] = bleu["score"]

        # get rouge metrics
        rouge = self.rouge.compute(predictions=predictions, references=labels)

        metrics.update({k: v for k, v in rouge.items() if "rouge" in k})

        # get the loss
        metrics.update(
            {"test_loss": loss.item(), "global_step": float(self.global_step)}
        )

        # log metrics
        self.log_dict(
            metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
