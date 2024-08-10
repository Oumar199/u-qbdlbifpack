import torch

class LSTMOutput:
    
    def __init__(self, logits, loss):
        
        self.logits = logits
        
        self.loss = loss

class LSTMSequenceToSequence(torch.nn.Module):

  def __init__(self, tokenizer, embedding_size = 128, num_layers = 300, hidden_size = 128, dropout=0.1, bidirectional = True):

    super().__init__()

    self.tokenizer = tokenizer
    
    self.vocab_size = self.tokenizer.vocab_size

    self.embedding = torch.nn.Embedding(self.vocab_size, embedding_size)

    self.encoder = torch.nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True,
                                 bidirectional = bidirectional, dropout=dropout)

    self.decoder = torch.nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True,
                                 bidirectional = bidirectional, dropout=dropout)

    copy = 2 if bidirectional else 1

    self.decoder_output_layer = torch.nn.Linear(copy * hidden_size, self.vocab_size)

    self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

  def forward(self, input, output):
    
    input_embed = self.embedding(input)

    print(type(input_embed))
    print(input_embed.shape)
    
    state, hidden = self.encoder(input_embed)

    # decal output for decoder
    decoder_input = output[:, :-1]

    decoder_input = self.embedding(decoder_input)

    decoder_output, _ = self.decoder(decoder_input, hidden)

    decoder_output = self.decoder_output_layer(output)
    
    loss = self.loss_fn(output.view(-1, output.shape[-1]), output[:, 1:].view(-1))

    return LSTMOutput(decoder_output, loss)

  def generate(self, input, max_new_tokens: int = 100, temperature: float = 0.0, use_sampling = False, **kwargs):

    input_embed = self.embedding(input)

    _, hidden = self.encoder(input_embed)

    # initialize predictions
    predictions = torch.tensor(self.tokenizer._bos_token, dtype=input.dtype, device=input.device).unsqueeze(0)

    # generate predictions
    for i in range(max_new_tokens):

      decoder_input = self.embedding(predictions)

      decoder_output, hidden = self.decoder(decoder_input, hidden)

      decoder_output = self.decoder_output_layer(decoder_output)

      if temperature > 0.0: decoder_output = (decoder_output / temperature)

      # get probs and sample the next token from a multinomial distribution
      probs = torch.softmax(decoder_output[:, -1], dim = -1)

      if use_sampling: prediction = torch.multinomial(probs, num_samples = 1)
      else: prediction = torch.argmax(probs, dim=-1, keepdim=True)

      # add new prediction
      predictions = torch.cat((predictions, prediction), dim = -1)

    # return predictions
    return predictions[:, 1:]

