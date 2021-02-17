import torch
import os
from tqdm import tqdm
import re 

class LSTM(nn.Module):
    ## Init which creates the actual model and loads the latest weights
    def __init__(self, input_size=1, n_hidden_layers=2, hidden_layer_size=32, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = n_hidden_layers

        # Create LSTM layer for prediction and Linear layer for output
        self.lstm = nn.LSTM(input_size, hidden_layer_size, n_hidden_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    
    # Function that simply puts the input sequence through the neural net and calculates the output
    def forward(self, input_seq):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_hidden_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.n_hidden_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_()
       
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        lstm_out, (hn, cn) = self.lstm(input_seq, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out[:, -1, :] -->  just want last time step hidden states!
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions






class StockPredictor():
    # Create a model instance, the loss function and the optimizer
    def __init__(self):
        self.model = LSTM()
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        print('Successfully create LSTM model for stock prediction')
        print(self.model)


        # Setup of parameters for training
        self.epochs = 150
        self.train_percentage = 0.8
        self.test_percentage = 1-self.train_percentage

    ## Function that take as input multiple pandas time series and converts them to a train and a test set
    def add_train_data(self, inputs):
        # Detect missing values and fill automatically using pandas module

        # Create the train and the test set
        self.train_sequences = []
        self.test_sequences = []

        # Loop through each of the inputs and add the corresponding sequence to the arrays
        print('Adding input batches to model for training')
        for batch in inputs:
            print('Processing batch: ', batch)
            # We need input_size + output_size items for the training
            required_values = self.model.input_size + self.model.output_size
            for i in range(0, len(batch), required_values):
                # Create a variable that detects whether we should add to the train or the test set
                current_batch_percentage = i/len(batch)
                if i+required_values < len(batch):
                    # Obtain the sequence and normalise based on first input
                    sequence = batch[i:i+required_values]
                    sequence = sequence/sequence[0]
                      
                    # Split into x and y and convert to pytorch tensor
                    x = torch.from_numpy(sequence[:self.input_size]).type(torch.Tensor)
                    y = torch.from_numpy(sequence[self.input_size:]).type(torch.Tensor)                   
                    
                    sequence = np.array([x, y])
                    print('Created sequence', sequence)

                    if current_batch_percentage > self.train_percentage:
                        self.test_sequences.append(sequence)                       
                    else:
                        self.train_sequences.append(sequence)                       

 
                    print()
                    print()
            print()
            print()
            print()
            print()
        print('Succesfully load train and test data')

    def save_current_checkpoint(self):
        torch.save({
              'model_state_dict': self.model.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
            }, './checkpoint_%s.pt' % (epoch))

    def load_current_checkpoint(self, path='latest'):
        if path == 'latest':
            files = os.listdir('.')
            path = sort(re.findall('checkpoint*'))[0]
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## Training
    def train(self):
        self.model.train()
        assert self.train_sequences.shape[2] == 2, 'Your input sequences do not contain any labels. Are you sure you want to train the model? Shape given by %s' % (self.train_sequences.shape)
        i = 0
        for sequence, label in self.train_sequences
            # Obtain the prediction of the current sequence
            prediction = self.model(sequence)

            # Caclulate the losses
            loss = self.loss_function(prediction, label)

            # Print current loss in case we are there
            if i % 25 == 0:
                print('Epoch %s, MSE: %.2f' % (i, loss.item()))
            if i % 100 = 0:
                print('Saving current checkpoint.')
                self.save_current_checkpoint()

            # Zero the gradients as otherwise they will accumulate between epochs 
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            i += 1


    ## Predicting
    def predict(self):
        self.model.eval()
        predictions = np.array([])
        assert self.train_sequences.shape[2] == 2, 'Your input sequences do not contain any labels. Are you sure you want to train the model?'
        for sequence in self.input_sequences:
            # Execute everything without gradient calculation so that it goes faster
            with torch.no_grad():
                prediction = self.model(sequence)
                predictions = np.append(predictions, prediction)

    ## Post processing
    # Convert the scaled data back to the previous data using the normalisation parameters of preprocessing


    ## Saving of weights


    ## Loading of weights
