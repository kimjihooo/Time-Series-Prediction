import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


class Train_LSTM:
    def __init__(self, random_seed: int):
        self.random_seed = random_seed

    def reshape_dataset(self, df: pd.DataFrame):
        if "y" in df.columns:
            self.df = df.drop(columns=["y"]).assign(y=df["y"])
        else:
            raise KeyError("Not found target column y in dataset.")
        
        # df to array
        dataset = df.values.reshape(df.shape)
        return dataset
    
    def sequential_dataset(self, dataset: np.array, seq_len: int, steps: int):

        """
        :param seq_len : Length of sequences. (window size)
        :param steps : Length to predict.
        """

        X, y = list(), list()
        for i, _ in enumerate(dataset):
            idx_in = i + seq_len
            idx_out = idx_in + steps
            if idx_out > len(dataset):
                break
            seq_x = dataset[i:idx_in, :-1]
            seq_y = dataset[idx_out - 1 : idx_out, -1]
            
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def split_train_valid_dataset(
        self,
        df: pd.DataFrame,
        seq_len: int,
        steps: int,
        validation_split: float = 0.3,
        verbose: bool = True):

        dataset = self.reshape_dataset(df=df)

        # Make sequential dataset
        X, y = self.sequential_dataset(
            dataset=dataset,
            seq_len=seq_len,
            steps=steps)

        # Split validation dataset
        dataset_size = len(X)
        train_size = int(dataset_size * (1 - validation_split))
        X_train, y_train = X[:train_size, :], y[:train_size, :]
        X_val, y_val = X[train_size:, :], y[train_size:, :]
        if verbose:
            print(f"X_train: {X_train.shape}")
            print(f"y_train: {y_train.shape}")
            print(f"X_val: {X_val.shape}")
            print(f"y_val: {y_val.shape}")
        return X_train, y_train, X_val, y_val
    
    def build_and_compile_lstm_model(
        self,
        seq_len: int,
        n_features: int,
        lstm_units: list,
        learning_rate: float,
        dropout: float,
        metrics: str,
        dense_units: list = None,
        activation: str = None,
    ):
        """
        :param seq_len: Length of sequences. (Look back window size)
        :param n_features: Number of features. It requires for model input shape.
        :param lstm_units: Number of cells each LSTM layers.
        :param learning_rate: Learning rate.
        :param dropout: Dropout rate.
        :param metrics: Model loss function metric.
        :param dense_units: Number of cells each Dense layers.
        :param activation: Activation function of Layers.
        """

        tf.random.set_seed(self.random_seed)
        model = Sequential()

        # LSTM layers
        if len(lstm_units) > 1:
            # LSTM -> ... -> LSTM -> Dense(steps)
            model.add(
                LSTM(
                    units=lstm_units[0],
                    activation=activation,
                    return_sequences=True,
                    input_shape=(seq_len, n_features),
                )
            )
            lstm_layers = lstm_units[1:]
            for i, n_units in enumerate(lstm_layers, start=1):
                if i == len(lstm_layers):
                    model.add(
                        LSTM(
                            units=n_units,
                            activation=activation,
                            return_sequences=False,
                        )
                    )
                else:
                    model.add(
                        LSTM(
                            units=n_units,
                            activation=activation,
                            return_sequences=True,
                        )
                    )
        else:
            # LSTM -> Dense(steps)
            model.add(
                LSTM(
                    units=lstm_units[0],
                    activation=activation,
                    return_sequences=False,
                    input_shape=(seq_len, n_features),
                )
            )
        
        # Dense layers and dropout
        if dense_units:
            for n_units in dense_units:
                model.add(Dense(units=n_units, activation=activation))
        if dropout > 0:
            model.add(Dropout(rate=dropout))
        model.add(Dense(1))
    
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=MSE, metrics=metrics)
        return model
    
    def fit_lstm(
        self,
        df: pd.DataFrame,
        steps: int,
        lstm_units: list,
        activation: str,
        dropout: float = 0,
        seq_len: int = 16,
        epochs: int = 200,
        batch_size: int = None,
        steps_per_epoch: int = None,
        learning_rate: float = 0.001,
        patience: int = 10,
        validation_split: float = 0.3,
        dense_units: list = None,
        metrics: str = "mse",
        check_point_name: str = None,
        verbose: bool = True,
    ):

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        # Set train and validation data
        (self.X_train,
        self.y_train,
        self.X_val,
        self.y_val,
        ) = self.split_train_valid_dataset(
            df=df,
            seq_len=seq_len,
            steps=steps,
            validation_split=validation_split,
            verbose=verbose,
        )

        # Build LSTM model
        n_features = df.shape[1] - 1
        self.model = self.build_and_compile_lstm_model(
            seq_len=seq_len,
            n_features=n_features,
            lstm_units=lstm_units,
            activation=activation,
            learning_rate=learning_rate,
            dropout=dropout,
            dense_units=dense_units,
            metrics=metrics
        )

        # Save best model
        if check_point_name is not None:
            # create checkpoint
            checkpoint_path = f"checkpoint/lstm_{check_point_name}.h5"
            checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=False,
                save_best_only=True,
                monitor="val_loss",
                verbose=verbose,
            )
            rlr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=patience, verbose=verbose
            )
            callbacks = [checkpoint, EarlyStopping(patience=patience), rlr]
        else:
            rlr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=patience, verbose=verbose
            )
            callbacks = [EarlyStopping(patience=patience), rlr]

        # Train model
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            use_multiprocessing=True,
            workers=8,
            verbose=verbose,
            callbacks=callbacks,
            shuffle=False,
        )

        # Load best model
        if check_point_name is not None:
            self.model.load_weights(f"checkpoint/lstm_{check_point_name}.h5")

    def plot_train_valid(self, metrics: str = "mse"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x= list(range(1, len(self.history.history[f"{metrics}"])+1)),
                                y = self.history.history[f"{metrics}"],
                                name = "Train"))
        fig.add_trace(go.Scatter(x= list(range(1, len(self.history.history[f"val_{metrics}"])+1)),
                                y = self.history.history[f"val_{metrics}"],
                                name = "val"))
        fig.update_layout(
            title="Performance Metric",
            xaxis_title="Epoch",
            yaxis_title=f"{metrics}")
        fig.show()
    
    def validation_pred_dataset(self):
        self.y_pred = self.model.predict(self.X_val).squeeze() 
        self.y_true = self.y_val.squeeze()
        
        return pd.DataFrame({"y": self.y_true, "y_pred": self.y_pred})
    
    def feature_importance_df(self):
        self.col = self.df.columns
        results = []
        self.baseline_mae = np.mean(np.abs(self.y_pred - self.y_true))
        results.append({'feature' : 'BASELINE','mae':self.baseline_mae})

        for k in tqdm(range(len(self.col) - 1)):
            # Shuffle feature K shuffle
            save_col = self.X_val[:,:,k].copy()
            np.random.shuffle(self.X_val[:,:,k])
                    
            # Compute OOF MAE with feature K shuffle
            oof_preds = self.model.predict(self.X_val).squeeze() 
            mae = np.mean(np.abs( oof_preds - self.y_true ))
            results.append({'feature':self.col[k],'mae':mae})
            self.X_val[:,:,k] = save_col

        df_fi = pd.DataFrame(results)
        self.df_fi = df_fi.sort_values('mae')

        return df_fi
    
    def feature_importance_plot(self):
        plt.figure(figsize=(10,20))
        plt.barh(np.arange(len(self.col)), self.df_fi['mae'])
        plt.yticks(np.arange(len(self.col)), self.df_fi['feature'].values)
        plt.title('LSTM Feature Importance',size=16)
        plt.ylim((-1,len(self.col)))
        plt.plot([self.baseline_mae,self.baseline_mae],[-1,len(self.col)+1], '--', color='orange',
                    label=f'Baseline OOF\nMAE={self.baseline_mae:.3f}')
        plt.xlabel(f'OOF MAE with feature permuted',size=14)
        plt.legend()
        plt.show()



class Test_LSTM:

    def reshape_dataset(self, df: pd.DataFrame):
        if "y" in df.columns:
            df = df.drop(columns=["y"]).assign(y=df["y"])
        else:
            raise KeyError("Not found target column y in dataset.")
        
        dataset = df.values.reshape(df.shape)
        return dataset
    
    def sequential_dataset(self, dataset: np.array, seq_len: int, steps: int):

        """
        :param seq_len : Length of sequences. (window size)
        :param steps : Length to predict.
        """

        X, y = list(), list()
        for i, _ in enumerate(dataset):
            idx_in = i + seq_len
            idx_out = idx_in + steps
            if idx_out > len(dataset):
                break
            seq_x = dataset[i:idx_in, :-1]
            seq_y = dataset[idx_out - 1 : idx_out, -1]
            
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)


    def test_pred_dataset(self, X_test : np.array, Y_test : np.array, model):
        self.y_test_pred = model.predict(X_test).squeeze() 
        self.y_test_true = Y_test.squeeze()
        
        return pd.DataFrame({"y": self.y_test_true, "y_pred": self.y_test_pred})
    
    def plot_result(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y = self.y_test_true,
                                name = "y"))
        fig.add_trace(go.Scatter(y = self.y_test_pred,
                                name = "y_pred"))
        fig.update_layout(xaxis_title="x")
        fig.show()
    





