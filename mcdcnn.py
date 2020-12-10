# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from utils import save_logs
from utils import calculate_metrics


class Classifier_MCDCNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
        self.output_directory = output_directory
        if build == True:
            print("开始创建模型。。。")
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'#不允许超出边界

        if n_t < 60: # for ItalyPowerOndemand
            padding = 'same'#填充允许超出边界

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t,1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732,activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)
        ##############################
        print(model.summary())#ljw修改
        #############################

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=0.0005),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    def prepare_input(self,x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]#维度，一维的数据集是一

        for i in range(n_vars):
            new_x.append(x[:,:,i:i+1])#拼接为一条序列

        return  new_x

    def fit(self, x, y, x_test, y_test, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        mini_batch_size = 16
        #nb_epochs = 120
        nb_epochs = 120 #修改为数据训练波
        print("开始分割数据集x_val，用于计算test loss。。。")
        x_train, x_val, y_train, y_val = \
            train_test_split(x, y, test_size=0.33)#分割数据集，0.33的数据作为测试（随机选取）
        print("分割结束。。。")
        x_test = self.prepare_input(x_test)
        x_train = self.prepare_input(x_train)
        x_val = self.prepare_input(x_val)

        start_time = time.time()
        print("开始训练。。。")
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        print("训练结束。。。")
        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')
        print("开始预测数据。。。")
        y_pred = model.predict(x_test)
        print("预测结束。。。")
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()

    def predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(self.prepare_input(x_test))
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred