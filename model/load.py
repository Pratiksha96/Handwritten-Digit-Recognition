from keras.models import model_from_json
import tensorflow as tf
from keras.optimizers import RMSprop


def init():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded Model from disk")
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # compile and evaluate loaded model
    loaded_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    #loss, accuracy = loaded_model.evaluate(X_test, y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return loaded_model, graph
