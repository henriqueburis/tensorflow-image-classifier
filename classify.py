import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

# definir locais de memória para gráficos e rótulos treinados em train.sh ##

# Desativar avisos de compilação de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

image_path = sys.argv[1]
# Especificação no console como um argumento após a chamada  


# Arquivo de imagem
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Obtém rótulos do arquivo na matriz
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("tf_files/retrained_labels.txt")]
# !! os rótulos estão sempre em suas próprias linhas -> nenhuma alteração em retrain.py necessária -> apresentação errada no editor do windows !!
				   
# leia gráfico, foi treinado em train.sh -> call retrain.py
with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
 
    graph_def = tf.GraphDef()	## O graph-graph_def é uma cópia salva de um gráfico do TensorFlow; Inicialização objeto
    graph_def.ParseFromString(f.read())	# Analisar dados de buffer de protocolo serializados na variável
    _ = tf.import_graph_def(graph_def, name='')	# importar um buffer de protocolo serializado TensorFlow GraphDef, extrair objetos no GraphDef como tf.Tensor 
	
	
with tf.Session() as sess:

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	# return: Tensor("final_result:0", shape=(?, 4), dtype=float32); stringname definiert in retrain.py, 

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    # retorna valores de previsão na matriz :

         
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	# classificado pelo mais alto


	# saida
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))


        
