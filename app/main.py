import sys
from flask import Flask, render_template, request, jsonify

import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns

################################
#Carrega modelos
################################

def numeric_to_string(y):
  if(y == 0):
    return 'CNV'
  elif(y == 1):
    return 'DME'
  elif(y == 2):
    return 'DRUSEN'
  elif(y == 3):
    return 'NORMAL'

class PerturbationCAM():

  #Passa a rede extratora de features, o modelo após a extração das features e o shape de input da rede inicial
  def __init__(self, featurenet, model, eh_nn = False, shape_input = (512, 512)):
    self.featurenet = featurenet
    self.model = model
    self.eh_nn = eh_nn
    self.shape_input = shape_input

  #Faz a predição de um batch de imagens (replica em 3 canais, extrai as features e faz o predict)
  #Note que se for uma rede neural o predict não é o mesmo método que um classificador com predict_proba
  def predict(self, batch_img):
    batch_img = np.repeat(batch_img[:, :, :, np.newaxis], 3, -1) #Faz virar imagens de 3 canais
    features_ref = self.featurenet.predict(batch_img)
    if(self.eh_nn):
      return self.model.predict(features_ref)
    else:
      return self.model.predict_proba(features_ref)

  #Faz a predição de uma só imagem e já retorna a classe predita em string
  def predict_label_one_img(self, img_array):
    batch_img = np.array([img_array])
    batch_img = np.repeat(batch_img[:, :, :, np.newaxis], 3, -1) #Faz virar imagens de 3 canais
    features_ref = self.featurenet.predict(batch_img)
    if(self.eh_nn):
      return numeric_to_string(np.argmax(self.model.predict(features_ref)[0]))
    else:
      return self.model.predict(features_ref)[0][0]
  
  #Faz os tratamentos necessários para a imagem de input 
  #(por enquanto só reescala para o shape da rede de extração de features e normaliza)
  def trata_imagem(self, img_array):
    return np.array(Image.fromarray(img_array).resize(self.shape_input[::-1]))/255.0

  #Recebe uma imagem só com um canal (preto e branco) normalizado entre 0 e 1
  #Retorna o mapa de calor para interpretabilidade
  def cria_mapa_calor(self, img_array, batch_cam = 32, num_random_mask = 10):
    shape_original = img_array.shape

    img_array = self.trata_imagem(img_array)

    #Faz uma predição da imagem sem alterações
    pred_ref = self.predict(np.array([img_array]))[0]

    #Faz um scan da imagem em conjunto de pixels dado pelo batch_cam
    i_max = int(img_array.shape[0]/batch_cam)
    j_max = int(img_array.shape[1]/batch_cam)
    perturbation_matrix = np.empty(shape = (i_max, j_max))
    for i in range(0, i_max):
      for j in range(0, j_max):
        #Para cada etapa do scan, alteramos os valores desses pixels da imagens nessa região num_random_mask vezes
        #E verificamos como essa alteração altera o valor predito
        img_temp = img_array.copy()
        batch = []
        for k in range(0, num_random_mask):
          img_temp[(batch_cam*i):(batch_cam*(i+1)), (batch_cam*j):(batch_cam*(j+1))] = np.random.rand(batch_cam, batch_cam)
          batch.append(img_temp)
        batch = np.array(batch)
        #Definimos a importância desse conjunto de pixels como a intensidade da perturbação média ocorrida na predição
        perturbation_matrix[i, j] = np.sum(np.sum((self.predict(batch) - pred_ref)**2, axis = 1)**0.5)/num_random_mask
    #Normaliza o valor das perturbações entre 0 e 1 (será nosso mapa de calor)
    perturbation_matrix = perturbation_matrix/np.max(perturbation_matrix)
    
    #Retornamos reescalando o mapa de calor para o shape da imagem original
    perturbation_matrix = np.array(Image.fromarray(perturbation_matrix).resize(shape_original[::-1]))
    return  np.where(perturbation_matrix < 0, 0, perturbation_matrix)

  def cria_mapa_calor_otimizado(self, img_array, batch_cam = 32, num_random_mask = 10, batch_predict = 2**7):
    shape_original = img_array.shape

    img_array = self.trata_imagem(img_array)

    #Faz uma predição da imagem sem alterações
    pred_ref = self.predict(np.array([img_array]))[0]

    trigger_predict = int(batch_predict/num_random_mask) #Trigger para sabe quando parar de acumular imagens na memória
    predicts = [] #Lista das predições feitas em batch
    count_predict = 0 #Contagem para saber quando atingimos o trigger de predição

    #Faz um scan da imagem em conjunto de pixels dado pelo batch_cam
    i_max = int(img_array.shape[0]/batch_cam)
    j_max = int(img_array.shape[1]/batch_cam)

    batch = []
    for i in range(0, i_max):
      for j in range(0, j_max):

        #Para cada etapa do scan, alteramos os valores desses pixels da imagens nessa região num_random_mask vezes
        #E verificamos como essa alteração altera o valor predito
        img_temp = img_array.copy()
        for k in range(0, num_random_mask):
          img_temp[(batch_cam*i):(batch_cam*(i+1)), (batch_cam*j):(batch_cam*(j+1))] = np.random.rand(batch_cam, batch_cam)
          batch.append(img_temp)

        #Checa se atingimos o trigger para fazer a predição do batch
        count_predict = count_predict + 1
        if(count_predict % trigger_predict == 0):
            batch = np.array(batch)
            predicts.append(self.predict(batch))
            batch = []
    #Faz a predição do último batch (o que sobrou após o loop acabar)
    if(len(batch) != 0):
      batch = np.array(batch)
      predicts.append(self.predict(batch))
      batch = []

    #Calcula a distância da predição de referência
    diff_preds = np.sum((np.concatenate(predicts) - pred_ref)**2, axis = 1)**0.5 
    
    #Calcula a média da distância da predição de referência para cada random de oclusão feito
    diff_preds = np.sum((np.concatenate(predicts) - pred_ref)**2, axis = 1)**0.5 
    diff_preds = np.mean(diff_preds.reshape(int(diff_preds.size/num_random_mask), num_random_mask), axis = 1)

    #Definimos a importância desse conjunto de pixels como a intensidade da perturbação média ocorrida na predição
    perturbation_matrix = diff_preds.reshape(i_max, j_max) #Recontrói o shape da matriz da imagem

    #Normaliza o valor das perturbações entre 0 e 1 (será nosso mapa de calor)
    perturbation_matrix = perturbation_matrix/np.max(perturbation_matrix)
    
    #Retornamos reescalando o mapa de calor para o shape da imagem original
    perturbation_matrix = np.array(Image.fromarray(perturbation_matrix).resize(shape_original[::-1]))
    return np.where(perturbation_matrix < 0, 0, perturbation_matrix)

#Dá cor para nosso mapa de calor
#Se use_heatmap_alpha = False retorna um mapa de calor que vai do azul para o vermelho
#Se use_heatmap_alpha = True retorna um mapa sempre vermelho mas com variação de alpha
def colorir_heatmap(img_heatmap, use_heatmap_alpha, alpha = 0.4):
    if(use_heatmap_alpha == False):
      jet = cm.get_cmap("jet") #Objeto para colorir o heatmap
      jet_colors = jet(np.arange(256))[:, :3]
      heatmap = np.uint8(255 * img_heatmap) # Reescala o heatmap entre 0 e 255 (inteiro)
      jet_heatmap = jet_colors[heatmap]

      #Define um alpha constante
      heatmap_alpha = img_heatmap
      jet_heatmap_nova = np.empty(shape = (img_heatmap.shape[0], img_heatmap.shape[1], 4))
      jet_heatmap_nova[:, :, :3] = jet_heatmap
      jet_heatmap_nova[:, :, 3] = alpha
      jet_heatmap = jet_heatmap_nova

    else:
      N = 256
      vals = np.ones((N, 4))
      vals[:, 0] = 1.0
      vals[:, 1] = 0.0
      vals[:, 2] = 0.0
      newcmp = ListedColormap(vals)
      jet_colors = newcmp(np.arange(256))[:, :3]
      heatmap = np.uint8(255 * img_heatmap) # Reescala o heatmap entre 0 e 255 (inteiro)
      jet_heatmap = jet_colors[heatmap]

      #Usa a própria intensidade da perturbação como alpha
      heatmap_alpha = img_heatmap*alpha
      jet_heatmap_nova = np.empty(shape = (img_heatmap.shape[0], img_heatmap.shape[1], 4))
      jet_heatmap_nova[:, :, :3] = jet_heatmap
      jet_heatmap_nova[:, :, 3] = heatmap_alpha
      jet_heatmap = jet_heatmap_nova

    return (jet_heatmap * 255).astype(np.uint8)

path_models = 'modelos/'

# Versão pública: o extrator (ResNet50V2 congelada + GlobalAveragePooling2D, pesos da ImageNet) é montado pelo Keras
# em vez de lido do featurenet.h5 de 94 MB, como o grupo já fazia na versão do Heroku. A arquitetura é a do featurenet.json.
loadnet = tf.keras.applications.ResNet50V2(input_shape = (512, 512, 3), include_top = False, weights = 'imagenet')
loadnet.trainable = False
featurenet = tf.keras.Sequential([loadnet, tf.keras.layers.GlobalAveragePooling2D()])

with open(path_models + 'model_baseline_nn.json', 'r') as json_file:
    model_baseline_nn_json = json_file.read()
model_baseline_nn = tf.keras.models.model_from_json(model_baseline_nn_json)
model_baseline_nn.load_weights(path_models + 'weights_baseline.h5')

pert_cam = PerturbationCAM(featurenet, model_baseline_nn, eh_nn = True) 

################################
#
################################

app = Flask(__name__)

@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    if request.method == 'POST':
        file_img = request.files['file']
        
        #Converte imagem para o encoder base64 (fácil para enviar para o HTML como string)
        img_base64 = base64.b64encode(file_img.read())
        #img_base64_string_html = 'data:image/jpeg;base64,' + str(img_base64)[2:-1]
        
        #lê imagem e converte para grayscale se necessário
        img_array = np.array(Image.open(io.BytesIO(base64.b64decode(img_base64))))
        if(len(img_array.shape) == 3 and img_array.shape[2] == 3):
            img_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        probs = pert_cam.predict(np.array([pert_cam.trata_imagem(img_array)]))[0]
        predito = pert_cam.predict_label_one_img(pert_cam.trata_imagem(img_array))
        img_heatmap = pert_cam.cria_mapa_calor_otimizado(img_array, batch_cam = 64, num_random_mask = 2, batch_predict = 128)
        heatmap_colorido = colorir_heatmap(img_heatmap, use_heatmap_alpha = True, alpha = 0.75)
        #print(predito, file = sys.stderr)
        
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img_array, cmap = 'gray')
        ax.set_axis_off()
        img_input_iobytes = io.BytesIO()
        plt.savefig(img_input_iobytes, format = 'jpg', bbox_inches='tight', pad_inches=0)
        img_input_iobytes.seek(0)
        plt.close()
        
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img_heatmap, cmap = cm.get_cmap("jet"))
        ax.set_axis_off()
        img_mask_iobytes = io.BytesIO()
        plt.savefig(img_mask_iobytes, format = 'jpg', bbox_inches='tight', pad_inches=0)
        img_mask_iobytes.seek(0)
        plt.close()
        
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img_array, cmap = 'gray')
        ax.imshow(heatmap_colorido)
        ax.set_axis_off()
        img_inputmask_iobytes = io.BytesIO()
        plt.savefig(img_inputmask_iobytes, format = 'jpg', bbox_inches='tight', pad_inches=0)
        img_inputmask_iobytes.seek(0)
        plt.close()
        
        paleta_cores = sns.color_palette("colorblind")
        classes = np.array(['CNV', 'DME', 'DRUSEN', 'NORMAL'])
        index_sort = np.argsort(probs)[::-1]
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(1, 1, figsize = (6, 4))
            ax.barh(classes[index_sort], probs[index_sort], align='center', color = paleta_cores[0])
            ax.set_title('Probabilidades:')
            img_probs_iobytes = io.BytesIO()
            plt.savefig(img_probs_iobytes, format = 'jpg', bbox_inches='tight', pad_inches=0)
            img_probs_iobytes.seek(0)
            plt.close()
        
        #Converte as imagens do matplotlib para o encoder base64 para enviar para o HTML
        
        img_input_base64 = base64.b64encode(img_input_iobytes.read())
        img_input_base64_html = 'data:image/jpeg;base64,' + str(img_input_base64)[2:-1]
        
        img_mask_base64 = base64.b64encode(img_mask_iobytes.read())
        img_mask_base64_html = 'data:image/jpeg;base64,' + str(img_mask_base64)[2:-1]
        
        img_inputmask_base64 = base64.b64encode(img_inputmask_iobytes.read())
        img_inputmask_base64_html = 'data:image/jpeg;base64,' + str(img_inputmask_base64)[2:-1]
        
        img_probs_base64 = base64.b64encode(img_probs_iobytes.read())
        img_probs_base64_html = 'data:image/jpeg;base64,' + str(img_probs_base64)[2:-1]
        
        return jsonify(img_input = img_input_base64_html, img_mask = img_mask_base64_html, img_inputmask = img_inputmask_base64_html,
                       predito = predito, img_probs = img_probs_base64_html)

@app.route("/")
def index():
    f = open("count.txt", "r")
    count = int(f.read())
    f.close()

    count += 1

    f = open("count.txt", "w")
    f.write(str(count))
    f.close()

    return render_template("index.html", count = count)

if __name__ == "__main__":
    app.run(debug = True)