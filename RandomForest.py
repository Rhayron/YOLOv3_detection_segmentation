from scipy import ndimage as nd
import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
import numpy as np
import Metricas
import cv2 as cv
import pickle
import math
import os

#Predict da imagem
def create_features(imagemTempErrada8, img, img_gray, cropped, label, train=True):
    dfImg = pd.DataFrame()
    
    Pixels_arr = []
    img_array = img
    img = img_gray
    img = img.reshape(-1)
    Pixels_arr.append(img)
        
    Pixels_arr = np.array(Pixels_arr)
    Pixels_arr = Pixels_arr.flatten()
    dfImg['Imagem Original'] = Pixels_arr

    #Feature da temperatura de cada píxel
    Pixels_arr = []
    img = cropped
    img = img.reshape(-1)
    Pixels_arr.append(img)

    Pixels_arr = np.array(Pixels_arr)
    Pixels_arr = Pixels_arr.flatten()

    dfImg['Temperaturas'] = Pixels_arr
        
    img2 = img_gray.reshape(-1)
    #Define os parametros e realiza a extração das gaborFeatures
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta/4.*np.pi 
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi/4):
                for gamma in(0.05, 0.5):
                    gabor_label = 'Gabor' + str(num)
                    ksize = 5
                    kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv.CV_32F)
                    kernels.append(kernel)
                    #print(img_gray.dtype,img_gray.shape)
                    fimg = cv.filter2D(img2, cv.CV_8UC3, kernel)
                    #print(fimg.dtype,fimg.shape)
                    filtered_img = fimg.reshape(-1)
                    dfImg[gabor_label] = filtered_img
                    num += 1
        
    #Transforma a imagem em uma matriz unidimensional e seta essa matriz para o data frame como Imagem Original
    img2 = img.reshape(-1)
        
    #Gaussian filtro
    Gaussian_arr = []
    img = img_gray
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    Gaussian_arr.append(gaussian_img1)

    Gaussian_arr = np.array(Gaussian_arr)
    Gaussian_arr = Gaussian_arr.flatten()
    dfImg['Gaussian filtro s3'] = Gaussian_arr
        
    #Gaussian filtro
    Gaussian_arr = []
    img = img_gray
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    Gaussian_arr.append(gaussian_img3)

    Gaussian_arr = np.array(Gaussian_arr)
    Gaussian_arr = Gaussian_arr.flatten()
    dfImg['Gaussian filtro s7'] = Gaussian_arr

    #MEDIAN filtro
    MEDIAN_arr = []
    img = img_gray
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    MEDIAN_arr.append(median_img1)

    MEDIAN_arr = np.array(MEDIAN_arr)
    MEDIAN_arr = MEDIAN_arr.flatten()
    dfImg['MEDIAN filtro'] = MEDIAN_arr
        
    #VARIANCE with size=3
    variance_arr =[]
    img = img_gray
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    variance_arr.append(variance_img1)
        
    variance_arr = np.array(variance_arr)
    variance_arr = variance_arr.flatten()    
    dfImg['Variance s3'] = variance_arr
        
    features = dfImg
    labels = None
        
    #display(dfImg)

    return features


#Método para setar as features da imagem
def compute_prediction(imagemTempErrada8, imagemTempErrada16, img_gray, cropped, load_model):

    features = create_features(imagemTempErrada8, img_gray, img_gray, cropped, label=None, train=False)
    predictions = load_model.predict(features)
    pred_size = int(math.sqrt(features.shape[0]))
    height, width = imagemTempErrada16.shape
    inference_img = predictions.reshape(height, width)

    return inference_img


#Método principal
def main(dirImgs8Bit, dirImgs16Bit, dirTxt, dirSave, worksheet, indLinha):

    #Lê todos os parâmetros da pasta e cria duas listas dos parametros
    parameters = []
    tempMinima = []
    tempDelta = []
    for txt in os.listdir(dirTxt):
        if txt.endswith(".txt"): 
            file_path = os.path.join(dirTxt, txt)
            with open (file_path, 'rt') as myfile:                                  #Define e le o diretório dos parametros
                for myline in myfile:                                               #Intera em cada linha do documento
                    parameters.append(myline.rstrip('\n'))                          #Adiciona o conteudo de cada linha na lista
            tempMinima.append(float(parameters[3]))                                 #Define a temperatura minima da imagem
            tempDelta.append(float(parameters[5]))                                  #Define a diferença de temperatura da imagem
            parameters = []
    
    #Utiliza o pickle para fazer a conversão do modelo encontrado
    filename = 'Modelo de segmentação RF - PontoQuente - 40Imagens - 38 Features - PC'
    load_model = pickle.load(open(filename, 'rb'))

    #Lendo as imagens na pasta
    ind = 0
    for imagem in os.listdir(dirImgs16Bit):
        imagemTempErrada8 = imread(os.path.join(dirImgs8Bit, imagem))
        imagemTempErrada16 = imread(os.path.join(dirImgs16Bit, imagem))
        cropped = ((imagemTempErrada16 * tempDelta[ind])/65535)+tempMinima[ind]

        img_gray = cv.cvtColor(imagemTempErrada8, cv.COLOR_BGR2GRAY)

        imgSegmentada = compute_prediction(imagemTempErrada8, imagemTempErrada16, img_gray, cropped, load_model)

        #Extrais as metricas da segmentação
        Metricas.main(imagem, cropped, imgSegmentada, worksheet, indLinha, 'RF')
        
        #Salva a imagem segmentada
        imgSave = os.path.join(dirSave, ('RF'+imagem))
        plt.imsave(imgSave, imgSegmentada, cmap='gray')

        #Indice da lista de parâmetros
        ind += 1
        indLinha += 1

    return indLinha


if __name__ == "__main__":
    main()
