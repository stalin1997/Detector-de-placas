# Main.py

import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossiblePlate

# mdodulo de variables
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         

    if blnKNNTrainingSuccessful == False:                               
        print("\nerror: KNN traning was not successful\n")  
        return                                                          
    # Abrir Imagen

    imgOriginalScene  = cv2.imread("Placas_imagenes/auto005.jpg")               

    if imgOriginalScene is None:                            
        print("\nerror: image not read from file \n\n")  
        os.system("pause")                                  
        return                                              
    

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detectar placas

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect las placas en los carros

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # mostrar imagen

    if len(listOfPossiblePlates) == 0:                         
        print("\nno license plates were detected\n")  
    else:                                                       
                

                
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)          
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     
            print("\nno characters were detected\n\n")  
            return                                          
            
       

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # dibujar un retangulo alrededor de la placa

        print("\nPlaca numero = " + licPlate.strChars + "\n")  # escribir el numero de licencia ya predicho
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # escribir el numero de placa encima de la imagen

        cv2.imshow("imgOriginalScene", imgOriginalScene)               

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           

    

    cv2.waitKey(0)					# abrir ventanas con las imagenes

    return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)           

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # dibujar las 4 lineas en la placa
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)



def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      
    fltFontScale = float(plateHeight) / 30.0                   
    intFontThickness = int(round(fltFontScale * 1.5))           

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)             
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         

    if intPlateCenterY < (sceneHeight * 0.75):                                                  
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      
    else:                                                                                       
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))     
    

    textSizeWidth, textSizeHeight = textSize                

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # el area de la placa donde esta situada
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          

            # escribir el texto en la imagen
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)



if __name__ == "__main__":
    main()


















