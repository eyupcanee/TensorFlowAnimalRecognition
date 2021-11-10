
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from PyQt5 import QtCore, QtGui, QtWidgets

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input


model = VGG16()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(503, 428)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photo_label = QtWidgets.QLabel(self.centralwidget)
        self.photo_label.setGeometry(QtCore.QRect(10, 38, 291, 241))
        self.photo_label.setText("")
        self.photo_label.setPixmap(QtGui.QPixmap("ArGeA/happy_dog.jpg"))
        self.photo_label.setScaledContents(True)
        self.photo_label.setObjectName("photo_label")
        self.photoText = QtWidgets.QLabel(self.centralwidget)
        self.photoText.setGeometry(QtCore.QRect(10, 10, 141, 31))
        self.photoText.setObjectName("photoText")
        self.open_image = QtWidgets.QPushButton(self.centralwidget)
        self.open_image.setGeometry(QtCore.QRect(10, 320, 141, 51))
        self.open_image.setObjectName("open_image")
        self.open_image.clicked.connect(self.open_file)
        self.predict_animal = QtWidgets.QPushButton(self.centralwidget)
        self.predict_animal.setGeometry(QtCore.QRect(160, 320, 141, 51))
        self.predict_animal.setObjectName("predict_animal")
        self.predict_animal.clicked.connect(self.preditct_animal_func)
        self.result_text = QtWidgets.QLabel(self.centralwidget)
        self.result_text.setGeometry(QtCore.QRect(310, 40, 181, 241))
        self.result_text.setText("")
        self.result_text.setObjectName("result_text")
        self.download_data = QtWidgets.QPushButton(self.centralwidget)
        self.download_data.setGeometry(QtCore.QRect(310, 320, 141, 51))
        self.download_data.setObjectName("download_data")
        self.download_data.clicked.connect(self.download_data_func)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 503, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.img_path = ""
        self.downloaded_data = False

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def open_file(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget,"Open Image",os.getenv("Desktop"))

        try:
            self.img_path = str(file_path[0])
            self.photo_label.setPixmap(QtGui.QPixmap(self.img_path))
        except:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("We Can't Imported This Photo.Please select another one.")
            msgBox.setWindowTitle("Error")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msgBox.show()
            msgBox.exec_()

    def preditct_animal_func(self):

        if self.downloaded_data == True:
            if self.img_path != "":

                image = image_utils.load_img(self.img_path,target_size=(224,224))
                original_shape = mpimg.imread(self.img_path).shape
                image = image_utils.img_to_array(image)
                image = image.reshape(1,224,224,3)
                image = preprocess_input(image)
                processed_shape = image.shape

                predictions = model.predict(image)


                if 69 <= np.argmax(predictions) <= 79:
                    self.result_text.setText("Original image shape:{}\nProcessed image shape:{}\nIt's looks like a BUG!".format(original_shape,processed_shape))
                elif 80 <= np.argmax(predictions) <= 100:
                    self.result_text.setText("Original image shape:{}\nProcessed image shape:{}\nIt's looks like a BIRD!".format(original_shape,processed_shape))
                elif 101 <= np.argmax(predictions) <= 106:
                    self.result_text.setText("Original image shape:{}\nProcessed image shape:{}\nIt's looks like a LAND ANIMAL!".format(original_shape,processed_shape))
                elif  107 <= np.argmax(predictions) <= 150:
                    self.result_text.setText("Original image shape:{}\nProcessed image shape:{}\nIt's looks like a FISH!".format(original_shape,processed_shape))
                elif 151 <= np.argmax(predictions) <= 268:
                    self.result_text.setText("Original image shape:{}\nProcessed image shape:{}\nIt's looks like a DOG!".format(original_shape,processed_shape))
                elif 281 <= np.argmax(predictions) <= 285:
                    self.result_text.setText("Original image shape:{}\nProcessed image shape:{}\nIt's looks like a CAT!".format(original_shape,processed_shape))
                else:
                    self.result_text.setText("This image not yet learned")
            else:
                msgBox = QtWidgets.QMessageBox()
                msgBox.setText("You Haven't Opened A Image!You Should Press Open Image And Select A Image!")
                msgBox.setWindowTitle("Error")
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msgBox.show()
                msgBox.exec_()
        else:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("You Haven't Downloaded Data Yet.You Should Press Download Data And Waiting Till End")
            msgBox.setWindowTitle("Error")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msgBox.show()
            msgBox.exec_()

    def download_data_func(self):
        self.result_text.setText("Downloading Data....")
        if self.downloaded_data == False:
            model = VGG16(weights="imagenet")
            self.result_text.setText("Downloaded Data!")
            self.downloaded_data = True
        else:
            self.result_text.setText("Downloaded Data!")


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Animal Prediction"))
        self.photoText.setText(_translate("MainWindow", "Your Photo :"))
        self.open_image.setText(_translate("MainWindow", "Open Image"))
        self.predict_animal.setText(_translate("MainWindow", "Predict Animal"))
        self.download_data.setText(_translate("MainWindow", "Download Data"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

