import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

#%%
skin_df=pd.read_csv("metadata.csv")

skin_df.head()
skin_df.info()
sns.countplot(x="dx", data=skin_df)

#%%

data_folder_name="images/"

ext=".jpg"



skin_df["path"] = [ data_folder_name  + i + ext for i in skin_df["image_id"] ]

skin_df["image"] = skin_df["path"].map( lambda x: np.asarray(Image.open(x).resize((100,75))))

skin_df["dx_idx"]=pd.Categorical(skin_df["dx"]).codes

skin_df.to_pickle("skin_df.pkl")

#%%

skin_df = pd.read_pickle("skin_df.pkl")

#%%

x_train = np.asarray(skin_df["image"].tolist())
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train = (x_train - x_train_mean) / x_train_std

y_train = to_categorical(skin_df["dx_idx"], num_classes=7)

#%%

input_shape = (75,100,3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same", input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu", padding = "Same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation="softmax"))
model.summary()

optimizer = Adam(lr = 0.001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

epochs = 5
batch_size = 25

history = model.fit(x = x_train, y=y_train, batch_size = batch_size, epochs= epochs, verbose=1, shuffle=True)

model.save("my_model1.h5")

#%%

model1=load_model("my_model1.h5")
model2=load_model("my_model2.h5")

#%%

index=5

y_pred = model1.predict(x_train[index].reshape(1,75,100,3))

y_pred_class = np.argmax(y_pred,axis=1)

#%% Deri kanseri sınıflandırma GUI

window=tk.Tk()
window.geometry("1080x640")
window.wm_title("Deri kanseri sınıflandırma")

# global değişkenler
img_name=""
count=0
img_jpg=""

# frame ler
frame_left=tk.Frame(window,width=540,height=640,bd="2")
frame_left.grid(row=0,column=0)

frame_right=tk.Frame(window, width=540, height=640, bd="2")
frame_right.grid(row=0,column=1)

frame1=tk.LabelFrame(frame_left, text="Resim",width=540, height=500)
frame1.grid(row=0,column=0)

frame2=tk.LabelFrame(frame_left, text="Model ve Kayıt",width=540, height=140)
frame2.grid(row=1,column=0)

frame3=tk.LabelFrame(frame_right, text="Özellikler",width=270, height=640)
frame3.grid(row=0,column=0)

frame4=tk.LabelFrame(frame_right, text="Sonuçlar",width=270, height=640)
frame4.grid(row=0,column=1,padx=10)

def imageResize(img):
    basewidth=500
    wpercent=(basewidth/float(img.size[0]))
    hsize=int((float(img.size[1])*float(wpercent)))
    img=img.resize((basewidth,hsize),Image.ANTIALIAS)
    return img


def openImage():
    global img_name
    global count
    global img_jpg
    
    count +=1
    if count !=1:
        messagebox.showinfo(title="Uyarı", message="Bir resim seçiniz")
    else:
        img_name=filedialog.askopenfilename(initialdir="",title="Bir resim seçiniz")
        
        img_jpg=img_name.split("/")[-1].split(".")[0]
        
        tk.Label(frame1,text=img_jpg,bd=3).pack(pady=10)
        
        img=Image.open(img_name)
        
        img=imageResize(img)
        
        img = ImageTk.PhotoImage(img)
        
        panel=tk.Label(frame1,image=img)
        panel.image=img
        panel.pack(padx=15,pady=10)
        
        # resim özellikleri
        data=pd.read_csv("metadata.csv")
        kanser=data[data.image_id==img_jpg]
        
        for i in range(kanser.size):
            x=0.4
            y=(i/10)/2
            tk.Label(frame3,font=("Times",12), text=str(kanser.iloc[0,i])).place(relx=x,rely=y)


menubar = tk.Menu(window)
window.config(menu=menubar)
file=tk.Menu(menubar)
menubar.add_cascade(label="Dosya",menu=file)
file.add_command(label="Aç",command=openImage)

# frame 3

columns=["lession_id","image_id","dx","dx_type","age","gender","localization"]

for i in range (len(columns)):
    x=0.1
    y=(i/10)/2
    tk.Label(frame3,font=("Times",12), text=str(columns[i])+" : ").place(relx=x,rely=y)
    
    
def classification():
    
    if img_name != "" and models.get() != "":
        # model seçimi
        if models.get() == "model1":
            
            classification_model= model1
            
        else:
            classification_model= model2
            
        z = skin_df[skin_df.image_id == img_jpg]
        z = z.image.values[0].reshape(1,75,100,3)
        
        # standartizasyon
        z = (z-x_train_mean)/x_train_std
        h = classification_model.predict(z)[0]
        h_index = np.argmax(h)
        predicted_kanser=list(skin_df.dx.unique())[h_index]
        
        for i in range(len(h)):
            x=0.5
            y = (i/10)/2
            
            if i != h_index:
                tk.Label(frame4, text=str(h[i])).place(relx=x,rely=y)
            
            else:
                tk.Label(frame4,bg="green", text=str(h[i])).place(relx=x,rely=y)
                
        if chvar.get()==1:
            val=entry.get()
            entry.config(state="disable")
            path_name=val+".txt"
            
            save_txt=img_name + "--" + str(predicted_kanser)
            
            text_file=open(path_name,"w")
            text_file.write(save_txt)
            text_file.close()
        else:
            print("Kayıt etme seçilmedi")
            
    else:
        messagebox.showinfo(title = "Uyarı", message = "İlk olarak resim ve model seçimi yapın!")
        tk.Label(frame3,text="İlk olarak resim ve model seçimi yapın!").place(relx=0.1,rely=0.6)
        
        
    
classify_button = tk.Button(frame3, bg="green", bd=4, font=("Times",13), activebackground="orange",text="Sınıflandır", command=classification)

classify_button.place(relx=0.1,rely=0.5)

# frame 4

labels=skin_df.dx.unique()

for i in range(len(columns)):
    x=0.1
    y=(i/10)/2
    tk.Label(frame4,font=("Times",12), text=str(labels[i])+" : ").place(relx=x,rely=y)

# frame 2

model_selection_label= tk.Label(frame2, text="Sınıflandırma modeli seçiniz : ")
model_selection_label.grid(row=0,column=0,padx=5)

models=tk.StringVar()
model_selection=ttk.Combobox(frame2,textvariable=models, values=("model1","model2"), state="readonly")
model_selection.grid(row=0,column=1,padx=5)

# checkbox
chvar=tk.IntVar()
chvar.set(0)
xbox=tk.Checkbutton(frame2,text="Sınıflandırma sonucunu kaydet",variable=chvar)
xbox.grid(row=1,column=0,pady=5)

# entry
entry=tk.Entry(frame2,width=23)
entry.insert(string="Kayıt ismi ...",index=0)
entry.grid(row=1,column=1)

window.mainloop()








































































