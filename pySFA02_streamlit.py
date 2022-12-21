import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
import datetime
import scipy.special as sp
import streamlit as st
from PIL import Image


def RadialMean(img,x,y,fi1,fi2,n):
	#I - image to be analyzed
	#x1,x2 - coordinates of the centre of the spot
	#fi1,fi1 - angles in degrees to start and finish averaging
	#n - number of profiles to average
	(Y,X) = np.shape(img)
	R = np.min([x,X-x,Y,Y-y])-5
	#n=8
	#dfi=2*pi/n;
	dfi = (fi2-fi1)*np.pi/(180*n)
	Ir = np.zeros(R)
	b = np.zeros(R)
	for i in range(n):
		fi=fi1*np.pi/180+dfi*(i-1)
		#print(fi*180/pi)
		for j in range(R):
			xj=x+round(j*np.cos(fi))
			yj=y+round(j*np.sin(fi))
			b[j]=img[yj,xj]
		Ir = Ir + b
	Ir = Ir/n
	return Ir

def normalyze_profile(Ir):
    if isinstance(Ir, list):
        I1 = np.mean(Ir[0][-round(0.1*len(Ir[0])):-1])
        I0 = np.mean(Ir[0][0:5])
        for i in range(len(Ir)):
            Ir[i] = (Ir[i]-np.mean(Ir[i][-round(0.1*len(Ir[0])):-1])+(I1-I0))/np.abs(I1-I0)
    return Ir

def find_spot_center(images):
    if isinstance(images, list):
        img = np.zeros(np.shape(images[0]))
        for i in range(len(images)):
            img = img + images[i]
        img = img/len(images)
    else:
        img = images
    #img = cv2.GaussianBlur(img,(7,7),0)    
    Ix=np.sum(img,0)
    Iy=np.sum(img,1)
    Ix = savgol_filter(Ix, 101, 3) # window size 101, polynomial order 3
    Iy = savgol_filter(Iy, 101, 3) # window size 101, polynomial order 3
    x = np.argmin(Ix)
    y = np.argmin(Iy)
    return x,y
    
# def Read_FRAP_Images(DirPath):
#     names = list(os.listdir(DirPath))
#     images = []
#     for filename in names:
#         if filename!="Thumbs.db":
#             img = cv2.imread(os.path.join(DirPath,filename))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img = img.astype('float64')
#             images = images  + [img]
#     return images,names

# def Process_FRAP_Images(images,names):
#     ref_img = []
#     t_stamps = []
#     images1 = []
#     Ir = []
#     for img,filename in zip(images,names):
#         if "reference" in filename:
#             ref_img = img
        
#         if "time" in filename:
#             if filename.find("time")+12 < len(filename):
#                 stime = filename[filename.find("time")+4:
#                                  filename.find("time")+12]
#                 #print(stime)
#                 ts = time.mktime(datetime.datetime.strptime("2000-01-01 "+stime, "%Y-%m-%d %H-%M-%S").timetuple())
#                 images1 = images1 + [img]
#             else:
#                 ts=-1
#             t_stamps = t_stamps +[ts]
#         else:
#             t_stamps = t_stamps +[-1]
    
#     images = images1            
#     if ref_img==[]:
#         ref_img=np.zeros(np.shape(img))
        
#     if len(np.shape(ref_img))==2:    
#         for i in range(len(images)):
#             images[i] = images[i] - ref_img
#             images[i] = images[i] - np.min(images[i])
    
#     x,y = find_spot_center(images)
#     for i in range(len(images)):
#         Ir = Ir + [RadialMean(images[i],x,y,0,360,180)]
#     Ir = normalyze_profile(Ir)
  
#     return Ir,np.array(t_stamps),images,ref_img

def ReadFrapData(DirPath):
    names = list(os.listdir(DirPath))
    names.sort()
    images = []
    ref_img = []
    t_stamps = []
    Ir = []
    for filename in names:
        if filename!="Thumbs.db":
            img = cv2.imread(os.path.join(DirPath,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype('float64')
            if "reference" in filename:
                ref_img = img
            else:
                images = images  + [img]
                if "time" in filename:
                    if filename.find("time")+12 < len(filename):
                        stime = filename[filename.find("time")+4:
                                         filename.find("time")+12]
                        #print(stime)
                        ts = time.mktime(datetime.datetime.strptime("2000-01-01 "+stime, "%Y-%m-%d %H-%M-%S").timetuple())
                        t_stamps = t_stamps +[ts]
                else:
                    ts = time.mktime(datetime.datetime.strptime("1999-01-01 "+"00-00-01", "%Y-%m-%d %H-%M-%S").timetuple())
                    t_stamps = t_stamps +[ts]
    #st.write(names)                    
    if len(names) > 0:
        if ref_img==[]:
            ref_img=np.zeros(np.shape(img))
            
        if len(np.shape(ref_img))==2:    
            for i in range(len(images)):
                images[i] = images[i] - ref_img
                images[i] = images[i] - np.min(images[i])
        
        x,y = find_spot_center(images)
        for i in range(len(images)):
            Ir = Ir + [RadialMean(images[i],x,y,0,360,180)]
        Ir = normalyze_profile(Ir)
  
    return Ir,np.array(t_stamps),images,ref_img

def Cm(r, m, I, r0):
    def integrand(r,m,I,r0):
        return I*sp.jv(0, sp.jn_zeros(0, m)[m-1]*r/r0)*r
        
    return 2*np.trapz(integrand(r,m,I,r0),r)/(r0*sp.jv(1, sp.jn_zeros(0, m)[m-1]))**2    
    
    
def PolarDifSolve(r,t,D,I0,r0,Ir0,n):
    if np.isscalar(r):
        r = np.array([r])
    if np.isscalar(t):
        t = np.array([t])

    I0 = I0 - Ir0
    I=0#np.zeros(np.shape(I0))
    for i in range(n):
        m = i+1
        mu_0m = sp.jn_zeros(0, m)[m-1]
        #I1 = Cm(m, I0, r0)*sp.jv(0, mu_0m*r/r0)*np.exp(-D*(mu_0m/r0)**2*t)
        I1 = np.outer(np.exp(-D*(mu_0m/r0)**2*t), Cm(r, m, I0, r0)*sp.jv(0, mu_0m*r/r0))
        I = I + I1
    I = I + Ir0
    return I

def fitD(Ir,Dinit,pixel_size,r,t,r0,Ir0,n):
    Dmin = Dinit
    dD = 0.45*Dmin
    min_du0 = 10
    k = 5
    break_flag = 0
    Dspace = np.linspace(Dmin-int(k/2)*dD, Dmin+int(k/2)*dD, k)
    for i in range(100):
        du = []
        for D in Dspace:
            u = PolarDifSolve(r,t,D,Ir[0],r0,Ir0,n)
            du = du + [np.sum((u-Ir)**2)/(len(r)*len(t))]
        min_du = min(du)
        min_index = du.index(min_du)
        print(min_du)
        print(Dspace[min_index])
        print(Dspace)
        
        if Dspace[min_index] == Dmin:
            Dmin = Dmin + dD/8
        else:
            Dmin = Dspace[min_index]
            if abs(min_du-min_du0) < 0.005*min_du:
                break_flag = 1
                break
    
        if (min_index==0):
            dD = 0.45*Dmin
        if (min_index!=k-1):
            dD = 2*dD/(k-1)
        Dspace = np.linspace(Dmin-int(k/2)*dD, Dmin+int(k/2)*dD, k)
        min_du0 = min_du
                
            
    if break_flag == 0:
        print("Optimal D was not found after 100 iterations. Start the process again using ",Dmin," as initial value")
    
    print('The acuracity of 1% has been archived. Process stoped.')
    return D,dD

def t_text2float(t_text):
    a = t_text.strip("[] ")
    b=a.split(" ")
    d=[]
    for c in b:
        if c!='':
            d=d+[float(c)]
    
    return np.array(d)
    
def GetData(datasrc,place1):
    if datasrc == 'Upload files':
        DirPath = './data/uploads'
        if not os.path.exists(DirPath):
           os.makedirs(DirPath)
        with place1:
            with st.form("my-form", clear_on_submit=True):
                image_files = st.file_uploader("Upload Images", 
                                               type=['png', 'jpeg', 'jpg'],
                                               accept_multiple_files=True
                                               )
                submitted = st.form_submit_button("submit")
                
                rm_names = list(os.listdir('./data/uploads'))
                for name in rm_names:
                    os.remove(os.path.join('./data/uploads', name))
                
        if image_files is not None:
            for file in image_files:
                #ts = datetime.timestamp(datetime.now())
                imgpath = os.path.join('./data/uploads', file.name)
                with open(imgpath, mode="wb") as f:
                    f.write(file.getbuffer())
            #DirPath = './data/uploads'
            if len(image_files) > 0:
                RunModel(DirPath)
        
    if datasrc == 'Preloaded data': 
        path_dict = {
            'POPC bilayer':'./data/popc-b1',
            'POPC70-DOTAP30 bilayer':'./data/dotap30b6', 
            'Simulated':'./data/simulation'
            }
        with place1:
            option = st.selectbox(
                'Select a data set',
                tuple(path_dict.keys())
                )
        DirPath = path_dict[option]
        RunModel(DirPath)
        
    return DirPath
       
def RunModel(DirPath):   
  
    Ir,t,images,ref_img = ReadFrapData(DirPath)
    if len(t)!=0:
        t = t - t[0]
    else:
        st.write("time stamps required")
        t = np.zeros(np.shape(images))
    
    x,y = find_spot_center(images)
    
    col1, col2, col3 = st.columns([1,1,3])
     
    with col2:
        st.write("**Model parameters**")
        #st.write("____________________")
        st.write("*Pixel size:*")
        pixel_size = st.number_input('(um/pixel)',value=0.556, format="%f")
        st.write("*Initial guess of the diffusion coefficient:*")
        Dinit = st.number_input('D (um²/s)',value = 1.5)
        st.write("*The center of the bleached spot is detected in*")
        x = st.number_input('x (pixels)',value = x)
        y = st.number_input('y (pixels)',value = y)
        st.write("(correct if wrong)")
        #st.write("____________________")
        st.write("*The time stamps extracted from the file name:*")
        t_text = st.text_input('t (s)', str(t))
        st.write("(correct if wrong)")
        #st.write("____________________")
    
    Ir = []
    for i in range(len(images)):
        Ir = Ir + [RadialMean(images[i],x,y,0,360,180)]
    Ir = normalyze_profile(Ir)
    
    t = t_text2float(t_text)
    
    with col1:
        st.write("**Data set**")
                              
        for i in range(len(images)):
            fig = plt.figure()
            plt.imshow(images[i],cmap='gray')
            plt.title("time: "+str(t[i])+" s")
            plt.plot([x],[y],'cx')
            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)
            st.pyplot(fig)
        
        fig_ref = plt.figure()
        plt.imshow(ref_img,cmap='gray')
        plt.title("Reference")
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        st.pyplot(fig_ref)
        
    #pixel_size = 0.556#um in one pixel
    r = np.array(range(len(Ir[0])))*pixel_size
    #Dinit = 1.5
    r0 = r[-1]#282.448
    Ir0 = 1
    n = 50 
    
    D,dD = fitD(Ir,Dinit,pixel_size,r,t,r0,Ir0,n)
 
    fig6 = plt.figure()
    for I1 in Ir:
        plt.plot(r,I1,lw=4)

    u = PolarDifSolve(r,t,D,Ir[0],r0,Ir0,n)
    for i in range(len(t)):
        plt.plot(r,u[i,:],'k',lw=1)
    #plt.title("Mean radial intensity profiles")
    plt.xlabel("Distance from the center of the spot, um")
    plt.ylabel("Normalized intensity")
    with col3:
        st.write("**Results**")
        st.markdown("*Fitted diffusion coefficient:* **D = "+str(np.round(D,2))+" um<sup>2</sup>/s**", unsafe_allow_html=True)
        #st.write("*Fitted diffusion coefficient:* **D = "+str(np.round(D,2))+" um^2/s**")
        st.write("*Mean radial intensity profiles: colored lines - experiment, black - fitted*")
        st.pyplot(fig6)


def ShowHelp():
    st.subheader('Model description')   
    st.write('''The model allows to fit a function to experimental FRAP data and 
             find the diffusion coefficient as a fitting parameter.  ''')
    st.write('''FRAP data is 
    loaded to the model as a set of images showing recovery of 
    fluorescence in a circular spot and a reference image taken before 
    the start of bleaching.''')
    st.write('''File names should contain a key word "time"
    followed by a timestamp in a format "hh-mm-ss". For example: 
    popc-bleach1-time15-45-35.png. The reference image should contain the word "reference".
    ''')
    st.write('''To load your images choose "Upload files" from the select box 
             "Select data source" in the left side panel. A "file upload" widget 
             will appear down there. Browse your files or drag and drop them there.
             When uploading is done click the "Submit" button. The uploaded images are displayed in the "Data set" column 
             in the main screen.''')
    st.write('''In the "Model parameters" column you can see several parameters you can change.
             The very important one is **"Pixel size"** (number of micrometers in one pixel of your image).
             **You have to insert the value relevant to your images** to get the correct results''')
    st.write('''The next parameter is "Initial guess of the diffusion coefficient".
             How fast the model works depends on this parameter.''')
    st.write('''The model tries to find the center of the bleached spot. It is 
             shown with it's coordinates x and y in the parameter list. The detected center is
             shown in each image with a blue "x". You can enlarge the image by clicking
             double-arrow sign which appears at the top right corner 
             of every image when you move the mouse cursor over this corner. 
             You can correct the x and y values manually if the model makes a mistake finding the center.''')
    st.write('''As the last parameter in the list you can see the array of times 
             extracted from the file names. You can also change it.''') 
    st.write('''In the "Results" panel see the fitted diffusion coefficient and 
             a graph showing the profiles of fluorescence intensity extracted from
              the images (colored lines) and fitted lines (black)''')
    st.write('''The model fits a solution of the Dirichlet problem for diffusion (based on
             Fick's second law) in the polar coordinate system.''')
    st.latex(r'''\begin{cases}
             \frac{\partial u}{\partial t} = D(\frac{{\partial}^2 u}{{\partial r}^2} 
             + \frac{1}{r} \frac{\partial u}{\partial r})
             \\
             u_{t=0} = u_{exp1} (r)
             \\
             u_{r=r_0} = 1
             \end{cases}''')
    st.write('''where u is normalized intensity, D – diffusion coefficient, r – distance from the
             center, r0 – radius of the area under consideration, t – time, uexp1 – experimental 
             profile of intensity at the first point of time. Using the method of separation of
             variables solution for this problem can be found as:''')
    st.latex(r'''u(r,t) = \displaystyle\sum_{m=1}^\infty C_m J_0(\frac{\mu_m^{(0)}}{r_0}r)\exp{(-D(\frac{\mu_m^{(0)}}{r_0})^2t)}
             ''')
    st.write('''where Cm is:''')
    st.latex(r'''C_m = \displaystyle
                         \frac{ 2\intop_{0}^{r_1}
                               {u_{exp1}(r)J_0(\frac{\mu_m^{(0)}}{r_0}r)rdr}}
                            {r_0^{2}J_1(\mu_m^{(0)})^2}
                                       ''')
    st.write('''This function is fitted to the experimental intensity profiles using the diffusion coefficient D as a fitting parameter.''')


def ShowAboutFrap():
    st.subheader('Fluorescence recovery after photobleaching (FRAP)')
       
    col11, col12 = st.columns([1,1])
    with col12:
        image = Image.open('./docs/AboutFig01.png')
        st.image(image, caption='''Fig.1 - A schematic
                representation of FRAP in the lipid
                bilayer. (A) Initially, fluorescently
                labeled lipid molecules are
                distributed uniformly. (B) A spot is
                bleached by laser light. (C) Due to
                diffusion bleached molecules move
                outside of the bleached area, while
                emitting light molecules move
                inside. (D) Uniform distribution of
                bright and bleached molecules is set
                and the dark spot disappears. (Image from Wikipedia (https://en.wikipedia.org/wiki/Fluorescence_recovery_after_photobleaching))''')
    with col11:
        st.write('''Fluorescence recovery after photo-bleaching (FRAP) is a method of
        observing and quantifying diffusion. It is usually used for two
        dimensional systems such as molecular layers or cell membranes of living cells. 
        The idea of this method is presented in fig. 1''')
        st.write('''Fluorescently labeled molecules have to be presented in the system. A spot
        is usually bleached in the sample by laser or by switching to a higher
        magnification objective. The fluorophores in the area of this spot are bleached and
        produce no more fluorescence. Due to Brownian motion in the plane of the sample
        the bleached molecules move out of the bleached area while light emitting
        molecules move into the bleached spot. This process makes the edges of the spot
        less sharp and finally leads to disappearing of the spot.
        Observing these changes one can calculate the diffusion coefficient of the
        labeled molecules. Classical approach was introduced in the pioneering work of
        Axelrod [128]. It is based on several assumptions. Firstly, no diffusion is assumed
        during the bleaching process. Secondly, an assumption about the shape of the
        intensity profile of the bleached spot is done. Usually, a circular bleached area
        with the Gaussian profile for the fluorescence intensity is assumed (as the intensity
        in the laser beam usually has Gaussian distribution). Under these conditions
        fluorescence recovery in the center of the spot can be described by an analytical formula [129]''')        
        st.write('''Although this approach is widely used, the assumption it requires are not
        always valid. At least, special experimental conditions should be provided
        ensuring special shape of the initial intensity profile. To avoid these difficulties
        other methods of analysis of the FRAP data were introduced based on numerical
        simulations of the diffusion process [134]. Numerical methods allow high
        flexibility of experimental conditions as almost any shape of the bleached spot and
        any intensity profile can be analyzed. At the same time, numerical methods require
        higher computational power and sophisticated algorithms.''')
    st.write('''Meanwhile, analytical analysis of the recovery after photo-bleaching is
    possible even without assuming Gaussian or any other special form of initial
    profile of the fluorescence intensity. Solution of the Dirichlet problem based on
    Fick's second law is possible assuming circular symmetry of the bleached spot. 
    For the present model this problem was solved by the variable separation method. 
    For more information, see the "Model Help" section.''')
    st.write('''[1] D. Axelrod, D. E. Koppel, J. Schlessinger, E. Elson, and W. W. Webb, “Mobility
    measurement by analysis of fluorescence photobleaching recovery kinetics.,” Biophysical
    journal, vol. 16, no. 9, pp. 1055–69, Sep. 1976.''')
    st.write('''[2] T. Meyvis, S. D. Smedt, and P. V. Oostveldt, “Fluorescence recovery after
    photobleaching: a versatile tool for mobility and interaction measurements in
    pharmaceutical research,” Pharmaceutical, 1999.''')
    st.write('''[3] O. N. Irrechukwu and M. E. Levenston, “Improved Estimation of Solute Diffusivity
    Through Numerical Analysis of FRAP Experiments,” Cellular and Molecular
    Bioengineering, vol. 2, no. 1, pp. 104–117, Jan. 2009.
    ''')
    
#--------------Main-----------------------            
st.set_page_config(layout="wide")

#--Sidebar
colA, colB = st.sidebar.columns([1,1])
with colA:
    but_Help = st.button('Model help')
 
with colB:
    but_FRAP = st.button('What is FRAP?')
    
st.sidebar.title('⚙️Data source')
opt_data_src = st.sidebar.selectbox(
    'Select data source',
    ('Preloaded data', 'Upload files' )
    )
place1 = st.sidebar.empty()
st.sidebar.write('[This model on GitHub](https://github.com/Dimarfin/Simple-FRAP-analyser)')
st.sidebar.write('[This model in science](https://juser.fz-juelich.de/record/138467)')     
# -- End of Sidebar

col_a, col_b = st.columns([3,1])
with col_a:
    st.subheader('Simple FRAP analyser')
    st.write('Analyse your fluorescence recovery after photobleaching data')
with col_b:
    st.markdown("<b style='text-align: right; color: grey;'>Created by</b>", unsafe_allow_html=True)
    st.markdown("<b style='text-align: right; color: grey;'>Dzmitry Afanasekau</b>", unsafe_allow_html=True)
    #st.write('Created by')
    
st.markdown("""---""")

if but_Help:
    ShowHelp()
elif but_FRAP:
    ShowAboutFrap()
else:
    DirPath = GetData(opt_data_src,place1)

        




