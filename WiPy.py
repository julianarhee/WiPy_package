
import cv2
import os
import glob
import numpy as np
from scipy import misc,interpolate,stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors
import time
import shutil

__version__ = '0.1.0'

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# MISCELANOUS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_theta_array(szY,szX):
    #Cesar Echavarria 11/2016


    x = np.linspace(-1, 1, szX)
    y = np.linspace(-1, 1, szY)
    xv, yv = np.meshgrid(x, y)

    [radius,theta]=cart2pol(xv,yv)
    theta = np.rad2deg(theta)
    
    return theta

def array2cmap(X):
    N = X.shape[0]

    r = np.linspace(0., 1., N+1)
    r = np.sort(np.concatenate((r, r)))[1:-1]

    rd = np.concatenate([[X[i, 0], X[i, 0]] for i in xrange(N)])
    gr = np.concatenate([[X[i, 1], X[i, 1]] for i in xrange(N)])
    bl = np.concatenate([[X[i, 2], X[i, 2]] for i in xrange(N)])

    rd = tuple([(r[i], rd[i], rd[i]) for i in xrange(2 * N)])
    gr = tuple([(r[i], gr[i], gr[i]) for i in xrange(2 * N)])
    bl = tuple([(r[i], bl[i], bl[i]) for i in xrange(2 * N)])


    cdict = {'red': rd, 'green': gr, 'blue': bl}
    return colors.LinearSegmentedColormap('my_colormap', cdict, N)
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# GENERAL PURPOSE WiPy FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #


def normalize_stack(frameStack):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("No Arguments Passed!")

    frameStack=np.true_divide((frameStack-np.min(frameStack)),(np.max(frameStack)-np.min(frameStack)))
    return frameStack


def get_frame_times(planFolder):
    #Cesar Echavarria 10/2016
    
    # READ IN FRAME TIMES FILE
    planFile=open(planFolder+'frameTimes.txt')
    frameTimes=[]
    frameCond=[]
    for line in planFile:
        x=line.split()
        frameCond.append(x[3])
        frameTimes.append(x[4])

    frameTimes.pop(0)#take out header
    frameCond.pop(0)#take out header
    frameTimes=np.array(map(float,frameTimes))
    frameCond=np.array(map(int,frameCond))
    frameCount=len(frameTimes)
    return frameTimes,frameCond,frameCount

def get_block_paradigm(planFolder):
    #Cesar Echavarria 10/2016
    
    # READ IN BLOCK PARADIGM
    planFile=open(planFolder+'blockParadigm.txt')
    blockCond=[]
    blockStartT=[]

    for line in planFile:
        x=line.split()
        blockCond.append(x[1])
        blockStartT.append(x[2])

    blockStartT.pop(0)#take out header
    blockCond.pop(0)#take out header
    blockStartT=np.array(map(float,blockStartT))
    blockCond=np.array(map(int,blockCond))
    nBlocks=len(blockCond)
    return blockCond,blockStartT,nBlocks

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# STRUCTURAL INFO FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_surface(sourceRoot,targetRoot,sessID):
    #Cesar Echavarria - 10/2016
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
        
    #DEFINE DIRECTORY
    sourceDir=glob.glob(sourceRoot+sessID+'_surface*')
    surfDir=sourceDir[0]+'/Surface/'
    outDir=targetRoot+'/Sessions/'+sessID+'/Surface/';
    if not os.path.exists(outDir):
        os.makedirs(outDir) 
    picList=glob.glob(surfDir+'*.tiff')
    nPics=len(picList)


    # READ IN FRAMES
    imFile=surfDir+'frame0.tiff'
    im0=cv2.imread(imFile,-1)
    sz=im0.shape

    allFrames=np.zeros(sz+(nPics,))
    allFrames[:,:,0]=im0
    for pic in range(1,nPics):
        imFile=surfDir+'frame'+str(pic)+'.tiff'
        im0=cv2.imread(imFile,-1)
        allFrames[:,:,pic]=im0

    #AVERAGE OVER IMAGES IN FOLDER
    imAvg=np.mean(allFrames,2)

    # #SAVE IMAGE

    outFile=outDir+'frame0.tiff'
    cv2.imwrite(outFile,np.uint16(imAvg))#THIS FILE MUST BE OPENED WITH CV2 MODULE

    outFile=outDir+'16bitSurf.tiff'
    imAvg=np.true_divide(imAvg,2**12)*(2**16)
    cv2.imwrite(outFile,np.uint16(imAvg))#THIS FILE MUST BE OPENED WITH CV2 MODULE



def get_reference_frame(sourceRoot,sessID,runList):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot is not defined!")
    if sessID is None:
        raise TypeError("sessID is not defined!")
    if runList is None:
        raise TypeError("runList is not defined!")
    
    runFolder=glob.glob(sourceRoot+sessID+'_run'+str(runList[0])+'_*')
    frameFolder=runFolder[0]+"/frames/"
        
    imFile=frameFolder+'frame0.tiff'
    imRef=misc.imread(imFile)
    return imRef

 # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# QUALITY CONTROL FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_first_frame_correlation(sourceRoot,sessID,runList):
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")

    for (runCount,run) in enumerate(runList):

        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"

        # READ IN FRAMES
        imFile=frameFolder+'frame0.tiff'
        im0=misc.imread(imFile)
        sz=im0.shape

        #STORE PIXEL VALUES OF FIRST FRAME
        if runCount==0:
            frame1PixMat=np.zeros((sz[0]*sz[1],len(runList)))
        frame1PixMat[:,runCount]=np.reshape(im0,sz[0]*sz[1])
    R=np.corrcoef(np.transpose(frame1PixMat))   
    
    return R

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# MOTION CORRECTION FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def motion_registration(imRef,frameStack):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if imRef is None:
        raise TypeError("imRef (reference image) is not defined!")
    if frameStack is None:
        raise TypeError("no frame stack defined!")

    imRef_forReg=np.uint8(np.true_divide(imRef,np.max(imRef))*255)
    frameCount=np.shape(frameStack)[2]

    
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Specify the number of iterations.
    number_of_iterations = 10;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    warpMatrices=np.zeros((2,3,frameCount))
    motionMag=np.zeros((frameCount))
    
    for f in range (0,frameCount):
        if f%1000==0:
            print('Motion Registration at frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        im0_forReg=np.uint8(np.true_divide(im0,np.max(im0))*255)

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)


        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (imRef_forReg,im0_forReg,warp_matrix, warp_mode, criteria)
        warpMatrices[:,:,f]=warp_matrix
        motionMag[f]=np.sum(np.square(np.eye(2, 3, dtype=np.float32)-warp_matrix))
    
    return warpMatrices,motionMag

def apply_motion_correction(frameStack=None,warpMatrices=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack defined!")
    if frameStack is None:
        raise TypeError("no warp matrices provided!")
        
  
        
    #APPLY TRANSFORMATION AND TRIM
    szY,szX,frameCount=np.shape(frameStack)
    newFrameStack=np.zeros((szY,szX,frameCount))
    
    for f in range (0,frameCount):
        if f%1000==0:
            print('Motion Correction at frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        warpMatrix=warpMatrices[:,:,f]
        im1 = cv2.warpAffine(im0, warpMatrix, (szX,szY), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        newFrameStack[:,:,f]=im1
    return newFrameStack



def get_boundaries(img=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if img is None:
        raise TypeError("no image array given!")
    
    szY,szX = np.shape(img)
    theta = get_theta_array(szY,szX)
    
    
    halfAngle=42.5
    physicalUp=np.logical_and(theta>-90-halfAngle, theta <-90+halfAngle)
    physicalRight=np.logical_and(theta>-halfAngle, theta <halfAngle)
    physicalDown=np.logical_and(theta>90-halfAngle, theta <90+halfAngle)
    physicalLeft=np.abs(theta)>180-halfAngle
    

    zeroUp=np.where(np.logical_and(img<10,physicalUp))[0]
    zeroDown=np.where(np.logical_and(img<10,physicalDown))[0]
    zeroLeft=np.where(np.logical_and(img<10,physicalLeft))[1]
    zeroRight=np.where(np.logical_and(img<10,physicalRight))[1]

    if np.size(zeroUp)==0:
        edgeUp=1
    else:
        edgeUp=np.max(zeroUp)+3

    if np.size(zeroDown)==0:
        edgeDown=szY
    else:
        edgeDown=np.min(zeroDown)-3

    if np.size(zeroLeft)==0:
        edgeLeft=1
    else:
        edgeLeft=np.max(zeroLeft)+3

    if np.size(zeroRight)==0:
        edgeRight=szX
    else:
        edgeRight=np.min(zeroRight)-3
        
    return edgeUp,edgeDown,edgeLeft,edgeRight

def get_motion_corrected_boundaries(frameStack=None,warpMatrices=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack given!")
    if warpMatrices is None:
        raise TypeError("no warp matrix stack given!")

    print('Getting motion corrected image boundaries...')
    szY,szX,frameCount=np.shape(frameStack) 
    
    edgeUpList=np.zeros((frameCount))
    edgeDownList=np.zeros((frameCount))
    edgeLeftList=np.zeros((frameCount))
    edgeRightList=np.zeros((frameCount))
    for f in range (0,frameCount):
        im0=np.copy(frameStack[:,:,f])
        
        edgeUp,edgeDown,edgeLeft,edgeRight = get_boundaries(im0)

        edgeUpList[f]=edgeUp
        edgeDownList[f]=edgeDown
        edgeLeftList[f]=edgeLeft
        edgeRightList[f]=edgeRight
    
    return edgeUpList,edgeDownList,edgeLeftList,edgeRightList
    

def apply_motion_correction_boundaries(frameStack=None,boundaries=None):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if frameStack is None:
        raise TypeError("no frame stack given!")
    if boundaries is None:
        raise TypeError("no boundaries given!")
    
    print('Applying motion corrected image boundaries...')
    frameCount=np.shape(frameStack)[2]
    
    edgeUp=boundaries[0]
    edgeDown=boundaries[1]
    edgeLeft=boundaries[2]
    edgeRight=boundaries[3]
    
    newSzY=edgeDown-edgeUp
    newSzX=edgeRight-edgeLeft
    
    newFrameStack=np.zeros((newSzY,newSzX,frameCount))
    for f in range (0,frameCount):
        if f%1000==0:
            print('frame ' +str(f)+' of '+str(frameCount))
        im0=np.copy(frameStack[:,:,f])
        newFrameStack[:,:,f]=im0[edgeUp:edgeDown,edgeLeft:edgeRight]
        
    return newFrameStack

def perform_motion_registration(sourceRoot,targetRoot,sessID,runList,makeMovies=False,frameRate=None):
    #Cesar Echavarria 11/2016
    
     #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if makeMovies and frameRate is None:
        raise TypeError("user input to make movies but frame rate not specified!")
        
    #DEFINE AND MAKE DIRECTORIES
    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
    if not os.path.exists(motionDir):
                os.makedirs(motionDir)

    motionFileDir=motionDir+'Registration/'
    if not os.path.exists(motionFileDir):
                os.makedirs(motionFileDir)

    motionFigDir=motionDir+'Figures/'
    if not os.path.exists(motionFigDir):
                os.makedirs(motionFigDir)
            
    motionMovieDir=motionDir+'Movies/'
    if makeMovies:
        if not os.path.exists(motionMovieDir):
            os.makedirs(motionMovieDir)

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
	outFile=motionFileDir+'analysis_version_info.txt'
	versionTextFile = open(outFile, 'w+')
	versionTextFile.write('WiPy version '+__version__+'\n')
	versionTextFile.close()

    #GET REFFERNCE FRAME
    imRef=get_reference_frame(sourceRoot,sessID,runList)
    szY,szX=imRef.shape


    #PERFORM REGISTRATION FOR ALL RUNS
    for (runCount,run) in enumerate(runList):
        print('Performing image registration for run '+str(run))

        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')
        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"

        frameTimes,frameCond,frameCount=get_frame_times(planFolder)

        #READ IN FRAMES
        frameArray=np.zeros((szY,szX,frameCount))
        print('Loading frames....')
        for f in range (0,frameCount):
            imFile=frameFolder+'frame'+str(f)+'.tiff'
            im0=misc.imread(imFile)
            frameArray[:,:,f]=im0[:,:]
            
        if makeMovies:
            #GENRATE RAW DATA MOVIE
            frameArrayNorm=normalize_stack(frameArray)
            outFile=motionMovieDir+sessID+'_run'+str(run)+'_raw_stack.avi'
            make_movie_from_stack(frameArrayNorm,frameRate,outFile)
    

        #MOTION REGISTRATION
        warpMatrices,motionMag=motion_registration(imRef,frameArray)

        #-> plot motion magnitude (squared error from identity matrix) and save figure
        fig=plt.figure()
        plt.plot(frameTimes,motionMag)
        fig.suptitle('Motion Over Time', fontsize=20)
        plt.xlabel('Time (secs)',fontsize=16)
        plt.ylabel('Motion Magnitude (AU)',fontsize=16)
        plt.savefig(motionFigDir+sessID+'_run'+str(run)+'_motionMagnitude.png')
        plt.close()

        #-> save warp matrices
        outFile=motionFileDir+sessID+'_run'+str(run)+'_motionRegistration'
        np.savez(outFile,warpMatrices=warpMatrices)

        #APPLY MOTION CORRECTION AND SAVE
        correctedFrameArray=apply_motion_correction(frameArray,warpMatrices)
        outFile=motionFileDir+sessID+'_run'+str(run)+'_correctedFrames'
        np.savez(outFile,correctedFrameArray=correctedFrameArray)

        #GET MOTION CORRECTED BOUNDARIES
        edgeUpList,edgeDownList,edgeLeftList,edgeRightList = get_motion_corrected_boundaries(correctedFrameArray,warpMatrices)

        if runCount==0:
            edgeUp=np.max(edgeUpList)
            edgeDown=np.min(edgeDownList)
            edgeLeft=np.max(edgeLeftList)
            edgeRight=np.min(edgeRightList)
        else:
            edgeUp=np.max([edgeUp,np.max(edgeUpList)])
            edgeDown=np.min([edgeDown,np.min(edgeDownList)])
            edgeLeft=np.max([edgeLeft,np.max(edgeLeftList)])
            edgeRight=np.min([edgeRight,np.min(edgeRightList)])

        if makeMovies:
            #APPLY BOUNDARIES
            tmp_boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
            trimCorrectedFrameArray = apply_motion_correction_boundaries(correctedFrameArray,tmp_boundaries)

            #GENRATE MOTION CORRECTED MOVIE
            frameArrayNorm=normalize_stack(trimCorrectedFrameArray)
            outFile=motionMovieDir+sessID+'_run'+str(run)+'_MC_trimmed_stack.avi'
            make_movie_from_stack(frameArrayNorm,frameRate,outFile)

    boundaries=(edgeUp,edgeDown,edgeLeft,edgeRight)  
    #->save boundaries
    outFile=motionFileDir+sessID+'_motionCorrectedBoundaries'
    np.savez(outFile,boundaries=boundaries)
    
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# STAGE 1 -- DATA ANALYSIS FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_analysis_path(analysisRoot,interp=False, removeRollingMean=False, \
    motionCorrection=False, smoothing=False, fwhm=0, \
    timecourseAnalysis=False, baseline_startT=-2, baseline_endT=0, timecourse_startT=0, timecourse_endT=8,\
     timeWindowAnalysis=False,tWindow1_startT=-2, tWindow1_endT=0, tWindow2_startT=0, tWindow2_endT=5):
	#Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if analysisRoot is None:
        raise TypeError("analysisRoot (directory) not specified!")

    imgOperationDir=''
    #DEFINE DIRECTORIES
    if motionCorrection:
    	imgOperationDir=imgOperationDir+'motionCorrection_'

    if smoothing:
    	imgOperationDir=imgOperationDir+'smoothing_fwhm'+str(fwhm)
    else:
    	imgOperationDir=imgOperationDir+'noSmoothing'

    procedureDir=''
    if interpolate:
        procedureDir=procedureDir+'interpolate'

    if removeRollingMean:
        procedureDir=procedureDir+'_minusRollingMean'

    generalProcessDir=imgOperationDir+'/'+procedureDir+'/'

    tCourseOutDir=None
    tWindowOutDir=None
    if timeWindowAnalysis:
        analysisDir=analysisRoot+'timeWindow/'+generalProcessDir
        tWindowDir=analysisDir+'tWindow1_'+str(tWindow1_startT)+'_'+str(tWindow1_endT)+'_tWindow2_'+str(tWindow2_startT)+'_'+str(tWindow2_endT)+'/'

    if timecourseAnalysis:
        analysisDir=analysisRoot+'timeCourse/'+generalProcessDir
        tCourseDir=analysisDir+'baseline_'+str(baseline_startT)+'_'+str(baseline_endT)+'_response_'+str(timecourse_startT)+'_'+str(timecourse_endT)+'/'

    return(tWindowDir,tCourseDir)


def analyze_blocked_data(sourceRoot, targetRoot, sessID, runList, frameRate, \
    interp=False, removeRollingMean=False, \
    motionCorrection=False, smoothing=False, fwhm=0, \
    timecourseAnalysis=False, baseline_startT=-2, baseline_endT=0, timecourse_startT=0, timecourse_endT=8,\
     timeWindowAnalysis=False,tWindow1_startT=-2, tWindow1_endT=0, tWindow2_startT=0, tWindow2_endT=5,\
     qualityControl=True):
    #Cesar Echavarria 11/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if frameRate is None:
        raise TypeError("frameRate not specified!")
        
    # DEFINE DIRECTORIES
    analysisRoot=targetRoot+'/Sessions/'+sessID+'/Analyses/';

    tWindowOutDir,tCourseOutDir=get_analysis_path(analysisRoot,interp, removeRollingMean, \
    	motionCorrection, smoothing, fwhm, \
    	timecourseAnalysis, baseline_startT, baseline_endT, timecourse_startT, timecourse_endT,\
    	timeWindowAnalysis,tWindow1_startT, tWindow1_endT, tWindow2_startT, tWindow2_endT)

    QCtargetFolder=targetRoot+'Sessions/'+sessID+'/QualControl/Figures/'

    motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'

    #MAKE NECESSARY DIRECTORIES
    if timeWindowAnalysis:
        tWindowOutDir=tWindowOutDir+'MidProcOutput/'
        if not os.path.exists(tWindowOutDir):
            os.makedirs(tWindowOutDir)


        #OUTPUT TEXT FILE WITH PACKAGE VERSION
        outFile=tWindowOutDir+'analysis_version_info.txt'
        versionTextFile = open(outFile, 'w+')
        versionTextFile.write('WiPy version '+__version__+'\n')
        versionTextFile.close()

    if timecourseAnalysis:
        tCourseOutDir=tCourseOutDir+'MidProcOutput/'
        if not os.path.exists(tCourseOutDir):
            os.makedirs(tCourseOutDir) 

        #OUTPUT TEXT FILE WITH PACKAGE VERSION
        outFile=tCourseOutDir+'analysis_version_info.txt'
        versionTextFile = open(outFile, 'w+')
        versionTextFile.write('WiPy version '+__version__+'\n')
        versionTextFile.close() 


    #BEGIN DATA PROCESSING
    for (runCount,run) in enumerate(runList):

        print('run='+str(run))

        runFolder=glob.glob(sourceRoot+sessID+'_run'+str(run)+'_*')

        frameFolder=runFolder[0]+"/frames/"
        planFolder=runFolder[0]+"/plan/"

        # READ IN FRAME TIMES FILE
        frameTimes,frameCond,frameCount=get_frame_times(planFolder)
        # READ IN BLOCK PARADIGM
        blockCond,blockStartT,nBlocks=get_block_paradigm(planFolder)

        # READ IN FIRST FRAME FOR SIZE
        imFile=frameFolder+'frame0.tiff'
        im0=misc.imread(imFile)
        sz=im0.shape

        # ***TO DO - APPLY SMOOTHING***
        print('Loading frames...')
        if motionCorrection:

            #LOAD MOTION CORRECTED FRAMES
            inFile=outFile=motionFileDir+sessID+'_run'+str(run)+'_correctedFrames.npz'
            f=np.load(inFile)
            frameArray=f['correctedFrameArray']

            #LOAD MOTION CORRECTED BOUNDARIES
            inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
            f=np.load(inFile)
            boundaries=f['boundaries']

            #APPLY BOUNDARIES
            frameArray = apply_motion_correction_boundaries(frameArray,boundaries)

            szY,szX = np.shape(frameArray[:,:,0])

        else:
            #GET REFFERNCE FRAME
            imRef=get_reference_frame(sourceRoot,sessID,runList)
            szY,szX=imRef.shape

            # READ IN FRAMES
            frameArray=np.zeros(sz+(frameCount,))
            frameArray[:,:,0]=im0
            for f in range (1,frameCount):
                imFile=frameFolder+'frame'+str(f)+'.tiff'
                im0=misc.imread(imFile)
                frameArray[:,:,f]=im0[:,:]

        frameArray=np.reshape(frameArray,(szY*szX,frameCount))

        if qualityControl:
            if not os.path.exists(QCtargetFolder):
                os.makedirs(QCtargetFolder)

            meanF=np.squeeze(np.mean(frameArray,0))#average over pixels

            fig=plt.figure()
            plt.plot(frameTimes,meanF)

            fig.suptitle('Mean Pixel Value Over Time', fontsize=20)
            plt.xlabel('Time (secs)',fontsize=16)
            plt.ylabel('Mean Pixel Value',fontsize=16)
            plt.savefig(QCtargetFolder+sessID+'_run'+str(run)+'meanPixelValue.png')
            plt.close()

            randPix=np.random.randint(0,szY*szX)

            fig=plt.figure()
            plt.plot(frameTimes,frameArray[randPix,:])

            fig.suptitle('Pixel '+str(randPix)+' Value Over Time', fontsize=20)
            plt.xlabel('Time (secs)',fontsize=16)
            plt.ylabel('Mean Pixel Value',fontsize=16)
            plt.savefig(QCtargetFolder+sessID+'+run'+str(run)+'randomPixelValue.png')
            plt.close()

        # INTERPOLATE FOR CONSTANT FRAME RATE
        if interp:
            print('Interpolating....')
            interpF = interpolate.interp1d(frameTimes, frameArray,1)
            newTimes=np.arange(frameTimes[0],frameTimes[-1],1.0/frameRate)
            frameArray=interpF(newTimes)   # use interpolation function returned by `interp1d`
            frameTimes=newTimes

        # REMOVE ROLLING AVERAGE
        if removeRollingMean:
            print('Removing rolling mean....')
            detrendedFrameArray=np.zeros(np.shape(frameArray))
            rollingWindowSz=int(np.ceil(((tWindow2_endT-tWindow2_startT)*2)*frameRate))

            for pix in range(0,szY*szX):
                meanVal=np.mean(frameArray[pix,:],0)

                tmp0=frameArray[pix,:];
                tmp1=np.concatenate((np.ones(rollingWindowSz)*tmp0[0], tmp0, np.ones(rollingWindowSz)*tmp0[-1]),0)

                rollingAvg=np.convolve(tmp1, np.ones(rollingWindowSz)/rollingWindowSz, 'same')
                rollingAvg=rollingAvg[rollingWindowSz:-rollingWindowSz]


                detrendedFrameArray[pix,:]=np.subtract(tmp0,rollingAvg)+meanVal;
            frameArray=detrendedFrameArray
            del detrendedFrameArray

        # GET BASELINE AND STIM RESPONSE TIME COURSE
        print('aggregating responses across trials....')
        stimBlockStartT=blockStartT[np.not_equal(blockCond,0)]
        stimBlockCond=blockCond[np.not_equal(blockCond,0)]
        nStimBlocks=len(stimBlockStartT)
        nPtsBase=int(np.ceil((tWindow1_endT-tWindow1_startT)*frameRate))
        nPtsStim=int(np.ceil((tWindow2_endT-tWindow2_startT)*frameRate))

        baseResp=np.zeros((szY*szX,nStimBlocks,nPtsBase))
        stimResp=np.zeros((szY*szX,nStimBlocks,nPtsStim))
        if timeWindowAnalysis:
            for stimBlock in range(0,nStimBlocks):
                window1_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+tWindow1_startT)[0][0]
                window1_endInd=window1_startInd+nPtsBase
                window2_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+tWindow2_startT)[0][0]
                window2_endInd=window2_startInd+nPtsStim

                baseResp[:,stimBlock,:]=frameArray[:,int(window1_startInd):int(window1_endInd)]
                stimResp[:,stimBlock,:]=frameArray[:,int(window2_startInd):int(window2_endInd)]

            outFile=tWindowOutDir+sessID+'_run'+str(run)
            np.savez(outFile,stimBlockCond=stimBlockCond,baseResp=baseResp,stimResp=stimResp,frameRate=frameRate,tWindow1_startT=tWindow1_startT,tWindow1_endT=tWindow1_endT,tWindow2_startT=tWindow2_startT,tWindow2_endT=tWindow2_endT)
        if timecourseAnalysis:
            nPtsBase=int(np.ceil((baseline_endT-baseline_startT)*frameRate))
            nPtsResponse=int(np.ceil((timecourse_endT-timecourse_startT)*frameRate))

            baseline=np.zeros((szY*szX,nStimBlocks,nPtsBase))
            response=np.zeros((szY*szX,nStimBlocks,nPtsResponse))

            for stimBlock in range(0,nStimBlocks):
                baseline_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+baseline_startT)[0][0]
                baseline_endInd=baseline_startInd+nPtsBase
                timecourse_startInd=np.where(frameTimes>stimBlockStartT[stimBlock]+timecourse_startT)[0][0]
                timecourse_endInd=timecourse_startInd+nPtsResponse

                baseline[:,stimBlock,:]=frameArray[:,int(baseline_startInd):int(baseline_endInd)]
                response[:,stimBlock,:]=frameArray[:,int(timecourse_startInd):int(timecourse_endInd)]

            outFile=tCourseOutDir+sessID+'_run'+str(run)
            np.savez(outFile,stimBlockCond=stimBlockCond,baseline=baseline,response=response,frameRate=frameRate,baseline_startT=baseline_startT,baseline_endT=baseline_endT,timecourse_startT=timecourse_startT,timecourse_endT=timecourse_endT)


    if qualityControl:
        R=get_first_frame_correlation(sourceRoot,sessID,runList)
        fig=plt.figure()
        plt.imshow(R,interpolation='none')
        plt.colorbar()
        plt.savefig(QCtargetFolder+sessID+'_firstFrame_CorrelationMatrix.png')
        plt.close()
        

        
def average_trials_timecourse(sourceRoot, targetRoot, analysisDir, sessID, nCond, runList, avgFolder,\
    percentSignalChange=True, SDmaps=False):
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if nCond is None:
        raise TypeError("nCond (number of conditions) not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")


    # DEFINE DIRECTORIES
    inDir=analysisDir+'/MidProcOutput/'
    outDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';
    if not os.path.exists(outDir):
        os.makedirs(outDir) 

    if SDmaps:
        figOutDir=analysisDir+'/Figures/'+avgFolder+'/SDmaps/'
        if not os.path.exists(figOutDir):
            os.makedirs(figOutDir) 

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=outDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()


    #GET REFFERNCE FRAME
    imRef=get_reference_frame(sourceRoot,sessID,runList)
    szY,szX=imRef.shape



    for cond in range(nCond):
        print("condition = "+str(cond+1))
        # RETRIEVE DATA
        for (runCount,run) in enumerate(runList):
            print("run = "+str(run))

            inFile=inDir+sessID+'_run'+str(run)+'.npz'
            f=np.load(inFile)
            stimBlockCond=f['stimBlockCond']
            baseResp=f['baseline']
            stimResp=f['response']
            
            if runCount == 0:
                frameRate=f['frameRate']
                timecourse_startT=f['timecourse_startT']
                timecourse_endT=f['timecourse_endT']



            nPix,nTrials,basePts=np.shape(baseResp)
            nPix,nTrials,respPts=np.shape(stimResp)

            condInd=np.where(stimBlockCond==cond+1)[0]

            if runCount==0:

                stimRespAll=np.zeros((nPix,len(condInd)*len(runList),respPts))
                baseRespAll=np.zeros((nPix,len(condInd)*len(runList),basePts))

            startInd=(runCount)*len(condInd)
            endInd=startInd+len(condInd)

            stimRespAll[:,startInd:endInd,:]=stimResp[:,condInd,:]
            baseRespAll[:,startInd:endInd,:]=baseResp[:,condInd,:]

        #WRITE NUMBER OF TRIALS PER CONDITION TO TEXT FILE
        if cond == 0:
            outFile=outDir+'total_trials_per_condition.txt'
            trialTextFile = open(outFile, 'w+')
            trialTextFile.write('NUMBER OF TRIALS\n')
            trialTextFile.write('\n')
        
        trialTextFile.write('condition '+str(cond+1)+' '+str(np.shape(stimRespAll)[1])+' trials\n')
        


        if percentSignalChange:#output percent signal change avged over trials
            if cond==0:
                stimRespMean=np.zeros((nPix,nCond,respPts))
            baseRespTimeMean=np.mean(baseRespAll,2)#avg over time

            allPSC=np.zeros(np.shape(stimRespAll))
            for pt in range(0,respPts):
                allPSC[:,:,pt]=np.true_divide(stimRespAll[:,:,pt]-baseRespTimeMean,baseRespTimeMean)*100

            stimRespMean[:,cond,:]=np.mean(allPSC,1)#avg over trials
        else: #output pixel values avged over trials
            if cond==0:
                stimRespMean=np.zeros((nPix,nCond,respPts))
            stimRespMean[:,cond,:]=np.mean(stimRespAll,1)#avg over trials

    trialTextFile.close()
    outFile=outDir+sessID+'_timecourse_trialAvg'
    np.savez(outFile,frameRate=frameRate,stimRespMean=stimRespMean,timecourse_startT=timecourse_startT,timecourse_endT=timecourse_endT)

    #OUTPUT AND SAVE STANDARD DEV
    if SDmaps:

        pixSD=np.std(stimRespMean,2)
        mapSD=np.reshape(pixSD,(szY,szX,nCond))

        #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
        inFile=targetRoot+'contrastDic.npz'
        f=np.load(inFile)
        contrastDic=f['contrastDic']

        for c in range(0,nCond):
            fig=plt.figure()
            plt.imshow(mapSD[:,:,c])
            plt.colorbar()
            plt.savefig(figOutDir+sessID+'_'+contrastDic[c]['name']+'_SDmap.png')
            plt.close()

        outFile=outDir+sessID+'_mapSD'
        np.savez(outFile,mapSD=mapSD)
        

def average_trials_tWindow(sourceRoot, targetRoot, sessID, nCond, runList, analysisDir, avgFolder,\
                          motionCorrection=False,retinoAnalysis=False,percentSignalChange=False, parametricStat=False):
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if runList is None:
        raise TypeError("runList not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if nCond is None:
        raise TypeError("nCond (number of conditions) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")

 
    
    # DEFINE DIRECTORIES
    inDir=analysisDir+'/MidProcOutput/'
    outDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';
    if not os.path.exists(outDir):
        os.makedirs(outDir) 

    #OUTPUT TEXT FILE WITH PACKAGE VERSION
    outFile=outDir+'analysis_version_info.txt'
    versionTextFile = open(outFile, 'w+')
    versionTextFile.write('WiPy version '+__version__+'\n')
    versionTextFile.close()


    if motionCorrection:
        #GET MOTION CORRECTED BOUNDARIES
        motionDir=targetRoot+'/Sessions/'+sessID+'/Motion/';
        motionFileDir=motionDir+'Registration/'
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        edgeUp=boundaries[0]
        edgeDown=boundaries[1]
        edgeLeft=boundaries[2]
        edgeRight=boundaries[3]
        #GET MOTION CORRECTED FRAME SIZE
        szY=edgeDown-edgeUp
        szX=edgeRight-edgeLeft

    else:
        #GET REFFERNCE FRAME
        imRef=get_reference_frame(sourceRoot,sessID,runList)
        szY,szX=imRef.shape

    for cond in range(nCond):
        print('cond = '+str(cond+1))
        # RETRIEVE DATA

        for (runCount,run) in enumerate(runList):
            print('run = '+str(run))
            
            inFile=inDir+sessID+'_run'+str(run)+'.npz'
            f=np.load(inFile)
            stimBlockCond=f['stimBlockCond']
            baseResp=f['baseResp']
            stimResp=f['stimResp']
            frameRate=f['frameRate']

            nPix,nTrials,stimPts=np.shape(stimResp)
            nPix,nTrials,basePts=np.shape(baseResp)

            condInd=np.where(stimBlockCond==cond+1)[0]

            if runCount==0:
                trialsPerCond=(len(runList)*nTrials)/nCond#assume all runs have same number of trials per condition
                
                stimRespAll=np.zeros((nPix,trialsPerCond,stimPts))
                baseRespAll=np.zeros((nPix,trialsPerCond,basePts))
                startInd=0

            endInd=startInd+len(condInd)

            stimRespAll[:,startInd:endInd,:]=stimResp[:,condInd,:]
            baseRespAll[:,startInd:endInd,:]=baseResp[:,condInd,:]
            startInd=endInd
        #WRITE NUMBER OF TRIALS PER CONDITION TO TEXT FILE
        if cond == 0:
            outFile=outDir+'total_trials_per_condition.txt'
            trialTextFile = open(outFile, 'w+')
            trialTextFile.write('NUMBER OF TRIALS\n')
            trialTextFile.write('\n')
        
        trialTextFile.write('condition '+str(cond+1)+' '+str(np.shape(stimRespAll)[1])+' trials\n')

        # AVERAGE OVER TIME
        if cond==0:
            
            baseRespTimeMean=np.zeros((nPix,trialsPerCond,nCond))
            stimRespTimeMean=np.zeros((nPix,trialsPerCond,nCond))
        baseRespTimeMean[:,:,cond]=np.mean(baseRespAll,2)#avg over time
        stimRespTimeMean[:,:,cond]=np.mean(stimRespAll,2)#avg over time
        
        if retinoAnalysis:
            #Z-SCORE RESPONSE RELATIVE TO BASELINE
            baseRespMean=np.mean(baseRespAll,1)#avg over trials
            stimRespMean=np.mean(stimRespAll,1)#avg over trials

            baseMean=np.mean(baseRespMean,1)#avg over time
            baseMean=np.tile(np.expand_dims(baseMean,1),(1,stimPts))
            baseSD=np.std(baseRespMean,1)#avg over time
            baseSD=np.tile(np.expand_dims(baseSD,1),(1,stimPts))


            zScoredStimResp=np.true_divide(stimRespMean-baseMean,baseSD)#z-score 
            observedResp=np.mean(zScoredStimResp,1)#avg over time

            respMap=np.reshape(observedResp,(szY,szX))
            outFile=outDir+sessID+'_condition'+str(cond)+'_zScoreMap'
            np.savez(outFile,respMap=respMap);
        
        if not retinoAnalysis:
            #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
            inFile=targetRoot+'contrastDic.npz'
            f=np.load(inFile)
            contrastDic=f['contrastDic']

            if percentSignalChange:
                # Percent Signal Change
                PSC=np.mean(np.true_divide(np.subtract(stimRespTimeMean,baseRespTimeMean),baseRespTimeMean)*100,1)#average over trials
                # ->contrast conditions
                for c in range(len(contrastDic)):
                    if contrastDic[c]['minus']==[0]:
                        PSCpix=np.mean(PSC[:,np.subtract(contrastDic[c]['plus'],1)],1)
                    else:
                        PSCpix=np.mean(PSC[:,np.subtract(contrastDic[c]['plus'],1)],1)-np.mean(PSC[:,np.subtract(contrastDic[c]['minus'],1)],1)
                    PSCmap=np.reshape(PSCpix,sz)
                    outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_PSCmap'
                    np.savez(outFile,PSCmap=PSCmap);

                    plt.imshow(PSCmap)

            if parametricStat:
                # SPM
                for c in range(len(contrastDic)):
                    if contrastDic[c]['minus']==[0]:
                        plusMat=np.mean(stimRespTimeMean[:,:,np.subtract(contrastDic[c]['plus'],1)],2)
                        minusMat=np.mean(baseRespTimeMean[:,:,np.subtract(contrastDic[c]['plus'],1)],2)
                    else:
                        plusMat=np.mean(stimRespTimeMean[:,:,np.subtract(contrastDic[c]['plus'],1)],2)
                        minusMat=np.mean(stimRespTimeMean[:,:,np.subtract(contrastDic[c]['minus'],1)],2)
                    tStatPix,pValPix=stats.ttest_rel(plusMat,minusMat,1)
                    tStatMap=np.reshape(tStatPix,(szY,szX))
                    pValMap=np.reshape(pValPix,(szY,szX))
                    outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_SPMmap'
                    np.savez(outFile,tStatMap=tStatMap,pValMap=pValMap);
    trialTextFile.close()
                    

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

# VISUALIZATION FUNCTIONS

# # # # # # # # # # # # # # # # # # # # # # # # # # # #

def make_pixel_plots(targetRoot, sessID, frameRate, analysisDir, avgFolder, nPix=100):

    #Cesar Echavarria 10/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if frameRate is None:
        raise TypeError("frameRate not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
        
    # DEFINE DIRECTORIES
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';

    figOutDir=analysisDir+'/Figures/'+avgFolder+'/timecourse_singlePixel/'
    if not os.path.exists(figOutDir):
        os.makedirs(figOutDir) 

    #LOAD IN STANDARD DEV MAP
    inFile=inDir+sessID+'_mapSD.npz'
    f=np.load(inFile)
    mapSD=f['mapSD']

    #GET PIXELS WITH MOST VARIANCE (MAYBE BEST SIGNAL)
    szY,szX,nCond=np.shape(mapSD)
    pixSD=np.reshape(mapSD,(szY*szX*nCond))

    pixList=np.unique(np.argsort(pixSD).argsort()[::-1][:nPix]%(szY*szX))

    #LOAD IN TIME COURSE AND RELATED PARAMETERS
    inFile=inDir+sessID+'_timecourse_trialAvg.npz'
    f=np.load(inFile)
    startT=f['timecourse_startT']
    frameRate=f['frameRate']
    stimRespMean=f['stimRespMean']
    sampRate=np.true_divide(1,frameRate)

    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=targetRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET SOME INFO ABOUT DATA
    nPix,nCond,respPts=np.shape(stimRespMean)
    frameTimes=np.linspace(startT,startT+(respPts)*sampRate,respPts)

    #MAKE PLOTS
    colorList='bgrmkc'
    for pix in pixList:
        legHand=[]
        fig=plt.figure()
        for c in range(0,nCond):
            plt.plot(frameTimes,stimRespMean[pix,c,:],colorList[c],linewidth=2)

            endInd=contrastDic[c]['name'].index('_')
            legHand.append(mlines.Line2D([], [], color=colorList[c], marker='_',
                                  markersize=15, label=contrastDic[c]['name'][0:endInd]))
        plt.legend(handles=legHand)

        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        plt.axvline(x=0, ymin=ymin, ymax = ymax, linewidth=1, color='k')
        plt.axvline(x=5, ymin=ymin, ymax = ymax, linewidth=1, color='r')

        fig.suptitle(sessID+' Pixel '+str(pix)+' Value Over Time', fontsize=16)
        plt.xlabel('Time (secs)',fontsize=16)
        plt.ylabel('PSC',fontsize=16)
        plt.savefig(figOutDir+sessID+'+pixel_'+str(pix)+'_timecourse_trialAvg.png')
        plt.close()
        
        
def make_movie_from_stack(frameStack,frameRate=24,movFile='test.avi'):
    #Cesar Echavarria 10/2016
    
    #CHECK INPUTS
    if frameStack is None:
        raise TypeError("no frame stack provided!")
    if np.amax(frameStack) > 1:
        raise TypeError("frame stack values must be in the range 0-1")
    
    
    #GET STACK INFO
    szY,szX,nFrames=np.shape(frameStack)

    #MAKE TEMP FOLDER
    tmpDir='./tmp/'
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)


    #WRITE TO TEMP FOLDER
    for i in range(0,nFrames):
        outFile=tmpDir+str(i)+'.png'
        frame=np.uint8(frameStack[:,:,i]*255)
        misc.imsave(outFile,frame)

    #WRITE VIDEO
    cmd='ffmpeg -y -r '+'%.3f'%frameRate+' -i '+tmpDir+'%d.png -vcodec mjpeg -f avi '+movFile
    os.system(cmd)


    #GET RID OF TEMP FOLDER
    shutil.rmtree(tmpDir)
    
                                    
def make_movie_from_stack_mark_stim(frameStack,frameRate=24,onFrame=0,offFrame=None,movFile='test.avi'):
    #Cesar Echavarria 10/2016
    
    #CHECK INPUTS
    if frameStack is None:
        raise TypeError("no frame stack provided!")
    if np.amax(frameStack) > 1:
        raise TypeError("frame stack values must be in the range 0-1")
    
    
    #GET STACK INFO
    szY,szX,nFrames=np.shape(frameStack)

    #MAKE TEMP FOLDER
    tmpDir='./tmp/'
    if not os.path.exists(tmpDir):
        os.makedirs(tmpDir)


    #WRITE TO TEMP FOLDER
    for i in range(0,nFrames):
        outFile=tmpDir+str(i)+'.png'
        frame=np.copy(frameStack[:,:,i])
        if offFrame is None:
            if i >= onFrame:
                frame=np.pad(frame,(5,), 'constant', constant_values=(1,))
            else:
                frame=np.pad(frame,(5,), 'constant', constant_values=(0,))
        else:
            if i >= onFrame and i<offFrame:
                frame=np.pad(frame,(5,), 'constant', constant_values=(1,))
            else:
                frame=np.pad(frame,(5,), 'constant', constant_values=(0,))
        frame=np.uint8(frame*255)
        misc.imsave(outFile,frame)

    #WRITE VIDEO
    cmd='ffmpeg -y -r '+'%.3f'%frameRate+' -i '+tmpDir+'%d.png -vcodec mjpeg '+movFile
    os.system(cmd)


    #GET RID OF TEMP FOLDER
    shutil.rmtree(tmpDir)  
    
def make_movie_from_timecourse_average(sourceRoot, targetRoot, sessID, frameRate, analysisDir, avgFolder, stimDur=5):
    #Cesar Echavarria 10/2016
    
    #MAKE SURE YOU GET SOME ARGUMENTS
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if targetRoot is None:
        raise TypeError("targetRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if frameRate is None:
        raise TypeError("frameRate not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
        
    # DEFINE DIRECTORIES
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';

    movieOutDir=analysisDir+'/Movies/'+avgFolder+'/'
    if not os.path.exists(movieOutDir):
        os.makedirs(movieOutDir) 

    #GET REFFERNCE FRAME
    imRef=get_reference_frame(sourceRoot,sessID,runList)
    szY,szX=imRef.shape


    #LOAD IN TIME COURSE AND RELATED PARAMETERS
    inFile=inDir+sessID+'_timecourse_trialAvg.npz'
    f=np.load(inFile)
    startT=f['timecourse_startT']
    frameRate=f['frameRate']
    stimRespMean=f['stimRespMean']
    sampRate=np.true_divide(1,frameRate)

    nPix,nCond,respPts=np.shape(stimRespMean)

    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=targetRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    #GET SOME INFO ABOUT DATA
    nPix,nCond,respPts=np.shape(stimRespMean)
    frameTimes=np.linspace(startT,startT+(respPts)*sampRate,respPts)

    onFrame=np.where(frameTimes>baseline_endT)[0][0]
    offFrame=np.where(frameTimes>stimDur)[0][0]


    for c in range(0,nCond):
        #NORMALIZE ARRAY
        frameArray=stimRespMean[:,c,:]
        arrayMin=np.amin(frameArray)
        arrayMax=np.amax(frameArray)
        frameArray=np.true_divide((frameArray-arrayMin),(arrayMax-arrayMin))

        #RESHAPE
        frameArray=np.reshape(frameArray,(szY,szX,respPts))


        #MAKE MOVIE
        endInd=contrastDic[c]['name'].index('_')
        outFile=movieOutDir+movFile+'_'+contrastDic[c]['name'][0:endInd]+'_movie.avi'
        #make_movie_from_stack(frameArray,frameRate=frameRate,movFile=outFile)
        make_movie_from_stack_mark_stim(frameArray,frameRate=frameRate,onFrame=onFrame,offFrame=offFrame,movFile=outFile)


def generate_PSC_map(sourceRoot,sessID,analysisDir,avgFolder, motionCorrection=False,\
    manualThresh = False, threshMin = 1, threshMax = 3):
    #Cesar Echavarria - 10/2016
    
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")

        
    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'

    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'
    if manualThresh:
        outDir=analysisDir+'/Figures/'+avgFolder+'/PSCcontrast/manualThresh/'
    else:
        outDir=analysisDir+'/Figures/'+avgFolder+'/PSCcontrast/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 


    # READ IN SURFACE

    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;

    if motionCorrection:
        #GET AND APPLY MOTION CORRECTED BOUNDARIES
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        edgeUp=boundaries[0]
        edgeDown=boundaries[1]
        edgeLeft=boundaries[2]
        edgeRight=boundaries[3]
        imSurf=imSurf[edgeUp:edgeDown,edgeLeft:edgeRight]

    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)

    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=sourceRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']

    for c in range(len(contrastDic)):
        # LOAD IN MAP WITH PSC PER PIXEL
        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_PSCmap.npz'
        f=np.load(mapFile)
        PSCmap=f['PSCmap']

        #RESIZE TO SIZE OF ANATOMICAL MAP
        PSCmap = cv2.resize(PSCmap,(szX,szY))
        funcOverlay=np.copy(funcOverlayTmp)

        #AUTOMATIC THRESHOLD 
        if not manualThresh:
            pMin=.2
            pMax=.8

            threshMin=pMin*np.max(np.absolute(PSCmap))
            if threshMin>np.max(np.absolute(PSCmap)):
                threshMax=threshMin
            else:
                threshMax=pMax*np.max(np.absolute(PSCmap))
        threshList=np.linspace(threshMin,threshMax,8);

        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=.5#SET SATURATION OF COLORS

        #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(PSCmap>-threshList[t+1],PSCmap<=-threshList[t]))

            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=(np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1))-imTemp
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=PSCmap<=-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
        funcOverlay[:,:,channel]=imTemp[:,:]

        #WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(PSCmap<threshList[t+1],PSCmap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=PSCmap>=threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
        funcOverlay[:,:,channel]=imTemp[:,:]

        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)

        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        plt.savefig(outDir+sessID+'_'+contrastDic[c]['name']+'_withColorbar.png')

        outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_image.png'
        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()
        
def generate_SPM(sourceRoot,sessID,analysisDir,avgFolder,motionCorrection=False,\
    manualThresh = False, threshMin = 1, threshMax = 3):
    #Cesar Echavarria - 10/2016
    
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
        
    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    resultsDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/'

    if manualThresh:
        outDir=analysisDir+'/Figures/'+avgFolder+'/SPMmaps/manualThresh/'
    else:
        outDir=analysisDir+'/Figures/'+avgFolder+'/SPMmaps/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 



    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;
    if motionCorrection:
        #GET AND APPLY MOTION CORRECTED BOUNDARIES
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        edgeUp=boundaries[0]
        edgeDown=boundaries[1]
        edgeLeft=boundaries[2]
        edgeRight=boundaries[3]
        imSurf=imSurf[edgeUp:edgeDown,edgeLeft:edgeRight]

    funcOverlayTmp=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlayTmp)



    #LOAD IN DICTIONARY WITH CONTRAST DEFINITIONS
    inFile=sourceRoot+'contrastDic.npz'
    f=np.load(inFile)
    contrastDic=f['contrastDic']


    for c in range(len(contrastDic)):
        # LOAD IN MAP WITH PSC PER PIXEL
        mapFile=resultsDir+sessID+'_'+contrastDic[c]['name']+'_SPMmap.npz'
        f=np.load(mapFile)
        pValMap=f['pValMap']
        tStatMap=f['tStatMap']

        SPMmap=(-np.log10(pValMap))*np.sign(tStatMap)

        #RESIZE TO SIZE OF ANATOMICAL MAP
        SPMmap = cv2.resize(SPMmap,(szX,szY))
        funcOverlay=np.copy(funcOverlayTmp)
        if not manualThresh:
            #AUTOMATIC THRESHOLD 
            pMin=.2
            pMax=.8

            threshMin=pMin*np.max(np.absolute(SPMmap))
            if threshMin<1.35:
                threshMin=1.35
            if threshMin>np.max(np.absolute(SPMmap)):
                threshMax=threshMin
            else:
                threshMax=pMax*np.max(np.absolute(SPMmap))
                
        threshList=np.linspace(threshMin,threshMax,8);

        #DEFINE COLOR MAP
        colorOverlayNeg=np.array([[0,0,.25],[0,0,.5],[0,0,.75],[0,0,1],[0,.25,1],[0,.5,1],[0,.75,1],[0,1,1]]);
        colorOverlayPos=np.array([[.25,0,0],[.5,0,0],[.75,0,0],[1,0,0],[1,.25,0],[1,.5,0],[1,.75,0],[1,1,0]]);
        cMap=np.vstack((np.flipud(colorOverlayNeg),np.array([.5,.5,.5]),colorOverlayPos))

        imSat=.5#SET SATURATION OF CLORS
        #COOL COLORS
        for t in range(len(threshList)-1):
            colorInd=np.where(np.logical_and(SPMmap>-threshList[t+1],SPMmap<=-threshList[t]))

            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=(np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1))-imTemp
                  #  imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=SPMmap<=-threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayNeg[t,channel]*((2**8)-1)),imTemp)
            #imTemp[colorInd]=colorOverlayNeg[t,channel]*((2**8)-1)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
        funcOverlay[:,:,channel]=imTemp[:,:]

        #WARM COLORS
        for t in range(len(threshList)-1):
            colorInd=np.logical_and(SPMmap<threshList[t+1],SPMmap>=threshList[t])
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
                imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
                #  imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
            funcOverlay[:,:,channel]=imTemp[:,:]

        colorInd=SPMmap>=threshList[t+1]
        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            diffVal=np.subtract((np.ones((szY,szX))*colorOverlayPos[t,channel]*((2**8)-1)),imTemp)
            imTemp[colorInd]=imTemp[colorInd]+np.multiply(diffVal[colorInd],imSat)
            #imTemp[colorInd]=colorOverlayPos[t,channel]*((2**8)-1)
        funcOverlay[:,:,channel]=imTemp[:,:]    
        #MAKE AND SAVE IMAGE WITH COLOR BAR
        mycmap = array2cmap(cMap)
        plt.imshow(np.uint8(funcOverlay),cmap=mycmap)
        plt.axis('off')

        nPos=np.shape(colorOverlayPos)[0]
        nNeg=np.shape(colorOverlayNeg)[0]
        nSteps=nPos+nNeg+1

        t1Loc=np.true_divide(1,nSteps)
        t2Loc=(np.true_divide(1,nSteps)*nPos)-np.true_divide(t1Loc,2)
        t3Loc=t2Loc+(2*np.true_divide(1,nSteps))
        t4Loc=1-np.true_divide(1,nSteps)
        tickLoc=np.array([t1Loc,t2Loc,t3Loc,t4Loc])
        tickLoc=tickLoc*np.max(np.uint8(funcOverlay))

        #*Priting out P-value directly is troublesome...small p-values not printed out correctly. Opting for printing out -log10(pValue)
        t1Label=-np.max(threshList)
        t2Label=-np.min(threshList)
        t3Label=np.min(threshList)
        t4Label=np.max(threshList)
        tickLabels=np.around(np.array([t1Label,t2Label,t3Label,t4Label]),4)


        cb = plt.colorbar()
        cb.set_ticks(tickLoc)
        cb.set_ticklabels(tickLabels)

        plt.savefig(outDir+sessID+'+'+contrastDic[c]['name']+'_withColorbar.png')

        outFile=outDir+sessID+'_'+contrastDic[c]['name']+'_image.png'
        misc.imsave(outFile,np.uint8(funcOverlay))

        plt.clf()


def generate_preference_map(sourceRoot,sessID,analysisDir,avgFolder,nCond,motionCorrection=False,positiveThresh=True,\
                            thresh=0,useThresh2=True,thresh2=0):
    #Cesar Echavarria - 10/2016
    #VERIFY WE GOT NECESSARY VALUES
    if sourceRoot is None:
        raise TypeError("sourceRoot (directory) not specified!")
    if analysisDir is None:
        raise TypeError("analysisDir (directory) not specified!")
    if sessID is None:
        raise TypeError("sessID not specified!")
    if avgFolder is None:
        raise TypeError("avgFolder not specified!")
    if nCond is None:
        raise TypeError("nCond (number of conditions) not specified!")

    # DEFINE DIRECTORIES
    anatSource=sourceRoot+'Sessions/'+sessID+'/Surface/'
    motionDir=sourceRoot+'/Sessions/'+sessID+'/Motion/';
    motionFileDir=motionDir+'Registration/'
    inDir=analysisDir+'/AnalysisOutput/'+avgFolder+'/';
    outDir=analysisDir+'/Figures/'+avgFolder+'/preferenceMap/'

    if not os.path.exists(outDir):
        os.makedirs(outDir) 


    # READ IN SURFACE
    imFile=anatSource+'frame0.tiff'
    imSurf=cv2.imread(imFile,-1)
    imSurf=np.true_divide(imSurf,2**12)*2**8;

    if motionCorrection:
        #GET AND APPLY MOTION CORRECTED BOUNDARIES
        inFile=motionFileDir+sessID+'_motionCorrectedBoundaries.npz'
        f=np.load(inFile)
        boundaries=f['boundaries']
        edgeUp=boundaries[0]
        edgeDown=boundaries[1]
        edgeLeft=boundaries[2]
        edgeRight=boundaries[3]
        imSurf=imSurf[edgeUp:edgeDown,edgeLeft:edgeRight]

    funcOverlay=np.dstack((imSurf,imSurf,imSurf))
    szY,szX,nChannels=np.shape(funcOverlay)


    #LOAD IN MAP: RESPONSES Z-SCORED RELATIVE TO BASELINE
    allMaps=np.zeros((szY,szX,nCond))
    for cond in range(0,nCond):

        mapFile=inDir+sessID+'_condition'+str(cond)+'_zScoreMap.npz'  
        f=np.load(mapFile)
        respMap=f['respMap']
        szYmap,szXmap=np.shape(respMap)

        if not positiveThresh:#for negative thresholding
            respMap=-respMap

        #threshold based on z-score relative to baseline
        respMap=np.reshape(respMap,szYmap*szXmap)
        respMapThresh=np.copy(respMap)
        respMapThresh[respMap<=thresh]=0#threshold
        respMask=respMapThresh>0

        #Z-SCORE AGAIN
        respMap2=np.zeros(np.shape(respMap))
        if np.std(respMapThresh) != 0:
            respMap2[respMap>thresh]=(respMapThresh[respMap>thresh]-np.mean(respMap))/np.std(respMap)
        #store into array for all conditions
        tmpMap=np.reshape(respMap2,(szYmap,szXmap))
        tmpMap=cv2.resize(tmpMap,(szX,szY))
        allMaps[:,:,cond]=tmpMap

    #SET 2ND THRESHOLD
    if useThresh2:
        allMaps[allMaps<thresh2]=0

    #FIND CONDITION THAT EVOKE MAX RESPONSE FOR EACH PIXEL
    sumSigMap=np.sum(allMaps,2)
    maxPos=np.argmax(allMaps,2)+1
    maxPos[sumSigMap==0]=0

    if nCond==4:
        #MAKE MAPS
        colorCode=np.array([[255,0,0],[0,0,255],[255,255,0],[0,255,0]])
        constantSat=.4

        for channel in range(nChannels):
            imTemp=funcOverlay[:,:,channel]
            for pos in range(1,nCond+1):
                diffVal=(np.ones((szY,szX))*colorCode[pos-1,channel])-imTemp
                imTemp[maxPos==pos]=imTemp[maxPos==pos]+np.multiply(diffVal[maxPos==pos],constantSat)
                funcOverlay[:,:,channel]=imTemp[:,:]
        #SAVE MAPS
        if useThresh2:
            outFile=outDir+sessID+'_colorCodedPositionMap_thresh_'+str(thresh)+'_thresh2_'+str(thresh2)+'.png'
        else:
            outFile=outDir+sessID+'_colorCodedPositionMap_thresh_'+str(thresh)+'.png'
        misc.imsave(outFile,np.uint8(funcOverlay))

        #MAKE LEGEND
        nRows=2
        nCols=2
        legend=np.zeros((nRows*100,nCols*100,3))

        count=0
        for y in range(0,nRows):
            Y1=(100*(y))
            Y2=Y1+100
            for x in range(0,nCols):
                X1=(100*(x))
                X2=X1+100
                for dim in range(0,3):
                    legend[Y1:Y2,X1:X2,dim]=colorCode[count,dim];
                count=count+1;

        outFile=outDir+'colorCodedPositionLegend.png'   
        misc.imsave(outFile,np.uint8(legend))
    elif nCond==15:
        #MAKE MAPS
        colorDic=[]
        colorDic.append({'name':'AllPositions',\
                         'values':np.array([[255,0,0],[255,0,127],[127,0,127],[127,0,255],[0,0,255],\
                                            [255,128,0],[255,128,127],[55,55,127],[127,128,255],[0,128,255],\
                                            [255,255,0],[255,255,127],[127,255,127],[127,255,255],[0,255,255]])})
        colorDic.append({'name':'Elevation',\
                         'values':np.array([[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],\
                                            [255,128,0],[255,128,0],[255,128,0],[255,128,0],[255,128,0],\
                                            [0,255,0],[0,255,0],[0,255,0],[0,255,0],[0,255,0]])})
        colorDic.append({'name':'Azimuth',\
                         'values':np.array([[255,0,0],[255,100,127],[127,0,127],[127,0,255],[0,0,255],\
                                            [255,0,0],[255,100,127],[127,0,127],[127,0,255],[0,0,255],\
                                            [255,0,0],[255,100,127],[127,0,127],[127,0,255],[0,0,255]])})
        constantSat=.4
        for cMap in range(0,len(colorDic)):
            colorCode=colorDic[cMap]['values']
            for channel in range(nChannels):
                imTemp=funcOverlay[:,:,channel]
                for pos in range(1,nCond+1):
                    diffVal=(np.ones((szY,szX))*colorCode[pos-1,channel])-imTemp
                    imTemp[maxPos==pos]=imTemp[maxPos==pos]+np.multiply(diffVal[maxPos==pos],constantSat)
                    funcOverlay[:,:,channel]=imTemp[:,:]
            #SAVE MAPS
            if useThresh2:
                outFile=outDir+sessID+'_colorCodedPositionMap'+colorDic[cMap]['name']+'_thresh_'+str(thresh)+'_thresh2_'+str(thresh2)+'.png'
            else:
                outFile=outDir+sessID+'_colorCodedPositionMap'+colorDic[cMap]['name']+'_thresh_'+str(thresh)+'.png'
            misc.imsave(outFile,np.uint8(funcOverlay))

            #MAKE LEGEND
            nRows=3
            nCols=5
            legend=np.zeros((nRows*100,nCols*100,3))

            count=0
            for y in range(0,nRows):
                Y1=(100*(y))
                Y2=Y1+100
                for x in range(0,nCols):
                    X1=(100*(x))
                    X2=X1+100
                    for dim in range(0,3):
                        legend[Y1:Y2,X1:X2,dim]=colorCode[count,dim];
                    count=count+1;

            outFile=outDir+'colorCodedPositionLegend.png'   
            misc.imsave(outFile,np.uint8(legend))