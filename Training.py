
import os.path
import random
import tarfile
import re
import sys
import hashlib
import os
import tensorflow as tf
import numpy as np

from six.moves import urllib
from datetime import datetime
from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


MinNumberImages = 10
MinNumberImagesTrain = 100

MinNumberImagesTest = 3

MaxNumberImg = 2 ** 27 - 1 


LinkDataTrain = os.getcwd() + '/DataTrain'

LinkDataTest = os.getcwd() + "/DataTest/"

Graph = os.getcwd() + '/' + 'sauv_graph.pb'


Graphs = os.getcwd() + '/intermediate_graph'


IntNum = 0


Info = os.getcwd() + '/' + 'Info.txt'


TensorboardInfo = os.getcwd() + '/' + 'tensorboard_logs'


Step=500


Lrate = 0.01


TestEcho = 10


ValidEcho = 10


IntNum2 = 10


IntNum3 = 100

IntNum4 = -1



IntNum5 = 100


Boolean = False

LinkModel = os.getcwd() + "/" + "MLModel"


BottleneckLink = os.getcwd() + '/' + 'BottleneckVarTxt'


Result = 'FinalRes'


Arch = 'mobilenet_1.0_224'
#Arch = 'inception_v3'
#Arch = 'vgg-16'

#######################################################################################################################
def main():

    tf.logging.set_verbosity(tf.logging.INFO)

    if not ExistImage():
        return

    SysFun()

    ML_model = create_ML_model(Arch)

    if not ML_model:
        tf.logging.error('ModelNotFound')
        return -1

    DownloadModelFun(ML_model['data_url'])
    
    graph, TensorBott, ImgRest_tensor = (CreatModelFun(ML_model))

    ListImg = CreatListFun(LinkDataTrain, TestEcho, ValidEcho)
    Countt = len(ListImg.keys())
    if Countt == 0:
        tf.logging.error('No valid  ' + LinkDataTrain)
        return -1
    # end if
    if Countt == 1:
        tf.logging.error('Only one valid  ' + LinkDataTrain + ' - .')
        return -1
    # end if

   
    with tf.sessionion(graph=graph) as session:

        DataTensor, TensorImg = DecodeImgFun( ML_model['WidthIn'],
                                                                    ML_model['HeightIn'],
                                                                    ML_model['PathIn'],
                                                                    ML_model['MeanIn'],
                                                                    ML_model['input_std'])
        print(". . .")
        DisDataTensor = None
        DisDataImgTensor = None

        cache_Botts(session, ListImg, LinkDataTrain, BottleneckLink, DataTensor, TensorImg,
                              ImgRest_tensor, TensorBott, Arch)
        

        (train_step, cross_entropy, BottIn, TruIn, final_tensor) = AddTrain(len(ListImg.keys()),
                                                                                                                 Result,
                                                                                                                 TensorBott,
                                                                                                                 ML_model['TenSize'],
                                                                                                                 ML_model['Qunt'])

        EvalStep, Predict = AddStep(final_tensor, TruIn)


        merged = tf.summary.merge_all()
        TrainWi = tf.summary.FileWriter(TensorboardInfo + '/train', session.graph)
        ValidWi = tf.summary.FileWriter(TensorboardInfo + '/validation')

        init = tf.global_variables_initializer()
        session.run(init)

        for i in range(Step):

            (BottTrain, GTrain, _) = BottsCached(session, ListImg, IntNum3, 'training',
                                                                                           BottleneckLink, LinkDataTrain, DataTensor,
                                                                                           TensorImg, ImgRest_tensor, TensorBott,
                                                                                           Arch)

            RunSess, _ = session.run([merged, train_step], feed_dict={BottIn: BottTrain, TruIn: GTrain})
            TrainWi.add_summary(RunSess, i)


            IsLlastSt = (i + 1 == Step)
            if (i % IntNum2) == 0 or IsLlastSt:
                ErrorTrain, EntrpyVal = session.run([EvalStep, cross_entropy], feed_dict={BottIn: BottTrain, TruIn: GTrain})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, ErrorTrain * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, EntrpyVal))
                BottsValid, Validparam, _ = (BottsCached(session, ListImg, IntNum5, 'validation',
                                                                                                    BottleneckLink, LinkDataTrain, DataTensor,
                                                                                                    TensorImg, ImgRest_tensor, TensorBott,
                                                                                                    Arch))

                Validparam2, Validparam3 = session.run(
                    [merged, EvalStep], feed_dict={BottIn: BottsValid, TruIn: Validparam})
                ValidWi.add_summary(Validparam2, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' % (datetime.now(), i, Validparam3 * 100, len(BottsValid)))

            Frq = IntNum

            if (Frq > 0 and (i % Frq == 0) and i > 0):
                FileNamee = (Graphs + 'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save : ' + FileNamee)
                CreatGraph(session, graph, FileNamee)


        BottsTest, Tstparam, test_FileN = (BottsCached(session, ListImg, IntNum4, 'testing', BottleneckLink,
                                                                                             LinkDataTrain, DataTensor, TensorImg, ImgRest_tensor,
                                                                                             TensorBott, Arch))
        TstErrot, Predicts = session.run([EvalStep, Predict], feed_dict={BottIn: BottsTest, TruIn: Tstparam})
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (TstErrot * 100, len(BottsTest)))

        if Boolean:
            tf.logging.info('++++ test +++')
            for i, test_FileN1 in enumerate(test_FileN):
                if Predicts[i] != Tstparam[i]:
                    tf.logging.info('%70s  %s' % (test_FileN1, list(ListImg.keys())[Predicts[i]]))

        CreatGraph(session, graph, Graph)
        with gfile.FastGFile(Info, 'w') as f:
            f.write('\n'.join(ListImg.keys()) + '\n')


        print("Bien fait !!")


#######################################################################################################################
def ExistImage(): 

    if not os.path.exists(LinkDataTrain):
        return False

    class TrainingSubDir:
        # constructor
        def __init__(self):
            self.loc = ""
            self.numImages = 0

    trainingSubDirs = []

    for dirName in os.listdir(LinkDataTrain): 
        currentTrainingImagesSubDir = os.path.join(LinkDataTrain, dirName)
        if os.path.isdir(currentTrainingImagesSubDir): 
            trainingSubDir = TrainingSubDir() 
            trainingSubDir.loc = currentTrainingImagesSubDir
            trainingSubDirs.append(trainingSubDir) 

    if len(trainingSubDirs) == 0: 
        return False

    for trainingSubDir in trainingSubDirs:

        for FileN1 in os.listdir(trainingSubDir.loc):
            if FileN1.endswith(".jpeg"):
                trainingSubDir.numImages += 1

    for trainingSubDir in trainingSubDirs:
        if trainingSubDir.numImages < MinNumberImages:
            return False

    for trainingSubDir in trainingSubDirs:
        if trainingSubDir.numImages < MinNumberImagesTrain:
            print("NUmber Image < MIn Images")

    if not os.path.exists(LinkDataTest):
        return False

    numImagesInTestDir = 0
    for FileN1 in os.listdir(LinkDataTest):
        if FileN1.endswith(".jpeg"):
            numImagesInTestDir += 1

    if numImagesInTestDir < MinNumberImagesTest:
        return False

    return True


#######################################################################################################################
def SysFun(): 

    if tf.gfile.Exists(TensorboardInfo):
        tf.gfile.DeleteRecursively(TensorboardInfo)



    tf.gfile.MakeDirs(TensorboardInfo)

    if IntNum > 0:
        makeDirIfDoesNotExist(Graphs)

    return

def makeDirIfDoesNotExist(LName):

    if not os.path.exists(LName):
        os.makedirs(LName)


def create_ML_model(Arch):

    Arch = Arch.lower() 
    is_quantized = False
    if Arch == 'inception_v3':
        
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        
        TensorBott_name = 'pool_3/_reshape:0'
        TenSize = 2048
        WidthIn = 299
        HeightIn = 299
        PathIn = 3
        TensorInput_name = 'Mul:0'
        model_file_name = 'classify_image_FileGraph.pb'
        MeanIn = 128
        input_std = 128
    elif Arch.startswith('mobilenet_'):
        parts = Arch.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand Arch name '%s'", Arch)
            return None
        # end if
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(""""The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25', but found '%s' for Arch '%s'""", version_string, Arch)
            return None
        # end if
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and size_string != '160' and size_string != '128'):
            tf.logging.error("""The Mobilenet input size should be '224', '192', '160', or '128', but found '%s' for Arch '%s'""", size_string, Arch)
            return None
        # end if
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error(
                    "Couldn't understand Arch suffix '%s' for '%s'", parts[3], Arch)
                return None
            is_quantized = True
        # end if

        if is_quantized:
            data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
            data_url += version_string + '_' + size_string + '_quantized_frozen.tgz'
            TensorBott_name = 'MobilenetV1/Predicts/Reshape:0'
            TensorInput_name = 'Placeholder:0'
            LinkModel_name = ('mobilenet_v1_' + version_string + '_' + size_string + '_quantized_frozen')
            model_BName = 'quantized_frozen_graph.pb'
        else:
            data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
            data_url += version_string + '_' + size_string + '_frozen.tgz'
            TensorBott_name = 'MobilenetV1/Predicts/Reshape:0'
            TensorInput_name = 'input:0'
            LinkModel_name = 'mobilenet_v1_' + version_string + '_' + size_string
            model_BName = 'frozen_graph.pb'
        # end if

        TenSize = 1001
        WidthIn = int(size_string)
        HeightIn = int(size_string)
        PathIn = 3
        model_file_name = os.path.join(LinkModel_name, model_BName)
        MeanIn = 127.5
        input_std = 127.5
    elif Arch == 'vgg-16':
        data_url = 'https://s3.amazonaws.com/cadl/models/vgg16.tfmodel'
        TensorBott_name = 'block2_conv2_b_1:0'
        TenSize = 2048
        WidthIn = 224
        HeightIn = 224
        PathIn = 3
        TensorInput_name = 'images:0'
        model_file_name = 'vgg16.tfmodel.pb'
        MeanIn = 128
        input_std = 128
    else:
        tf.logging.error("Couldn't understand Arch name '%s'", Arch)
        raise ValueError('Unknown Arch', Arch)
    # end if

    return {'data_url': data_url, 'TensorBott_name': TensorBott_name, 'TenSize': TenSize,
            'WidthIn': WidthIn, 'HeightIn': HeightIn, 'PathIn': PathIn, 'TensorInput_name': TensorInput_name,
            'model_file_name': model_file_name, 'MeanIn': MeanIn, 'input_std': input_std, 'Qunt': is_quantized, }


def DownloadModelFun(data_url):

    LinkDer = LinkModel
    if not os.path.exists(LinkDer):
        os.makedirs(LinkDer) 

    FileN1 = data_url.split('/')[-1]
    PathFile2 = os.path.join(LinkDer, FileN1)
    if not os.path.exists(PathFile2):
        
        def _progress(count, IntSize, FnSize):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (FileN1, float(count * IntSize) / float(FnSize) * 100.0))
            sys.stdout.flush()

        PathFile2, _ = urllib.request.urlretrieve(data_url, PathFile2, _progress)
        print()
        StatInf = os.stat(PathFile2)
        tf.logging.info('Successfully download ' + str(FileN1) + ', size (bytes)= ' + str(StatInf.st_size) + '_')

        tarfile.open(PathFile2, 'r:gz').extractall(LinkDer)
    else:
        print('error')

def CreatModelFun(ML_model):

    with tf.Graph().as_default() as graph:
        PathModel = os.path.join(LinkModel, ML_model['model_file_name']) 
        with gfile.FastGFile(PathModel, 'rb') as f:
            FileGraph = tf.GraphDef()
            FileGraph.ParseFromString(f.read())
            TensorBott, TensorInput = (tf.import_FileGraph(FileGraph, name='', return_elements=[ML_model['TensorBott_name'], ML_model['TensorInput_name'],]))

    return graph, TensorBott, TensorInput

def CreatListFun(LinkImg, TestEcho, ValidEcho):

    if not gfile.Exists(LinkImg):
        tf.logging.error(">>>>>>> '" + LinkImg + "' not found.")
        return None

    result = {}

    LinkSubs = [x[0] for x in gfile.Walk(LinkImg)]


    IsBoolean = True
    for LinkSub in LinkSubs:
        

        if IsBoolean:
            IsBoolean = False


        LName = os.path.basename(LinkSub) 
        if LName == LinkImg:
            continue



        Ext = ['jpg', 'jpeg']
        file_list = []
        tf.logging.info(">>>>>>>'" + LName + "'")
        for extension in Ext:
            file_glob = os.path.join(LinkImg, LName, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))



        if not file_list:
            tf.logging.warning('No files found')
            continue


        
        if len(file_list) < 20:
            tf.logging.warning('error!!!!')
        elif len(file_list) > MaxNumberImg:
            tf.logging.warning('Valid')


        LabN = re.sub(r'[^a-z0-9]+', ' ', LName.lower())
        ImgTrainArray = []
        ImgTestArray = []
        validation_images = []
        for file_name in file_list:
            BName = os.path.basename(file_name)

            Hname = re.sub(r'_nohash_.*$', '', file_name)

            Hname_hashed = hashlib.sha1(compat.as_bytes(Hname)).hexdigest()
            PercentageImg = ((int(Hname_hashed, 16) % (MaxNumberImg + 1)) * (100.0 / MaxNumberImg))
            if PercentageImg < ValidEcho:
                validation_images.append(BName)
            elif PercentageImg < (TestEcho + ValidEcho):
                ImgTestArray.append(BName)
            else:
                ImgTrainArray.append(BName)

        result[LabN] = {'dir': LName, 'training': ImgTrainArray, 'testing': ImgTestArray, 'validation': validation_images,}
    return result

def DecodeImgFun(WidthIn, HeightIn, PathIn, MeanIn, SInput):

    ImgDataDecode = tf.placeholder(tf.string, name='DecodeJPGInput')
    ImgDecoding = tf.image.decode_jpeg(ImgDataDecode, channels=PathIn)
    ImgDecoding_as_float = tf.cast(ImgDecoding, dtype=tf.float32)
    ImgDecoding_4d = tf.expand_dims(ImgDecoding_as_float, 0)
    Shap = tf.stack([HeightIn, WidthIn])
    Shap_as_int = tf.cast(Shap, dtype=tf.int32)
    ImgRest = tf.image.resize_bilinear(ImgDecoding_4d, Shap_as_int)
    imgSubtract = tf.subtract(ImgRest, MeanIn)
    ImgMultipl = tf.multiply(imgSubtract, 1.0 / input_std)
    return ImgDataDecode, ImgMultipl



def cache_Botts(session, ListImg, LinkImg, BottleneckLink, DataTensor, TensorImg,
                      TensorInput, TensorBott, Arch):

    MBotts = 0
    makeDirIfDoesNotExist(BottleneckLink)
    for LabN, Listl in ListImg.items():
        for ClassOutput in ['training', 'testing', 'validation']:
            ClassOutput_list = Listl[ClassOutput]
            for Index, unused_BName in enumerate(ClassOutput_list):
                GetBottCreate(session, ListImg, LabN, Index, LinkImg, ClassOutput, BottleneckLink,
                                         DataTensor, TensorImg, TensorInput, TensorBott, Arch)
            MBotts += 1
            if MBotts % 100 == 0:
                tf.logging.info(str(MBotts) + '  files created.')

def GetBottCreate(session, ListImg, LabN, Index, LinkImg, ClassOutput, BottleneckLink, DataTensor,
                             TensorImg, TensorInput, TensorBott, Arch):

    Listl = ListImg[LabN]
    LinkSub = Listl['dir']
    LinkSLink = os.path.join(BottleneckLink, LinkSub)
    makeDirIfDoesNotExist(LinkSLink)
    BottLinkk = GetBottLink(ListImg, LabN, Index, BottleneckLink, ClassOutput, Arch)
    if not os.path.exists(BottLinkk):
        CreateBottFileFun(BottLinkk, ListImg, LabN, Index, LinkImg, ClassOutput, session, DataTensor,
                               TensorImg, TensorInput, TensorBott)



    with open(BottLinkk, 'r') as bottleneck_file:
        Bottstr = bottleneck_file.read()
 

    BottArray = []
    errorOccurred = False
    try:

        BottArray = [float(individualString) for individualString in Bottstr.split(',')]
    except ValueError:
        tf.logging.warning('Invalid')
        errorOccurred = True


    if errorOccurred:

        CreateBottFileFun(BottLinkk, ListImg, LabN, Index, LinkImg, ClassOutput, session,
                               DataTensor, TensorImg, TensorInput, TensorBott)


        with open(BottLinkk, 'r') as bottleneck_file:
            Bottstr = bottleneck_file.read()

        BottArray = [float(individualString) for individualString in Bottstr.split(',')]

    return BottArray
# end function

def GetBottLink(ListImg, LabN, Index, BottleneckLink, ClassOutput, Arch):

    return LinkImg(ListImg, LabN, Index, BottleneckLink, ClassOutput) + '_CHUFES_' + Arch + '.txt'
def CreateBottFileFun(BottLinkk, ListImg, LabN, Index,
                           LinkImg, ClassOutput, session, DataTensor,
                           TensorImg, TensorInput,
                           TensorBott):

    tf.logging.info('Creating bottleneck at ' + BottLinkk)
    LinkImagp = LinkImg(ListImg, LabN, Index, LinkImg, ClassOutput)
    if not gfile.Exists(LinkImagp):
        tf.logging.fatal('File does not exist %s', LinkImagp)

    Data_imgg = gfile.FastGFile(LinkImagp, 'rb').read()
    try:
        BottVal = BottImagesFun(session, Data_imgg, DataTensor, TensorImg, TensorInput, TensorBott)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (LinkImagp, str(e)))


    bottleneck_string = ','.join(str(x) for x in BottVal)
    with open(BottLinkk, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def BottImagesFun(session, Data_imgg, Data_imggTensor, TensorImg, TensorInput, TensorBott):


    resized_input_values = session.run(TensorImg, {Data_imggTensor: Data_imgg})
    BottVal = session.run(TensorBott, {TensorInput: resized_input_values})
    BottVal = np.squeeze(BottVal)
    return BottVal


def LinkImg(ListImg, LabN, Index, LinkImg, ClassOutput):

    if LabN not in ListImg:
        tf.logging.fatal('Label does not exist %s.', LabN)

    Listl = ListImg[LabN]
    if ClassOutput not in Listl:
        tf.logging.fatal('ClassOutput does not exist %s.', ClassOutput)
  
    ClassOutput_list = Listl[ClassOutput]
    if not ClassOutput_list:
        tf.logging.fatal('Label %s has no images in the ClassOutput %s.', LabN, ClassOutput)

    InxMod = Index % len(ClassOutput_list)
    BName = ClassOutput_list[InxMod]
    LinkSub = Listl['dir']
    LinkFull = os.path.join(LinkImg, LinkSub, BName)
    return LinkFull


def AddTrain(Countt, Result, TensorBott, TenSize, Qunt): #creatTensorboard

    with tf.name_scope('input'):
        BottIn = tf.placeholder_with_default(TensorBott, shape=[None, TenSize], name='BottleneckInputPlaceholder')
        TruIn = tf.placeholder(tf.int64, [None], name='GroundTruthInput')

    LayN = 'final_training_ops'
    with tf.name_scope(LayN):
        w1 = None
        w2 = None
        with tf.name_scope('weights'):
            VarDef = tf.truncated_normal([TenSize, Countt], stddev=0.001)
            LayDef = tf.Variable(VarDef, name='final_weights')
            if Qunt:
                w1 = quant_ops.MovingAvgQuantize(LayDef, is_training=True)
                TensorboardFun(w1)

            TensorboardFun(LayDef)

        with tf.name_scope('biases'):
            BiasesVal = tf.Variable(tf.zeros([Countt]), name='final_biases')
            if Qunt:
                w2 = quant_ops.MovingAvgQuantize(BiasesVal, is_training=True)
                TensorboardFun(w2)

            TensorboardFun(BiasesVal)

        with tf.name_scope('Wx_plus_b'):
            if Qunt:
                logits = tf.matmul(BottIn, w1) + w2
                logits = quant_ops.MovingAvgQuantize(logits, init_min=-32.0, init_max=32.0, is_training=True, num_bits=8,
                                                     narrow_range=False, ema_decay=0.5)
                tf.summary.histogram('pre_activations', logits)
            else:
                logits = tf.matmul(BottIn, LayDef) + BiasesVal
                tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=Result)

    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=TruIn, logits=logits)


    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(Lrate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, BottIn, TruIn, final_tensor)


def TensorboardFun(var):

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # end with
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def AddStep(ResTensor, TensorG):


    with tf.name_scope('accuracy'):
        with tf.name_scope('CPredict'):
            Predict = tf.argmax(ResTensor, 1)
            CPredict = tf.equal(Predict, TensorG)

        with tf.name_scope('accuracy'):
            EvalStep = tf.reduce_mean(tf.cast(CPredict, tf.float32))

    tf.summary.scalar('accuracy', EvalStep)
    return EvalStep, Predict



def BottsCached(session, ListImg, Many, ClassOutput, BottleneckLink, LinkImg, DataTensor,
                                  TensorImg, TensorInput, TensorBott, Arch):

    Countt = len(ListImg.keys())
    Botts = []
    TrG = []
    FileN = []
    if Many >= 0:
        for unused_i in range(Many):
            LIndx = random.randrange(Countt)
            LabN = list(ListImg.keys())[LIndx]
            InxI = random.randrange(MaxNumberImg + 1)
            ImgN = LinkImg(ListImg, LabN, InxI, LinkImg, ClassOutput)
            bottleneck = GetBottCreate(session, ListImg, LabN, InxI, LinkImg, ClassOutput, BottleneckLink,
                                                  DataTensor, TensorImg, TensorInput, TensorBott, Arch)
            Botts.append(bottleneck)
            TrG.append(LIndx)
            FileN.append(ImgN)

    else:

        for LIndx, LabN in enumerate(ListImg.keys()):
            for InxI, ImgN in enumerate(ListImg[LabN][ClassOutput]):
                ImgN = LinkImg(ListImg, LabN, InxI, LinkImg, ClassOutput)
                bottleneck = GetBottCreate(session, ListImg, LabN, InxI, LinkImg, ClassOutput, BottleneckLink,
                                                      DataTensor, TensorImg, TensorInput, TensorBott, Arch)
                Botts.append(bottleneck)
                TrG.append(LIndx)
                FileN.append(ImgN)
    return Botts, TrG, FileN


def CreatGraph(session, graph, graphName):
    FileGraph = graph_util.convert_variables_to_constants(session, graph.as_FileGraph(), [Result])
    with gfile.FastGFile(graphName, 'wb') as f:
        f.write(FileGraph.SerializeToString())
    return

#Main
if __name__ == '__main__':
    main()