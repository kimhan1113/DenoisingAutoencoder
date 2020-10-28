import os
import random
import glob
import cv2
from tqdm import tqdm
from tkinter import filedialog
import json
import numpy as np
import shutil
from scipy import ndimage
import matplotlib.pyplot as plt
import copy
from core.ImageProcessing import *

PackageTypeResistor=0
PackageTypeCapacitor=1
PackageTypeTantal=2
PackageTypeInductor=3
PackageTypeDiode=6
PackageTypeMelf=7
PackageTypeAluminum_capacitor=8
PackageTypeLed=9
PackageTypeSOD=10
PackageTypeCrystal=11
PackageTypeBGA=13
PackageTypeSOIC=15
PackageTypeDIP=26
PackageTypeSOJ=27
PackageTypeQFP=28
PackageTypeQFN=32
PackageTypeSOT=34
PackageTypeConnector=36
PackageTypeDPAK=38
PackageTypeThroughHole=40
PackageTypePressfit=41
PackageTypeOther=43
PackageTypeChipArray=44
PackageTypeShield=45

class metaClass:

    def __init__(self):
        self.AnalyticFilterMode = False
        self.PackageNumber = {}
        self.PackageProperty_On_Off = False
        self.PackagePropertyRatio ={
        'PackageTypeResistor': 0.0,
        'PackageTypeCapacitor': 0.0,
        'PackageTypeTantal': 0.0,
        'PackageTypeInductor': 0.0,
        'PackageTypeDiode': 0.0,
        'PackageTypeMelf': 0.0,
        'PackageTypeAluminum_capacitor': 0.0,
        'PackageTypeLed': 0.0,
        'PackageTypeSOD': 0.0,
        'PackageTypeCrystal': 0.0,
        'PackageTypeBGA': 0.0,
        'PackageTypeSOIC': 0.0,
        'PackageTypeDIP': 0.0,
        'PackageTypeSOJ': 0.0,
        'PackageTypeQFP': 0.0,
        'PackageTypeQFN': 0.0,
        'PackageTypeSOT': 0.0,
        'PackageTypeConnector': 0.0,
        'PackageTypeDPAK': 0.0,
        'PackageTypeThroughHole': 0.0,
        'PackageTypePressfit': 0.0,
        'PackageTypeOther': 0.0,
        'PackageTypeChipArray': 0.0,
        'PackageTypeShield': 0.0,
        'None':0.0
        }

        self.DataQ = []
        self.CompList = []
        self.CompData = {
            "RootPath": None,
             "FileName": None,
            "BoardFolderName": None,
            "ChannelFolderName": None,
            "ChannelComb": None,
            "CompName": None,
            "PackageTypeName": None,
            "PackageTypeID": None,
            "DataOrigin": None,
            "NoisePower_SAG": 0,
            "NoisePower_RMS": 0,
            "AngleMap_Path": None,
            "HeightMap_Path":None,
            "K3D_Path": None,
            "TopImage_Path": None,
            "MiddleImage_Path" : None,
            "BottomImage_Path": None,
            "PariedCompData_Path" : None,
            "AttentionMap_Path" : None,
            "ImprovedHeightMap_Path" : None
        }
        self.PackageSAGNoiseAVG = {'PackageTypeResistor': 0.0,
        'PackageTypeCapacitor': 0.0,
        'PackageTypeTantal': 0.0,
        'PackageTypeInductor': 0.0,
        'PackageTypeDiode': 0.0,
        'PackageTypeMelf': 0.0,
        'PackageTypeAluminum_capacitor': 0.0,
        'PackageTypeLed': 0.0,
        'PackageTypeSOD': 0.0,
        'PackageTypeCrystal': 0.0,
        'PackageTypeBGA': 0.0,
        'PackageTypeSOIC': 0.0,
        'PackageTypeDIP': 0.0,
        'PackageTypeSOJ': 0.0,
        'PackageTypeQFP': 0.0,
        'PackageTypeQFN': 0.0,
        'PackageTypeSOT': 0.0,
        'PackageTypeConnector': 0.0,
        'PackageTypeDPAK': 0.0,
        'PackageTypeThroughHole': 0.0,
        'PackageTypePressfit': 0.0,
        'PackageTypeOther': 0.0,
        'PackageTypeChipArray': 0.0,
        'PackageTypeShield': 0.0,
        'None':0.0}
        self.PackageRMSNoiseAVG = {'PackageTypeResistor': 0.0,
        'PackageTypeCapacitor': 0.0,
        'PackageTypeTantal': 0.0,
        'PackageTypeInductor': 0.0,
        'PackageTypeDiode': 0.0,
        'PackageTypeMelf': 0.0,
        'PackageTypeAluminum_capacitor': 0.0,
        'PackageTypeLed': 0.0,
        'PackageTypeSOD': 0.0,
        'PackageTypeCrystal': 0.0,
        'PackageTypeBGA': 0.0,
        'PackageTypeSOIC': 0.0,
        'PackageTypeDIP': 0.0,
        'PackageTypeSOJ': 0.0,
        'PackageTypeQFP': 0.0,
        'PackageTypeQFN': 0.0,
        'PackageTypeSOT': 0.0,
        'PackageTypeConnector': 0.0,
        'PackageTypeDPAK': 0.0,
        'PackageTypeThroughHole': 0.0,
        'PackageTypePressfit': 0.0,
        'PackageTypeOther': 0.0,
        'PackageTypeChipArray': 0.0,
        'PackageTypeShield': 0.0,
        'None':0.0}

    def generateRMS_SAGinMeta(self, RootPath = None, metaDataList = []):
        new_metaDataList = []
        for k in tqdm(range(len(metaDataList))):
            SourceHeight, RefHeight = self.loadHeight(RootPath, metaDataList[k])
            # SourceHeight, RefHeight = mg.loadHeight_onlySRC(RootPath, metaDataList[k])

            Noise_SAG = self.SAG(SourceHeight)
            Noise_RMS = self.RMS(SourceHeight, RefHeight)

            metaDataList[k]["NoisePower_SAG"] = Noise_SAG
            metaDataList[k]["NoisePower_RMS"] = Noise_RMS

            self.writeMetaData(RootPath, metaDataList[k])
        print("Complete calculation of RMS and SAG..")

        new_metaDataList = self.readMetaFiles(RootPath)
        return new_metaDataList

    def readMetaFiles(self, RootPath = None ):

        metafileList = glob.glob1(RootPath, "*.json")
        metaDataList = []
        for k in tqdm(range(len(metafileList))):
            try:
                self.CompData = self.readMetaData(RootPath, metafileList[k])
                metaDataList.append(self.CompData)
            except:
                continue

    def generateMeta(self, RootPath =  None, SavePath = None):

        if not os.path.exists(SavePath):
            os.makedirs(SavePath)

        total = 0
        count = 0

        for dirpath, dirnames, files in os.walk(RootPath):

            if len(dirnames) > 0:
                total += len(dirnames)

            if RootPath == dirpath:
                continue

            if len(files) == 0:
                continue

            ProgressString = str(count) + " / " + str(total)
            # Folders = dirpath.split(os.path.sep)

            # if not Folders[-1].startswith("Copy"):
            #     continue

            file_csv = [file for file in files if file.endswith("csv")]

            for fileIndex in tqdm(range(len(file_csv)), desc=ProgressString):
                self.CompData["RootPath"] = dirpath
                self.FilePaser(file_csv[fileIndex])
                fileName = SavePath + "/" + self.CompData["FileName"] + ".json"
                self.CompData["PariedCompData_Path"] = SavePath + '/' + self.CompData["FileName"].split("[Conf]")[
                    0] + "[Conf]8_[AF]1_[RBF]1_[CH]11111111_.json"
                # mg.CompData["PariedCompData_Path"] = SavePath + '/' + mg.CompData["FileName"].split("[Conf]")[0] + "[Conf]8_[AF]1_[RBF]0_[CH]11111111_.json"
                jstring = json.dumps(self.CompData, indent=4)
                f = open(fileName, "w")
                f.write(jstring)
                f.close()

            count += 1

    # self.PackageNoiseAVG Dictionary에 Package 별 평균 SAG, RMS noise 값이 저장됨
    def calculatePackageNoiseAVG(self, metaDataList = None):
        if len(self.PackageNumber) == 0:
            print("Package Number가 0입니다. getPackageDistributionFromMeta 함수를 우선 실행해주세요.")
            exit(0)

        for k in tqdm(range(len(metaDataList)),desc="Calculating Average Noise in each package.."):
            pkgName = metaDataList[k].get("PackageTypeName")
            sagNoise = metaDataList[k].get("NoisePower_SAG")
            rmsNoise = metaDataList[k].get("NoisePower_RMS")

            self.PackageSAGNoiseAVG[pkgName] += sagNoise
            self.PackageRMSNoiseAVG[pkgName] += rmsNoise

        for key, value in self.PackageNumber.items():
            self.PackageSAGNoiseAVG[key] /= value
            self.PackageRMSNoiseAVG[key] /= value

    def RMS(self, src=None, ref=None):
        try:
            # srcnp = np.array(src)
            # refnp = np.array(ref)
            # temp = srcnp - refnp
            # temp = temp * temp
            # temp = np.sqrt(temp)
            # res = temp/len(src)

            res = np.sqrt(np.sum(np.square(np.subtract(src, ref))) / len(src))
            return res
        except:
            return 0

    # BodyHeight 구하는 함수
    # Height Map의 모든 Height 값을 오름차순으로 정렬 후, 상위 3% 제거한 뒤 그 다음 하위 8% height들의 평균값을 취함
    def getStatisticBodyHeight(self, input =None):

        input_array2D = np.array(input)
        input_array1D = input_array2D.reshape(input_array2D.shape[0] * input_array2D.shape[1])
        input_list = input_array1D.tolist()
        input_list.sort(reverse=True)

        for i in range(0, int(len(input_list) * 0.03)):
            input_list.pop(i)

        input_list_8 = list()

        for i in range(0, int(len(input_list) * 0.08)):
            input_list_8.append(input_list[i])

        input_mean = np.mean(np.array(input_list_8))
        return input_mean

    def Off(self, src=None, ref=None):

        src_mean = self.getStatisticBodyHeight(src)
        ref_mean = self.getStatisticBodyHeight(ref)

        res = abs(src_mean - ref_mean)

        return res

    def SAG(self, src=None):
        kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        Gx = ndimage.convolve(src, kx, mode='constant', cval=0.0)
        Gy = ndimage.convolve(src, ky, mode='constant', cval=0.0)

        BodyHeight = self.getStatisticBodyHeight(src)

        res = (np.sqrt(np.sum(np.add(np.square(Gx), np.square(Gy)))) / BodyHeight) / len(src)

        return res

    # json 파일(meta file)을 읽어서 CompData로 변환
    def readMetaData(self, RootPath = None, metaFilePath = None):
        if RootPath == None:
            jstring = open(metaFilePath, "r").read()
        else:
            jstring = open(RootPath + '/' + metaFilePath, "r").read()

        CompData = json.loads(jstring)
        return CompData

    # metaData를 json파일로 출력 저장
    def writeMetaData(self, RootPath = None, metaData = None):
        jstring = json.dumps(metaData, indent=4)
        fileName = RootPath + "/" + metaData["FileName"] + ".json"
        f = open(fileName, "w")
        f.write(jstring)
        f.close()

    # Height Map을 읽는 함수
    # Ref만 읽을 수도 있고, Src와 Ref를 함께 읽을 수도 있어서 분리함
    def loadHeight(self, RootPath = None, metaData = None):
        if metaData.get("DataOrigin") == 0:
            HeightMapPath = metaData.get("HeightMap_Path")
            PairPath = metaData.get("PariedCompData_Path")

            CompData = self.readMetaData(None, PairPath)
            PairedHeightMapPath = CompData.get("HeightMap_Path")

            SrcHeightMap = np.genfromtxt(HeightMapPath, delimiter=",")[:, :-1]
            RefHeightMap = np.genfromtxt(PairedHeightMapPath, delimiter=",")[:, :-1]

            return SrcHeightMap, RefHeightMap

        elif metaData.get("DataOrigin") == 1:
            HeightMapPath = metaData.get("HeightMap_Path")
            SrcHeightMap = np.genfromtxt(HeightMapPath, delimiter=",")[:, :-1]

            return SrcHeightMap, None

    def loadHeight_onlySRC(self, RootPath=None, metaData=None):
        if metaData.get("DataOrigin") == 0:
            HeightMapPath = metaData.get("HeightMap_Path")

            SrcHeightMap = np.genfromtxt(HeightMapPath, delimiter=",")[:, :-1]

            return SrcHeightMap, None

        elif metaData.get("DataOrigin") == 1:
            HeightMapPath = metaData.get("HeightMap_Path")
            SrcHeightMap = np.genfromtxt(HeightMapPath, delimiter=",")[:, :-1]

            return SrcHeightMap, None

    #PackageIndex를 PackageTypeName으로 변환해서 return
    def GetPackageName(self, PackageIdx=None):
        if PackageIdx == PackageTypeResistor:
            Package_TypeName = "PackageTypeResistor"
        elif PackageIdx == PackageTypeCapacitor:
            Package_TypeName = "PackageTypeCapacitor"
        elif PackageIdx == PackageTypeTantal:
            Package_TypeName = "PackageTypeTantal"
        elif PackageIdx == PackageTypeInductor:
            Package_TypeName = "PackageTypeInductor"
        elif PackageIdx == PackageTypeDiode:
            Package_TypeName = "PackageTypeDiode"
        elif PackageIdx == PackageTypeMelf:
            Package_TypeName = "PackageTypeMelf"
        elif PackageIdx == PackageTypeAluminum_capacitor:
            Package_TypeName = "PackageTypeAluminum_capacitor"
        elif PackageIdx == PackageTypeLed:
            Package_TypeName = "PackageTypeLed"
        elif PackageIdx == PackageTypeSOD:
            Package_TypeName = "PackageTypeSOD"
        elif PackageIdx == PackageTypeCrystal:
            Package_TypeName = "PackageTypeCrystal"
        elif PackageIdx == PackageTypeBGA:
            Package_TypeName = "PackageTypeBGA"
        elif PackageIdx == PackageTypeSOIC:
            Package_TypeName = "PackageTypeSOIC"
        elif PackageIdx == PackageTypeDIP:
            Package_TypeName = "PackageTypeDIP"
        elif PackageIdx == PackageTypeSOJ:
            Package_TypeName = "PackageTypeSOJ"
        elif PackageIdx == PackageTypeQFP:
            Package_TypeName = "PackageTypeQFP"
        elif PackageIdx == PackageTypeQFN:
            Package_TypeName = "PackageTypeQFN"
        elif PackageIdx == PackageTypeSOT:
            Package_TypeName = "PackageTypeSOT"
        elif PackageIdx == PackageTypeConnector:
            Package_TypeName = "PackageTypeConnector"
        elif PackageIdx == PackageTypeDPAK:
            Package_TypeName = "PackageTypeDPAK"
        elif PackageIdx == PackageTypeThroughHole:
            Package_TypeName = "PackageTypeThroughHole"
        elif PackageIdx == PackageTypePressfit:
            Package_TypeName = "PackageTypePressfit"
        elif PackageIdx == PackageTypeOther:
            Package_TypeName = "PackageTypeOther"
        elif PackageIdx == PackageTypeChipArray:
            Package_TypeName = "PackageTypeChipArray"
        elif PackageIdx == PackageTypeShield:
            Package_TypeName = "PackageTypeShield"
        else:
            Package_TypeName = "None"

        return Package_TypeName

    # MetaDataList를 넣으면, SAG와 RMS에 대해 분포를 계산
    def saveNoiseDistribution(self, metaDataList = {}):
        NoiseSAG = [0] * len(metaDataList)
        NoiseRMS = [0] * len(metaDataList)

        for k in tqdm(range(len(metaDataList)), desc="Generating noise histogram of meta data"):
            NoiseSAG[k] = metaDataList[k].get("NoisePower_SAG")
            NoiseRMS[k] = metaDataList[k].get("NoisePower_RMS")

        plt.figure(1)
        xAxis_SAG = []
        xAxis_RMS = []
        for k in range(0, 31):
            xAxis_SAG.append(k * (1000 / 30))
            xAxis_RMS.append(k * (0.1 / 33))

        plt.hist(NoiseSAG, xAxis_SAG)
        plt.figure(2)
        plt.hist(NoiseRMS, xAxis_RMS)
        plt.show()

    #
    def savePackageDistribution(self, PackageNumber = {}):
        totalNum = 0
        if len(PackageNumber) == 0:
            print("Package Number Data가 없습니다. 확인해주세요. ")
            exit(0)

        for key, value in PackageNumber.items():
            totalNum += value
        if totalNum == 0:
            print("Package 별 Data 수 정보가 없습니다. 확인해주세요. ")
            exit(0)

        Number = []
        Name2 = []
        Name1 = []

        for key, value in PackageNumber.items():
            displayString = key + ' ' + str(round(value/totalNum * 100, 2)) + '%' + '   ' + str(value) + '/' + str(totalNum)
            Name2.append(displayString)
            Number.append(value)
            Name1.append(key)

        plt.figure('Package Distribution', figsize=(15, 5), facecolor='white')
        pie = plt.pie(Number, labels=Name1)
        plt.axis('equal')
        plt.legend(pie[0], Name2, bbox_to_anchor=(0.0, 1), loc="upper left",
                   bbox_transform=plt.gcf().transFigure)
        plt.show()
        plt.savefig('distribution.png')
        plt.close()

    def FilePaser(self, file):

        RootPath = self.CompData["RootPath"]

        # [Arr]2_[Crd]C0501_[Pkg]A2C00010738_[PkgType]1_[Conf]_[AF]0_[RBF]0_[CH]0000110000000000_[Type]2_AfterRBF

        # FileName
        FileName = file.split("[Type]")[0]
        self.CompData["FileName"] = FileName

        # CompName
        CompName = file.split("[Conf]")[0]
        self.CompData["CompName"] = CompName

        # ChannelComb
        ChannelComb = os.path.basename(file).split("[CH]")[1].split("[Type]")[0].split("_")[0]
        self.CompData["ChannelComb"] = ChannelComb

        # PackageTypeID
        PackageTypeID = CompName.split("[PkgType]")[1].split("_")[0]
        self.CompData["PackageTypeID"] = int(PackageTypeID)

        # PackageTypeName
        PackageTypeName = self.GetPackageName(int(PackageTypeID))
        self.CompData["PackageTypeName"] = PackageTypeName

        #ImagePaths
        typeNum = file.split("[Type]")[1].split("_")[0]
        mainName = file.split("[Type]")[0]

        self.CompData["AngleMap_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_Ang.bmp"
        self.CompData["HeightMap_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_AfterRBF.csv"
        self.CompData["K3D_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_K3D.K3D"
        self.CompData["TopImage_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_Top.bmp"
        self.CompData["MiddleImage_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_Mid.bmp"
        self.CompData["BottomImage_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_Bot.bmp"
        self.CompData["AttentionMap_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_ObjectBaseMap.bmp"
        self.CompData["ImprovedHeightMap_Path"] = RootPath + "/" + mainName + "[Type]" + typeNum + "_ImprovedHeight.csv"

        folders = RootPath.split(os.path.sep)
        ChannelFolderName = folders[-1]
        BoardFolderName = folders[-2]

        # Board Folder Name
        self.CompData["BoardFolderName"] = BoardFolderName

        # Channel Folder Name
        self.CompData["ChannelFolderName"] = ChannelFolderName

        # Data Origin
        chIdx = ChannelFolderName.split("[chIdx]")[1].split("[AF]")[0].split("_")[0]
        self.CompData["DataOrigin"] = 0
        if chIdx == "-1":
            self.CompData["DataOrigin"] = 1

        return

    # "RootPath": None,
    # "FileName": None,
    # "BoardFolderName": None,
    # "ChannelFolderName": None,
    # "ChannelComb": None,
    # "CompName": None,
    # "PackageTypeName": None,
    # "PackageTypeID": None,
    # "DataOrigin": None,
    # "NoisePower_SAG": 0,
    # "NoisePower_RMS": 0,
    # "AngleMap_Path": None,
    # "HeightMap_Path": None,
    # "K3D_Path": None,
    # "TopImage_Path": None,
    # "MiddleImage_Path": None,
    # "BottomImage_Path": None,
    # "PariedCompData_Path": None
    def filter_BoardName(self, metaDataList = None, BoardFolderNameList = None):
        output = []
        k = 0
        MaxMeta = len(metaDataList)

        for q in range(len(BoardFolderNameList)):
            while k < MaxMeta:
                if BoardFolderNameList[q] == metaDataList[k].get("BoardFolderName"):
                    output.append(metaDataList[k])
                    metaDataList.pop(metaDataList[k])
                    k += 1
                    MaxMeta -= 1

        return output

    def filter_ChannelFolderName(self, metaDataList=None, ChannelFolderNameList=None):
        output = []
        k = 0
        MaxMeta_ = str(len(metaDataList))
        MaxMeta = int(MaxMeta_)
        tbar = tqdm(total=MaxMeta, desc="Filtering noise....")

        while k < MaxMeta:
            tbar.update(1)
            for q in range(0, len(ChannelFolderNameList)):
                if ChannelFolderNameList[q] == metaDataList[k].get("ChannelFolderName"):
                    output.append(metaDataList[k])
                    metaDataList.remove(metaDataList[k])
                    MaxMeta -= 1
                    k -= 1
            k += 1

        return output

    def filter_ChaanelCombList(self, metaDataList = None, ChannelCombList = None):
        output = []
        k = 0
        MaxMeta_ = str(len(metaDataList))
        MaxMeta = int(MaxMeta_)

        for q in range(len(ChannelCombList)):
            while k < MaxMeta:
                if ChannelCombList[q] == metaDataList[k].get("ChannelComb"):
                    output.append(metaDataList[k])
                    metaDataList.pop(metaDataList[k])
                    k += 1
                    MaxMeta -= 1

        return output

    # PackageType 안에 있는 녀석만 제외시킵니다.
    def filter_PackageType(self, metaDataList = None, PackageType = None):

        output = []
        k = 0
        MaxMeta_ = str(len(metaDataList))
        MaxMeta = int(MaxMeta_)
        tbar = tqdm(total=MaxMeta, desc="Filtering PackageType....")

        for q in range(0, len(PackageType)):
            k = 0
            while k < MaxMeta:
                tbar.update(1)
                if PackageType[q] == metaDataList[k].get("PackageTypeName"):
                    output.append(metaDataList[k])
                    metaDataList.remove(metaDataList[k])
                    MaxMeta -= 1
                    k -= 1
                k += 1

        return output

    def filter_Noise(self, metaDataList = None, NoiseThreshold = None, MetricType = None):
        output = []
        k = 0
        MaxMeta_ = str(len(metaDataList))
        MaxMeta = int(MaxMeta_)

        tbar = tqdm(total=MaxMeta, desc="Filtering noise....")
        if MetricType == "SAG":
            KeyName = "NoisePower_SAG"

            while k < MaxMeta:
                tbar.update(1)
                pkgName = metaDataList[k].get("PackageTypeName")
                threshold = self.PackageSAGNoiseAVG.get(pkgName) * NoiseThreshold

                if metaDataList[k].get(KeyName) < threshold:
                    output.append(metaDataList[k])
                    metaDataList.remove(metaDataList[k])
                    MaxMeta -= 1

                k += 1

        elif MetricType == "RMS":
            KeyName = "NoisePower_RMS"

            while k < MaxMeta:
                tbar.update(1)
                pkgName = metaDataList[k].get("PackageTypeName")
                threshold = self.PackageRMSNoiseAVG.get(pkgName) * NoiseThreshold
                self.PackageSAGNoiseAVG.get(pkgName)
                if metaDataList[k].get(KeyName) < threshold:
                    output.append(metaDataList[k])
                    metaDataList.remove(metaDataList[k])
                    MaxMeta -= 1
                k += 1

        return output

    # PackageProp을 넣으면, Package 비율이 고려됨.
    # NoisePower를 넣으면 Noise Power값 이상만 고려됨
    # DataNum 만큼 metaData에서 output을 만들어 return (filtered meta, origin meta)
    # BoardFolderNameList는 사용하고 싶은 Board들로 구성되어 있는 List.
    def metaFilter(self, metaDataList = [], PackageProp = [], DataNum = 0, NoisePower_SAG=0, NoisePower_RMS = 0,
                   BoardFolderNameList = [], PackageType = [], ChannelFolderNameList = [], CompNameList = [], ChannelCombList = []):

        if len(metaDataList) == 0:
            print("metaData가 없습니다. 확인해주세요.")
            exit(0)

        if DataNum == 0:
            print("DataNum이 0 입니다. 확인해주세요.")
            exit(0)

        if len(PackageProp) == 0:
            print("Package 정보가 없습니다. 확인해주세요.")
            exit(0)

        if not len(ChannelFolderNameList) == 0:
            filteredMeta = self.filter_ChannelFolderName(metaDataList, ChannelFolderNameList)


        # 특정 보드만 사용하고 싶을 경우
        if not len(BoardFolderNameList) == 0:
            # filteredMeta는 선택된 Board들에 대한 Meta가 저장되어 있음.
            filteredMeta = self.filter_BoardName(metaDataList, BoardFolderNameList)

        # 특정 채널에 대해서만 하고 싶을 경우
        if not len(ChannelCombList) == 0:
            filteredMeta = self.filter_ChaanelCombList(metaDataList, CompNameList)

        # 특정 Noise값 이상에 대해서만 수행하고 싶을 경우
        if not NoisePower_RMS == 0:
            filteredMeta = self.filter_Noise(metaDataList, NoisePower_RMS, "RMS")

        if not NoisePower_SAG == 0:
            filteredMeta = self.filter_Noise(metaDataList, NoisePower_SAG, "SAG")

        if not len(PackageType) == 0:
            filteredMeta = self.filter_PackageType(metaDataList, PackageType)

        # Package 비율에 따라 Data를 얻고 싶은 경우
        if self.PackageProperty_On_Off == True:
            StandardNum = DataNum / len(self.PackageNumber)

            for key, value in self.PackageNumber.items():
                eachPackageNum = self.PackagePropertyRatio[key] * value

                if eachPackageNum > StandardNum:
                    self.PackageNumber[key] = StandardNum
                    continue

                if value > StandardNum:
                    self.PackageNumber[key] = StandardNum
                elif value < StandardNum:
                    self.PackageNumber[key] = value

            outputMetaList = []

            MaxMeta = len(metaDataList)
            tbar = tqdm(total=len(self.PackageNumber), desc="PackageNum : ")
            for key, value in self.PackageNumber.items():
                tbar.update(1)

                countNum = 0

                if value == 0:
                    continue

                if key == "PackageTypeThroughHole" or key == "PackageTypePressfit":
                    continue

                maxNum = value

                random.shuffle(metaDataList)
                k = 0
                q = 0
                tbar_local = tqdm(total=maxNum, desc=str(key) + " :")
                while q < maxNum:
                    # index = random.randrange(0, MaxMeta)
                    try:
                        if metaDataList[k].get("PackageTypeName") == key:
                            tbar_local.update(1)
                            outputMetaList.append(metaDataList[k])
                            metaDataList.remove(metaDataList[k])
                            countNum += 1
                            q += 1
                        k += 1
                    except:
                        break

        else:
            outputMetaList = []
            countNum = 0
            maxNum = DataNum
            if maxNum > len(metaDataList):
                maxNum = len(metaDataList)

            random.shuffle(metaDataList)
            k = 0
            q = 0
            tbar = tqdm(total=maxNum, desc="Total : ")
            while q < maxNum:
                try:
                    if countNum < maxNum:
                        outputMetaList.append(metaDataList[k])
                        metaDataList.remove(metaDataList[k])
                        countNum += 1
                        q += 1
                        tbar.update(1)
                    k += 1
                except:
                    break

        return outputMetaList

    def PairingMeta(self,RootPath = None, metaData = None):
        if len(metaData) == 0:
            print("metaData가 없습니다. 확인해주세요.")
            exit(0)

        for k in range(len(metaData)):
            ProgressString = str(k) + " / " + str(len(metaData))
            if metaData[k].get("DataOrigin") == 0:
                for q in tqdm(range(len(metaData)), desc=ProgressString):
                    if metaData[q].get("DataOrigin") == 1 and metaData[q].get("CompName") == metaData[k].get("CompName"):
                        metaData[k]["PariedCompData_Path"] = RootPath + '/' + metaData[q]["FileName"] + ".json"
                        # print(metaData[k])
                        jstring = json.dumps(metaData[k], indent=4)
                        SaveName = RootPath + '/' + metaData[k].get("FileName") + '.json'
                        f = open(SaveName, "w")
                        f.write(jstring)
                        f.close()
                        break

    def CopyDataEach(self, PathPlace = None, DataList = []):
        if len(DataList) > 0:
            ProgressString = PathPlace
            tbar = tqdm(total=len(DataList), desc=ProgressString)
            for item in DataList:
                try:
                    tbar.update(1)
                    AnglePath_SRC = item.get("AngleMap_Path")


                    HeightPath_SRC = item.get("HeightMap_Path")
                    K3DPath_SRC = item.get("K3D_Path")
                    TopPath_SRC = item.get("TopImage_Path")
                    MidPath_SRC = item.get("MiddleImage_Path")
                    BotPath_SRC = item.get("BottomImage_Path")
                    AttentionPath_SRC = item.get("AttentionMap_Path")

                    # folders_src = AnglePath_SRC.split(os.path.sep)

                    AnglePath_SRC_Save = PathPlace + '/' + AnglePath_SRC.split(os.path.sep)[-1]
                    HeightPath_SRC_Save = PathPlace + '/' + HeightPath_SRC.split(os.path.sep)[-1]
                    K3DPath_SRC_Save = PathPlace + '/' + K3DPath_SRC.split(os.path.sep)[-1]
                    TopPath_SRC_Save = PathPlace + '/' + TopPath_SRC.split(os.path.sep)[-1]
                    MidPath_SRC_Save = PathPlace + '/' + MidPath_SRC.split(os.path.sep)[-1]
                    BotPath_SRC_Save = PathPlace + '/' + BotPath_SRC.split(os.path.sep)[-1]
                    AttentionPath_SRC_Save = PathPlace + '/' + AttentionPath_SRC.split(os.path.sep)[-1]

                    PairedMeta = self.readMetaData(metaFilePath=item.get("PariedCompData_Path"))

                    AnglePath_REF = PairedMeta.get("AngleMap_Path")
                    AttentionPath_REF = PairedMeta.get("AttentionMap_Path")
                    HeightPath_REF = PairedMeta.get("HeightMap_Path")

                    if self.AnalyticFilterMode == False:
                        HeightPath_REF = PairedMeta.get("HeightMap_Path")
                    else:
                        HeightPath_REF_new = PairedMeta.get("ImprovedHeightMap_Path")

                        if not os.path.exists(HeightPath_REF_new) == True:
                            HeightPath_REF_old = PairedMeta.get("HeightMap_Path")
                            AttentionPath_REF__ = PairedMeta.get("AttentionMap_Path")
                            # ImprovedHeight 생성.
                            ImprovedHeightMap = self.AnalyticalFilter(HeightPath_REF_old, AttentionPath_REF__)

                            np.savetxt(HeightPath_REF_new, ImprovedHeightMap.astype(np.float32), delimiter=',')

                        HeightPath_REF = PairedMeta.get("ImprovedHeightMap_Path")

                    K3DPath_REF = PairedMeta.get("K3D_Path")
                    TopPath_REF = PairedMeta.get("TopImage_Path")
                    MidPath_REF = PairedMeta.get("MiddleImage_Path")
                    BotPath_REF = PairedMeta.get("BottomImage_Path")


                    # folders_ref = AnglePath_REF.split(os.path.sep)

                    AnglePath_REF_Save = PathPlace + '/' + AnglePath_REF.split(os.path.sep)[-1]
                    HeightPath_REF_Save = PathPlace + '/' + HeightPath_REF.split(os.path.sep)[-1]
                    K3DPath_REF_Save = PathPlace + '/' + K3DPath_REF.split(os.path.sep)[-1]
                    TopPath_REF_Save = PathPlace + '/' + TopPath_REF.split(os.path.sep)[-1]
                    MidPath_REF_Save = PathPlace + '/' + MidPath_REF.split(os.path.sep)[-1]
                    BotPath_REF_Save = PathPlace + '/' + BotPath_REF.split(os.path.sep)[-1]
                    AttentionPath_REF_Save = PathPlace + '/' + AttentionPath_REF.split(os.path.sep)[-1]

                    AnglePath_SRC.replace('\\','/')
                    AnglePath_SRC_Save.replace('\\', '/')
                    K3DPath_SRC.replace('\\', '/')
                    K3DPath_SRC_Save.replace('\\', '/')
                    HeightPath_SRC.replace('\\', '/')
                    HeightPath_SRC_Save.replace('\\', '/')
                    TopPath_SRC.replace('\\', '/')
                    TopPath_SRC_Save.replace('\\', '/')
                    MidPath_SRC.replace('\\', '/')
                    MidPath_SRC.replace('\\', '/')
                    BotPath_SRC.replace('\\', '/')
                    BotPath_SRC_Save.replace('\\', '/')
                    AttentionPath_SRC.replace('\\', '/')
                    AttentionPath_SRC_Save.replace('\\', '/')

                    AnglePath_REF.replace('\\', '/')
                    AnglePath_REF_Save.replace('\\', '/')
                    K3DPath_REF.replace('\\', '/')
                    K3DPath_REF_Save.replace('\\', '/')
                    HeightPath_REF.replace('\\', '/')
                    HeightPath_REF_Save.replace('\\', '/')
                    TopPath_REF.replace('\\', '/')
                    TopPath_REF_Save.replace('\\', '/')
                    MidPath_REF.replace('\\', '/')
                    MidPath_REF.replace('\\', '/')
                    BotPath_REF.replace('\\', '/')
                    BotPath_REF_Save.replace('\\', '/')
                    AttentionPath_REF.replace('\\', '/')
                    AttentionPath_REF_Save.replace('\\', '/')

                    if not os.path.exists(os.path.dirname(AnglePath_SRC_Save)):
                        os.makedirs(os.path.dirname(AnglePath_SRC_Save))

                    if not os.path.exists(os.path.dirname(AnglePath_REF_Save)):
                        os.makedirs(os.path.dirname(AnglePath_REF_Save))

                    shutil.copy(AnglePath_SRC, AnglePath_SRC_Save)
                    shutil.copy(HeightPath_SRC, HeightPath_SRC_Save)
                    shutil.copy(K3DPath_SRC, K3DPath_SRC_Save)
                    shutil.copy(TopPath_SRC, TopPath_SRC_Save)
                    shutil.copy(MidPath_SRC, MidPath_SRC_Save)
                    shutil.copy(BotPath_SRC, BotPath_SRC_Save)
                    shutil.copy(AttentionPath_SRC, AttentionPath_SRC_Save)

                    shutil.copy(AnglePath_REF, AnglePath_REF_Save)
                    shutil.copy(HeightPath_REF, HeightPath_REF_Save)
                    shutil.copy(K3DPath_REF, K3DPath_REF_Save)
                    shutil.copy(TopPath_REF, TopPath_REF_Save)
                    shutil.copy(MidPath_REF, MidPath_REF_Save)
                    shutil.copy(BotPath_REF, BotPath_REF_Save)
                    shutil.copy(AttentionPath_REF, AttentionPath_REF_Save)
                except:
                    print("Copy error")
                    continue

    def CopyData(self, SavePath, TrainData = [], ValidData = [], TestData = []):

        TrainPath = SavePath + '/' + "trainData"
        ValidPath = SavePath + '/' + "validData"
        TestPath = SavePath + '/' + "testData"

        if not os.path.exists(TrainPath):
            os.makedirs(TrainPath)

        if not os.path.exists(TestPath):
            os.makedirs(TestPath)

        if not os.path.exists(ValidPath):
            os.makedirs(ValidPath)

        if len(TrainData) > 0:
            self.CopyDataEach(TrainPath, TrainData)
        if len(ValidData) > 0:
            self.CopyDataEach(ValidPath, ValidData)
        if len(TestData) > 0:
            self.CopyDataEach(TestPath, TestData)

    def getPackageDistributionFromMeta(self, metaDataList):
        count = 0
        temp = {}
        tbar = tqdm(total=len(metaDataList), desc="getPackageDistribution")
        while count < len(metaDataList):
            tbar.update(1)
            for key, value in self.PackagePropertyRatio.items():
                if metaDataList[count].get("PackageTypeName") == key:
                    try:
                        temp[key] += 1
                        break
                    except:
                        temp[key] = 1
                        break
            count += 1

        for key, value in temp.items():
            self.PackagePropertyRatio[key] = float(value/len(metaDataList))
        print("finish getPackageDistribution")
        print(self.PackagePropertyRatio)

        return temp




    def setPackageProperty(self, Average = True, PackagesNumber = {}):
        self.PackageNumber = copy.deepcopy(PackagesNumber)
        total = 0
        for key, value in self.PackagePropertyRatio.items():
            if value > 1:
                print("Package 비율 값은 1을 넘을 수 없습니다.")
                exit(0)

            total += value

        if total == 0 and Average == True:
            for Key, Value in self.PackagePropertyRatio.items():
                self.PackagePropertyRatio[Key] = 1/len(self.PackagePropertyRatio)

            self.PackageProperty_On_Off = True
            return True

        elif total > 0 and Average == True:
            print("Package 비율 값이 Setting 되어 있지만, Average로 강제 Setting 합니다.")
            self.PackageProperty_On_Off = True
        elif total == 0 and Average == False:
            print("Package 비율을 적용하지 않습니다.")
            return True

        # Average 모드로 강제 setting.
        if Average == True:
            for Key, Value in self.PackagePropertyRatio.items():
                self.PackagePropertyRatio[Key] = 1 / len(self.PackagePropertyRatio)

            self.PackageProperty_On_Off = True
            return True

    def AnalyticalFilter(self, OriginHeight_Path=None, AttentionMap_Path_=None):

        Attentionmap = cv2.imread(AttentionMap_Path_, 0)

        Heightmap = np.genfromtxt(OriginHeight_Path, delimiter=",")
        Heightmap = np.clip(Heightmap, 0, 8000)

        Heightmap_uint8 = Heightmap.copy()
        cv2.normalize(Heightmap, Heightmap_uint8, 0, 255, norm_type=cv2.NORM_MINMAX)

        Heightmap_uint8 = Heightmap_uint8.astype(np.uint8)
        ret, Attentionmap_manual = cv2.threshold(Heightmap_uint8, 128, 255, cv2.THRESH_OTSU)

        cv2.normalize(Attentionmap, Attentionmap, 0, 255, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(Attentionmap_manual, Attentionmap_manual, 0, 255, norm_type=cv2.NORM_MINMAX)

        Attentionmap = Attentionmap.astype(np.uint8)
        Attentionmap_manual = Attentionmap_manual.astype(np.uint8)

        ratioX = Heightmap.shape[1] / float(Attentionmap.shape[1])
        ratioY = Heightmap.shape[0] / float(Attentionmap.shape[0])

        ResizedAttention = cv2.resize(Attentionmap, None, fx=ratioX, fy=ratioY)

        Attentionmap = np.maximum(Attentionmap_manual, ResizedAttention)
        Attentionmap_Inv = 255 - Attentionmap

        Distancemap = cv2.distanceTransform(Attentionmap_Inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        cv2.normalize(Distancemap, Distancemap, 0, 1, norm_type=cv2.NORM_MINMAX)

        Distancemap_ = np.max(Distancemap) - Distancemap
        Distancemap_ = Distancemap_ * Distancemap_ * Distancemap_ * Distancemap_ * Distancemap_

        Heightmap = Heightmap.astype(np.float32)
        Heightmap = cv2.bilateralFilter(Heightmap, 21, 75, 75)
        AttentionFiltered = Heightmap * Distancemap_

        where_are_Nanas = np.isnan(AttentionFiltered)
        AttentionFiltered[where_are_Nanas] = 0

        kernel1 = np.ones((3, 3), np.float32)
        kernel2 = np.ones((3, 3), np.float32)

        if (AttentionFiltered.shape[0] * AttentionFiltered.shape[1]) > 256 * 256:
            kernel1 = np.ones((5, 5), np.float32)
            kernel2 = np.ones((5, 5), np.float32)
        elif (AttentionFiltered.shape[0] * AttentionFiltered.shape[1]) > 512 * 512:
            kernel1 = np.ones((7, 7), np.float32)
            kernel2 = np.ones((7, 7), np.float32)

        AttentionFiltered = cv2.morphologyEx(AttentionFiltered, cv2.MORPH_OPEN, kernel1)
        AttentionFiltered = cv2.morphologyEx(AttentionFiltered, cv2.MORPH_CLOSE, kernel2)

        # cv2.normalize(AttentionFiltered, AttentionFiltered, 0, 255, norm_type=cv2.NORM_MINMAX)
        # AttentionFiltered = AttentionFiltered.astype(np.uint8)
        # cv2.imshow("AttentionFiltered", AttentionFiltered)
        # cv2.waitKey(0)

        return AttentionFiltered













