import  os
import  sys

import cv2
import numpy as np
import struct
import shutil


# typedef  struct  tagBITMAPFILEHEADER
# {
# unsigned short int  bfType;       //位图文件的类型，必须为BM
# unsigned int       bfSize;       //文件大小，以字节为单位
# unsigned short int  bfReserverd1; //位图文件保留字，必须为0
# unsigned short int  bfReserverd2; //位图文件保留字，必须为0
# unsigned int       bfbfOffBits;  //位图文件头到数据的偏移量，以字节为单位
# }BITMAPFILEHEADER;

class BITMAPFILEHEADER:
    def __init__(self,bfType,bfSize,bfbfOffBits,bfReserverd=0):
        self.bfType = bfType
        self.bfSize = bfSize
        self.bfbfOffBits = bfbfOffBits
        self.bfReserverd = bfReserverd

    def pack(self):
        #<: 表示使用小端字节序（Little-endian），即低位字节存储在内存的低地址处。
        # H: 无符号短整型（unsigned short int），大小为 2 个字节。
        # I: 无符号整型（unsigned int），大小为 4 个字节。
        # B: 无符号字符（unsigned char），大小为 1 个字节。
        return struct.pack('<BBIII',self.bfType[0],self.bfType[1],self.bfSize,self.bfReserverd,self.bfbfOffBits)

    @classmethod
    def unpack(cls, data):
        bfType1,bfType2,bfSize,bfReserverd,bfbfOffBits = struct.unpack('<BBIII',data)

        return cls([bfType1,bfType2],bfSize,bfbfOffBits,bfReserverd )

    def printInfo(self):
        print("####################################")
        print("file header:")
        print(f"bfType={chr(self.bfType[0])}{chr(self.bfType[1])}")
        print("bfSize={}".format(self.bfSize))
        print("bfReserverd={}".format(self.bfReserverd))
        print("bfbfOffBits={}".format(self.bfbfOffBits))



##typedef  struct  tagBITMAPINFOHEADER
# {
# int biSize;                       //该结构大小，字节为单位
# int  biWidth;                     //图形宽度以象素为单位
# int  biHeight;                    //图形高度以象素为单位
# short int  biPlanes;              //目标设备的级别，必须为1
# short int  biBitcount;            //颜色深度，每个象素所需要的位数
# biBitcount 还有1位（单色），2位（4色，CGA），4位（16色，VGA），8位（256色），16位（增强色），24位（真彩色）和32位等
# short int  biCompression;         //位图的压缩类型
# int  biSizeImage;                 //位图的大小，以字节为单位
# int  biXPelsPermeter;             //位图水平分辨率，每米像素数
# int  biYPelsPermeter;             //位图垂直分辨率，每米像素数
# int  biClrUsed;                   //位图实际使用的颜色表中的颜色数
# int  biClrImportant;              //位图显示过程中重要的颜色数
# }BITMAPINFOHEADER;

class BITMAPINFOHEADER():
    def __init__(self,biSize,biWidth,biHeight,biPlanes,biBitcount,biCompression,
                 biSizeImage,biXPelsPermeter,biYPelsPermeter,biClrUsed,biClrImportant):

        self.biSize = biSize
        self.biWidth = biWidth
        self.biHeight = biHeight
        self.biPlanes = biPlanes
        self.biBitcount = biBitcount
        self.biCompression = biCompression
        self.biSizeImage = biSizeImage
        self.biXPelsPermeter = biXPelsPermeter
        self.biYPelsPermeter = biYPelsPermeter
        self.biClrUsed = biClrUsed
        self.biClrImportant = biClrImportant

    def pack(self):
        return struct.pack('IIIHHIIIIII',self.biSize,self.biWidth,self.biHeight,
                           self.biPlanes,self.biBitcount,
                           self.biCompression,self.biSizeImage,
                           self.biXPelsPermeter,self.biYPelsPermeter,
                           self.biClrUsed,self.biClrImportant)


    @classmethod
    def unpack(cls, data):
        biSize, biWidth, biHeight, biPlanes, biBitcount, biCompression, \
        biSizeImage, biXPelsPermeter, biYPelsPermeter, biClrUsed, biClrImportant = \
        struct.unpack('IIIHHIIIIII', data)

        return cls(biSize, biWidth, biHeight, biPlanes, biBitcount, biCompression,
                   biSizeImage, biXPelsPermeter, biYPelsPermeter, biClrUsed, biClrImportant)

    def printInfo(self):
        print("####################################")
        print("bit map file header:")
        print(f"biSize: {self.biSize}")
        print(f"biWidth: {self.biWidth}")
        print(f"biHeight: {self.biHeight}")
        print(f"biPlanes: {self.biPlanes}")
        print(f"biBitcount: {self.biBitcount}")
        print(f"biCompression: {self.biCompression}")
        print(f"biSizeImage: {self.biSizeImage}")
        print(f"biXPelsPermeter: {self.biXPelsPermeter}")
        print(f"biYPelsPermeter: {self.biYPelsPermeter}")
        print(f"biClrUsed: {self.biClrUsed}")
        print(f"biClrImportant: {self.biClrImportant}")
class BMP():
    def __init__(self):
        self.fileBuff=None
        self.fileSize=None
        self.header = None
        self.bitmapHeader=None
        self.colourTable =None
        self.imageData = None
        pass

    def printInfo(self):
        # 获取 self.fileBuff 的字节长度
        print(f"fileSize={self.fileSize}")
        self.header.printInfo()
        self.bitmapHeader.printInfo()


    def readFile(self,filePath):
        with open(filePath, 'rb') as file:
            self.fileBuff = file.read()
            self.fileSize = os.path.getsize(filePath)

    def parse(self):
        if self.fileBuff:
            self.parseHeader()
            self.parseBitMapHeader()
            self.parseColourTable()
            #54字节到1078字节是调色板，真彩的不需要调色板（24位，32位的情况）

            #1078字节后是图像数据
            self.parseImageData()

    def parseHeader(self):
        self.header = BITMAPFILEHEADER.unpack(self.fileBuff[:14])

    def parseBitMapHeader(self):

        self.bitmapHeader = BITMAPINFOHEADER.unpack(self.fileBuff[14:54])

    def parseColourTable(self):
        if self.bitmapHeader.biBitcount <= 8 or self.header.bfbfOffBits > 54:
            self.colourTable = self.fileBuff[54:self.header.bfbfOffBits]
            self.colourTable = np.frombuffer(self.colourTable,np.uint32)
            self.colourTableList = []
            for color in self.colourTable:
                # 提取每个颜色的四个分量
                b = (color & 0xFF000000) >> 24  # 蓝色分量
                g = (color & 0x00FF0000) >> 16  # 绿色分量
                r = (color & 0x0000FF00) >> 8  # 红色分量
                a = (color & 0x000000FF)  # Alpha 通道

                # 将分量添加到列表中
                self.colourTableList.append([r, g, b, a])

    def parseImageData(self):
        if self.header:
            self.imageData = np.frombuffer(self.fileBuff[self.header.bfbfOffBits:],dtype=np.uint8).reshape(self.bitmapHeader.biHeight,self.bitmapHeader.biWidth)

    def showImage(self):
        cv2.imshow("bmp",self.imageData)
        cv2.waitKey()

    @staticmethod
    def genColourTable(biBitcount,colourSize,elementByteSize):
        palette = []
        if biBitcount == 8:
            for i in range(0,colourSize):
                r = g = b = i  # Grayscale palette, R = G = B = index value
                a = i  # Alpha channel set to 0 for full transparency
                palette.extend([b, g, r, a])

        palette = np.array(palette).astype(np.uint8)

        return palette

    def saveFile(self,bmp_output_path):
        # colourTable = None
        #
        # # if self.bitmapHeader.biBitcount <= 8:
        # #     colourSize = 2 ** self.bitmapHeader.biBitcount
        # #     elementByteSize = 4
        # #     colourTable= self.genColourTable(self.bitmapHeader.biBitcount, colourSize, elementByteSize)
        # #     self.header.bfbfOffBits = 54 + colourSize * elementByteSize

        with open(bmp_output_path, 'wb') as f:
            f.write(self.header.pack())
            f.write(self.bitmapHeader.pack())
            if self.colourTable is not None :
                f.write(np.array(self.colourTable).tobytes())
            f.write(self.imageData.tobytes())

    @classmethod
    def array_to_bmp(cls,image_array_hwc_bgr_or_hw_gray,rgb2bgr=False):
        image_array = image_array_hwc_bgr_or_hw_gray
        if not isinstance(image_array, np.ndarray):
            raise ValueError("The image_array must be a NumPy array.")

        if image_array.dtype != np.uint8:
            raise ValueError("The image_array must be of type np.uint8.")

        width  = None
        height = None
        channels = None
        biBitcount = None




        if len(image_array.shape) == 2:
            height, width = image_array.shape
            channels = 1  # Grayscale image
        elif len(image_array.shape) == 3:
            height, width,channels = image_array.shape
            if channels == 1:
                image_array.reshape((height,width))

        else:
            raise ValueError("Unsupported image_array shape: {}".format(image_array.shape))

        bmp = cls()
        ColourTable = None

        if channels == 1:
            image_array = np.flipud(image_array)
            biBitcount = 8 * channels
            colourSize = 2 ** biBitcount
            elementByteSize = 4
            ColourTable = bmp.genColourTable(biBitcount,colourSize,elementByteSize)
            bmp.colourTable = ColourTable
        elif channels == 3:
            image_array = np.flip(image_array, axis=0)
            biBitcount = 8 * channels
            ColourTable = None
        elif channels == 4:
            image_array = np.flip(image_array, axis=2)
            biBitcount = 8 * channels
            ColourTable = None
        else:
            raise ValueError(f"Unsupported channels: {channels}")

        bmp.imageData = image_array

        fileSize = 54
        bfbfOffBits = 54

        if ColourTable is not None and ColourTable.size > 0:
            ColourTableByteSize = ColourTable.size * ColourTable.itemsize
            fileSize    +=  ColourTableByteSize
            bfbfOffBits +=  ColourTableByteSize
        biSizeImage = bmp.imageData.size * bmp.imageData.itemsize
        fileSize += biSizeImage
        biClrUsed = 2 ** biBitcount
        biClrImportant = 2 ** biBitcount
        bmp.header = BITMAPFILEHEADER([ord('B'), ord('M')], fileSize,bfbfOffBits)
        bmp.bitmapHeader = BITMAPINFOHEADER(40,width,height,1,biBitcount,0,
                                            biSizeImage,0,0,biClrUsed,biClrImportant)

        return bmp

def test():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_directory)

    bmp_dir_path = current_directory

    bmp_file            = os.path.join(root_dir,"./data/gray/strawberries_coffee_1x256x256.bmp")
    raw_gray_u8_file    = os.path.join(root_dir,"./data/gray/strawberries_coffee_1x256x256_u8.raw")
    raw_rgb_u8_file     = os.path.join(root_dir,"./data/rgb/strawberries_coffee_3x256x256_u8.raw")

    bmp  =  BMP()
    bmp.readFile(bmp_file)
    bmp.parse()
    bmp.printInfo()
    #bmp.showImage()

    bmp.saveFile(os.path.join(bmp_dir_path,"test.bmp"))

    with open(raw_gray_u8_file, 'rb') as f:
        # 读取文件内容并按照u32类型reshape
        image_data = np.fromfile(f, dtype=np.uint8)

        # 获取数据的shape，这里假设数据是宽高均为256x256
        height = 256
        width = 256
        image_data_arry = image_data.reshape((height, width))
        xx = bmp.array_to_bmp(image_data_arry)
        xx.saveFile(os.path.join(bmp_dir_path,"gray.bmp"))

    with open(raw_rgb_u8_file, 'rb') as f:
        # 读取文件内容并按照u32类型reshape
        image_data = np.fromfile(f, dtype=np.uint8)

        # 获取数据的shape，这里假设数据是宽高均为256x256
        height = 256
        width = 256
        image_data_arry = image_data.reshape((height, width,3))
        xx = bmp.array_to_bmp(image_data_arry,True)
        xx.saveFile(os.path.join(bmp_dir_path,"rgb.bmp"))

if __name__ == "__main__":
    test()