# -*- coding: utf-8 -*-
def getStringFromNumber(self,size,value):
        """
        转为十六进制（Hex）字符串
        :param size:
        :param value:
        :return:
        """
        size=int(size)
        value=int(value)
        by = bytearray([])
        for i in range(1,size+1):
            val = value >> 8 * (size - i) & 255
            by.append(val)
        val = by.hex()
        print("===============================")
        print("%s转为%s个字节十六进制（Hex）字符串:%s"%(value,size,val))
        print("===============================")
        return val
getStringFromNumber(5,11)
