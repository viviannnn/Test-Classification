# -- coding: utf-8 --
import codecs
import math
import sys
import os

#input_file = "SogouC/Segment/C000020_pre.txt"
#ClassCode = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000023', 'C000024']
classCodes = ['C000013','C000024']
#path after cutting
file_path = sys.path[0] + "/../Data"

# Stop words
def isStopWord(word):
    with open ('stopwords.txt', 'r') as f:
        words = f.readlines()
        if word in words:
            return True
        else:
            return False

# 卡方计算公式 CHI的缺点就是夸大了低频次的作用，因为A,B,C,D只标记词在某一文档中出现与否，不标记词在某一个文档中出现的频率，因此CHI适合用来进行特征词的提取，即适合进行文档的分类特征词语的提取
def Chi(a, b, c, d):
    return float(pow((a * d - b * c), 2)) /float((a + c) * (a + b) * (b + d) * (c + d))
# 对卡方检验所需的 a b c d 进行计算
# a：在这个分类下包含这个词的文档数量
# b：不在该分类下包含这个词的文档数量
# c：在这个分类下不包含这个词的文档数量
# d：不在该分类下，且不包含这个词的文档数量

#classDic... map the classcode with the class list 
#each class list item maps a doc
#each doc maps a set putting the words
def buildSets():
    classDocDic = dict()
    classWordDic = dict()
    for classCode in classCodes:
        path = file_path + '/' + classCode + '_train.txt'
        classDocList = list()
        classWordSet = set()
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines: #按行读文件（每行为一个文档）
                docWordSet = set()
                doc = line.strip('\n').strip().split(" ")#doc-->[, , ,]
                for word in doc:
                    stripword = word.strip().strip('\n')
                    classWordSet.add(stripword) #全部无重复单词
                    docWordSet.add(stripword) #该行/文档无重复单词

                classDocList.append(docWordSet) #classDocList[(文档), (文档)]
        classDocDic[classCode] = classDocList #classDocDic code -> classDocList[set(文档), set(文档)]
        classWordDic[classCode] = classWordSet #classWordDic code -> classWordSet(该文档所有单词)
    return classDocDic, classWordDic


    

# 对得到的两个词典进行计算，可以得到a b c d 值
# K 为每个类别选取的特征个数
    

def featureSelection(classDocDic, classWordDic, K):
    wordCountDic = dict()
    #the dict of the chi value for each word
    for key in classWordDic: # 对某个类别下的每一个单词的 a b c d 进行计算
        classWordSets = classWordDic[key] #classWordSet(该文档所有单词)
        classWordCountDic = dict()
        for eachword in classWordSets:#取某个单词
            #print eachword
            a = 0
            b = 0
            c = 0
            d = 0
            for eachclass in classDocDic:#classDocDic code -> classDocList[set(文档), set(文档)]
                if eachclass == key: #if同类
                    #a, c
                    for eachdoc in classDocDic[eachclass]: #eachdoc [set(文档), set(文档)....]
                        if eachword in eachdoc:
                            a += 1
                        else:
                            c += 1
                else:
                    #b, d
                    for eachdoc in classDocDic[eachclass]:
                        if eachword in eachdoc:
                            b += 1
                        else:
                            d += 1
            #print (str(a) + " "+str(c)+" "+str(b)+" "+str(d))
            #print("a+c:"+str(a+c)+"b+d"+str(b+d))
            classWordCountDic[eachword] = Chi(a, b, c, d)
        sortedClassWordCountDic = sorted(classWordCountDic.items(), key = lambda d:d[1], reverse = True)
        #print sortedClassWordCountDic
        tmp = dict()
        for i in range(K): #取前k个卡方值最大的词为特征词
            tmp[sortedClassWordCountDic[i][0]] = sortedClassWordCountDic[i][1]
        wordCountDic[key] = tmp
    return wordCountDic
        # print(sortedClassTermCountDic)

# 调用buildItemSets
# buildItemSets形参表示每个类别的文档数目,在这里训练模型时每个类别取前200个文件
classDocDic, classWordDic = buildSets()
wordCountDic = featureSelection(classDocDic, classWordDic, 100)

results = set()
for eachclass in wordCountDic:
    for word in wordCountDic[eachclass]:
        results.add(word)
with open("svm/model/SVMFeature.txt", 'w') as f:
    count = 1
    for result in results:
        final_result = result.strip('\n').strip(' ')
        if len(final_result) > 0 and result != " ":
            f.write(str(count) + ' ' + final_result + '\n')
            count = count + 1















