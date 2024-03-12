## assignment1

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/

思路与感想：从第四个数（T3）开始，之后的数都是前三个数相加，将最后三个数字相加得到的数放在下一位，依次类推出Tn及之前的数，最后输出Tn，这种相加需进行（n-2）次，即想得到第四个数（n=3）只需要加1（3-2）次。这道题让我记起了考场上对不齐角标的恐慌……尤其学了R之后，总要迟疑一下第n个角标是n还是n-1。

##### 代码

```python
# 
n=int(input())
T=[0,1,1]
if n>=3:
    for i in range(n-2):
        T.append(T[i]+T[i+1]+T[i+2])
print(T[n])
```

代码运行截图 

![image-20240226200908564](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240226200908564.png)

### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A

思路与感想：本题曾在去年计概时AC过，当时采用了答案的解法，这次用作复习。总体思路是，保证某个字母出现之前，前边的字母都按次序出现过并标记，如果次序不按单词顺序就要去掉所有标记重新检索标记。

##### 代码

```python
#
w=input()
h=0
e=0
l1=0
l2=0
o=0
for i in range(len(w)):
    if w[i]=='h':
        h=1
    elif w[i]=='e' and h==1:
        e=1
        h=0
    elif w[i]=='l' and e==1:
        l1=1
        e=0
    elif w[i]=='l' and l1==1:
        l2=1
        l1=0
    elif w[i]=='o' and l2==1:
        o=1
        
if o==1:
    print("YES")
else:
    print("NO")

```

代码运行截图 

![image-20240226203537019](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240226203537019.png)

### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A

思路与感想：也是曾经AC过的题，先将所有字母都转换为小写，再筛出非元音字母，输出时可以用循环，也可以用join()将一整个列表都输出。

例如去年使用了：

```python
print('.',end="")
print('.'.join(wordout))
```

##### 代码

```python
# 
word=list(input().lower())
string=["a",'o','u','e','y','i']
wordout=[]
for i in range (len(word)):
    if word[i] not in string:
        wordout.append(word[i])
for i in range(len(wordout)):
    print('.',end="")
    print(wordout[i],end="")
```

代码运行截图

![image-20240226204334909](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240226204334909.png)

### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/

思路与感想：找到n之前的所有素数（余数不为零），定住素数A之后判断n-A是否也是素数，如果是，则n-A就是素数B，输出A和B即可。

##### 代码

```python
# 
n=int(input())
prime=[]

for i in range(2,n-1):
    if (n%i)!=0:
        prime.append(i)
    else:
        continue

for pri in prime:
    if (n-pri) in prime:
        print(pri,n-pri)
    break
```

代码运行截图

![image-20240226222104508](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240226222104508.png)

### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/

思路与感想：字符串形如“ax^b+ax^b+……ax^b”将字符串分割两次得到a和b，补全1n，去掉0n，选出其中最大的b，输出“n^”和b即可。这道题总觉得似曾相识，好像是在某次月考中做过，但是并没有找到记录，很奇怪，熟悉的陌生题。

##### 代码

```python
# 
n=(input().split("+"))
Omax=0
for i in range(len(n)):
    n[i]=n[i].split("n^")
    
    if n[i][0]=="":
        n[i][0]="1"
    elif n[i][0]=="0":
        n[i][1]="0"
    
    n[i][1]=int(n[i][1])
    if n[i][1]>=Omax:
        Omax=n[i][1]
print("n^",Omax,sep="")
```

代码运行截图 

![image-20240226231626891](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240226231626891.png)

### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/

思路与感想：使用字典计数，找到票数最多的序号，排序输出序号。由于当年没好好学字典，遇到题都是硬套列表，所以完全没有字典的语法基础，还在考场上因为角标对不齐吃了大亏，今天就在字典语法上花了些时间，最后的join()直接“搬运”了我以前的代码——字符串转换为int再转回来——将字符串转换为int，这样在排序时就不会出现问题；join()只能用于字符串，所以排序后一定要再转换回字符串。

##### 代码

```python
# 
n=list(input().split())
ndict={}
for i in range(len(n)):    
    if ndict.get(n[i]) is not None:
        x=ndict[n[i]]
        x+=1
        ndict.update({n[i]:x})
    else:
        ndict[n[i]]=1
maxvalue=max(ndict.values())
maxlist=[]
for i,j in ndict.items():
    if j==maxvalue:
        maxlist.append(int(str(i)))   
print(" ".join('%s' %id for id in sorted(maxlist)))
```

代码运行截图 

![image-20240226215929793](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240226215929793.png)

### 感想：

在计概中拿到了较为理想的成绩后，就根据专业需要将学习重点转向了R语言，所以在做这次作业之前已经有整整一年没有使用过python了。这次作业中虽然有几道题是曾经做过的，但时隔一年再次打开oj等网站已经“手生”，甚至python的基础语法也混淆了，很多函数已经不记得用法了，所以这次作业在查询语法上花了较多时间。之后大概会给自己定一个每天练习的小目标，争取不做ddl战士。第一周的作业完成较快，代码基本能一次AC（吃的唯一一个WA是因为忘记删掉测试的那一行输出），思路也很顺。但因为语法生疏，用了差不多三个小时，从机考的角度来看还是得多多练习呀。