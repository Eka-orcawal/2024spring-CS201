assignment3

## **Week3**



Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by ==靳咏歌 城市与环境学院==

**说明：**

The complete process to learn DSA from scratch can be broken into 4 parts:

- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

**编程环境**

操作系统：Windows 11

Python编程环境：Spyder IDE 5.4.3（conda）, Python 3.11.5 64-bit | Qt 5.15.2 | PyQt5 5.15.7

### **02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/

思路：dp，要找到每个导弹后最多能有几个一连串比它高度低的，所以每个dp[i]要选择在它之后最大的dp[j]+1。

##### 代码

```python
# 
k=int(input())
x=list(map(int,input().split()))
dp=[0]*k
for i in range(k-1,-1,-1):
    maxn=1
    for j in range(k-1,i,-1):
        if x[i]>=x[j]:
            maxn=max(maxn,dp[j]+1)
    dp[i]=maxn    
print(max(dp))
```

代码运行截图 

![image-20240307171452325](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240307171452325.png)

### **04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147

思路：直接去看了题下面给的参考，这是一道阅读理解题（确信）。会套递归的逻辑就很容易

##### 代码

```python
# 将编号为numdisk的盘子从init杆移至desti杆
def moveOne(numDisk : int, init : str, desti : str):
    print("{}:{}->{}".format(numDisk, init, desti))

#将numDisks个盘子从init杆借助temp杆移至desti杆
def move(numDisks : int, init : str, temp : str, desti : str):
    if numDisks == 1:
        moveOne(1, init, desti)
    else: 
        # 首先将上面的（numDisk-1）个盘子从init杆借助desti杆移至temp杆
        move(numDisks-1, init, desti, temp) 
        
        # 然后将编号为numDisks的盘子从init杆移至desti杆
        moveOne(numDisks, init, desti)
        
        # 最后将上面的（numDisks-1）个盘子从temp杆借助init杆移至desti杆 
        move(numDisks-1, temp, init, desti)

n, a, b, c = input().split()
move(int(n), a, b, c)
```

代码运行截图

![image-20240307170123069](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240307170123069.png)

### **03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253

思路：按顺序弹出到一个输出列表里就行，因为看错了题意浪费了很多时间……

##### 代码

```python
# 
while True:
    n,p,m=map(int,input().split())
    if n==0:
        break
    children_out=[]
    num=list(range(1,n+1))
    for i in range(p-1):
        tmp=num.pop(0)
        num.append(tmp)
    count=0
    while num:
        tmp=num.pop(0)
        count+=1
        if count==m:
            count=0
            children_out.append(tmp)
            continue
        num.append(tmp)       

    print(",".join(map(str,children_out)))
```

代码运行截图 

![image-20240307164801705](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240307164801705.png)

### **21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554

思路：让用时短的人先做实验就能使时间最短

##### 代码

```python
# 
n=int(input())
T=list(map(int,input().split()))
sum_time=0
Time=[]
timelist=[]
for i in range(n):      
    timelist.append([T[i],i+1])
timelist=sorted(timelist)
waitinglist=[x[1] for x in timelist]
for i in range(n):
    sum_time+=timelist[i][0]
    Time.append(sum_time) 
sum_waiting=sum(Time)-sum(T)

print(" ".join("%s" %id for id in waitinglist))
print("%.2f"%(sum_waiting/n))
```

代码运行截图 

![image-20240310214943750](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240310214943750.png)

### **19963:买学区房**

http://cs101.openjudge.cn/practice/19963

思路：进行两次排序比较大小即可，比较麻烦的是数据的接收

##### 代码

```python
# 
n=int(input())
xy=[tuple(map(int, pair.strip("()").split(","))) for pair in input().split()]
prices=list(map(int,input().split()))
houseratio=[[],[]]
ratios=[]
for i in range(n):
    ratio=(xy[i][0]+xy[i][1])/prices[i]
    ratios.append(ratio)
    houseratio[0].append(ratio)
    houseratio[1].append(prices[i])
midratio=[]
for i in range(2):
    sqeue=sorted(houseratio[i])
    if n%2==1:
        mid=(sqeue[n//2])
    else:
        mid_right=n//2
        mid_left=mid_right-1
        mid=(sqeue[mid_left]+sqeue[mid_right])/2
    midratio.append(mid)
output=0
for i in range(n):
    if houseratio[0][i]>midratio[0] and houseratio[1][i]<midratio[1] :
        output+=1
print(output)
```

代码运行截图

![image-20240311112908831](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240311112908831.png)

### **27300: 模型整理**

http://cs101.openjudge.cn/practice/27300

思路：这题主要麻烦在语法上，借鉴了答案里的处理方法，在每一个参数量的后边缀一个数字，用来作比较。

##### 代码

```python
# 
n=int(input())
modeldict={}
for i in range(n):
    modelname,modelquantity=(input().split("-"))
    if modelquantity[-1]=="M":							#用元组添加参数量的后缀
        modelquantity=(modelquantity,float(modelquantity[:-1])/1000)
    else:
        modelquantity=(modelquantity,float(modelquantity[:-1]))
    if modelname in modeldict:
        modeldict[modelname].append(modelquantity)
    else:
        modeldict[modelname]=[modelquantity]
    
names=sorted(modeldict)
for name in names:
    modeldict[name]=sorted(modeldict[name],key=lambda x:x[1])#排序依据是参数量的后缀
    print(name,": ", ', '.join([x[0]for x in modeldict[name]]),sep="")

```

代码运行截图 

![image-20240310224133660](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240310224133660.png)

### 感想

由于上机时间和专业课撞了，所以并没有去机房月考。耗时比前两周的长了不少，每题花费的时间30min到2h不等（如果是上机限时做最多AC3……）。有两道题耗时太久，所以看了题解。这六道题的思路基本没问题，但是语法太薄弱了，导致耗时过长（大部分时间都在搜语法）。还有就是，GPT用来学语法真的很方便（虽然偶尔它会胡说八道）。

