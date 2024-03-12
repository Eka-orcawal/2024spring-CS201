## Week2

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/

思路：先按照分数的运算法则算出未化简的分子分母，再寻找分子分母的最大公因数约分。如果直接用for循环一个一个找会Runtime Error，所以使用了辗转相除法：用较大数m除较小数n，若能整除则最大公约数就是较小数。若有余数r，则用较小数除余数，直到整除。最后一步的较小数就是所求的最大公因数。

这道题第一次尝试超时，毫不意外。

##### 代码

```python
# 
a1,b1,a2,b2=map(int,input().split())
a=a1*b2+a2*b1
b=b1*b2
if a==b:
    print("1")
else:
    m=max(a,b)
    n=min(a,b)
    r=m%n
    while r!=0:#辗转相除法求最大公因数
        m=n
        n=r
        r=m%n

    print(a//n,"/",b//n,sep="")
```

代码运行截图 

![image-20240229120712827](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240229120712827.png)

### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110

思路：由于可以随意拆分，所以不存在装不满的情况，找出单价最高的糖果优先装箱即可

##### 代码

```python
# 
n,w=map(int,input().split())
danjia=[]

for i in range(n):
    zongjia,xiangshu=map(int,input().split())
    for j in range(xiangshu):
        danjia.append(zongjia/xiangshu)
danjia.sort()
danjia.reverse()#最大单价在前
print('%.1f'%sum(danjia[:w]))
```

代码运行截图

![image-20240229121842022](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240229121842022.png)

### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/

思路：先将所有输入的技能用字典储存起来，然后按照时间顺序优先输出伤害最高的技能。这个题以前也做过，按照以前的思路过了一遍，字典确实很好用。

##### 代码

```python
#
nCases=int(input())
for Casse in range(nCases):
    n,m,b=map(int,input().split())
    ns={}
    for i in range(n):#将每刻对应的所有技能伤害存在dict中
        ti,xi=map(int,input().split())
        if ti in ns:
            ns[ti].append(xi)
        else:
            ns[ti]=[xi]
    times=sorted(ns)#有技能的时刻
    ending=0
    for time in times:
        if m>=len(ns[time]):#可释放数量>=技能数量时不用排序
            b-=sum(ns[time])
        else:
            ns[time].sort()
            ns[time].reverse()
            b-=sum(ns[time][0:m])#优先输出最高伤害的技能
        if b<=0:
            print(time)
            break
    if b>0:
        print("alive")
```

代码运行截图

![image-20240229124723328](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240229124723328.png)

### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B

思路：素数的平方就是T-prime，所以先是创建列表来判断是否为质数，最后注意1的特殊情况就可以。判断质数的方法是从以前的AC题解找来的，欧拉筛还是那么的“不明觉厉”，在纸上演算了几遍搞懂了机制。

##### 代码

```python
# 欧拉筛，找到prime
n=int(input())
x=list(map(int,input().split()))
pri=[True]*1000001
for i in range (1,500001):
    if pri[i]:
        for j in range(i,1000001,i+1):
            pri[j]=False
            pri[i]=True
for i in range(len(x)):#判断T-prime
    if x[i]==1:
        print("NO")
    elif x[i]**0.5%1==0:
        num=int(x[i]**0.5)
        
        if pri[num-1]:
            print("YES")
        else:
            print("NO")
    else:
        print("NO")
```

代码运行截图

![image-20240301212414133](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240301212414133.png)

### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A

思路：使用两个循环嵌套，将之间的数字加起来判断余数，但是显然会超时，果不其然收获了一连串的Time limit exceeded on test 3。

一开始的代码如下，结果是对的，但两个循环嵌套且每层都有判断所以用时爆炸：

```python
t=int(input())
for i in range(t):
    n,x=map(int,input().split())
    xs=list(map(int,input().split()))
    am=[-1]
    for j in range(n):
        for y in range(j,n+1):
            if j==y and xs[j]%x!=0:
                am.append(1)
            elif(sum(xs[j:y]))%x!=0:
                am.append(y-j)
            elif j==0 and y==n and sum(xs)%x!=0:
                am.append(n)
    print(max(am))
```

尝试减少运算次数，但是缝缝补补一小时无果。先尝试将所有数字转换为余数，但是总想着求sum，所以还是要用循环嵌套（看到答案解法豁然开朗了，其实根本不需要每次求sum）。加上本人不是很会双指针，双指针写了一半写晕乎了……太菜了，大晚上心态炸了，第二天起来再写orz）

最后因为耗时太多，选择看答案，发现有一个思路和我很像的解法——首先是判断所有余数之和是否能被x除尽，若不能，则直接输出整个列表的长度；若能，从两端开始向中间掐（减少了一半循环次数，因为用or来判断两头），只要出现了一个非0数，就说明从这一位开始，”能被整除“的状态就被破坏了。这个数字在前在后不重要，也不需要分情况讨论，只需要知道这个”距离两头的最近的非零余数“的距离减掉就可以了。

（等下也看看双指针怎么搞，先交作业防止错过ddl）

##### 代码

```python
#
t=int(input())
for i in range(t):
    n,x=map(int,input().split())
    xs=list(map(lambda y: int(y)%x,input().split())) 
    #直接运算余数，要用lambda定义一个运算
    am=-1
    if sum(xs)%x: #True说明sum余数不能被整除，即符合题意，直接输出
        print(n)
        continue
    for j in range(n//2+1):
        if xs[j] or xs[~j]:   #说明能被整除的状态被破坏
            am=n-j-1         #减去“距离两头最近的非0余数”
            break
    print(am)
```

代码运行截图

![image-20240302153204352](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240302153204352.png)

### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/

思路：（看完题目的第一感受：控分大佬）照搬230B应该能直接出来（知道欧拉筛的话这一题应该不是很难）。

##### 代码

```python
# 欧拉筛和T-prime（同230B）
pri=[True]*10001
for i in range (1,5001):
    if pri[i]:
        for j in range(i,10001,i+1):
            pri[j]=False
            pri[i]=True
m,n=map(int,input().split())           
for _ in range(m):
    x=list(map(int,input().split()))
    mark=0
    for i in range(len(x)):#直接将非T-p项改成0
        if x[i]==1 :
            x[i]=0
        elif x[i]**0.5%1==0:
            num=int(x[i]**0.5)
            if not pri[num-1]:
                x[i]=0
        else:
            x[i]=0
    
    mark=max(mark,sum(x)/len(x))
    if mark==0:#0不能有小数位，所以直接输出
        print(mark)
    else:
        print("{:.2f}".format(mark))#转换为字符串输出小数点后两位
        #print("%.2f"%mark) ⬅也可以用这个语法
```

代码运行截图

![image-20240302162500409](C:\Users\靳咏歌\AppData\Roaming\Typora\typora-user-images\image-20240302162500409.png)

### 感想：

群内大佬云集，我在其中格格不入……感觉，自己还是太菜了orz。代码越打越晕，算法稍微复杂一些就不会做了。

究其原因，可能是因为一直以来我都偏好用数学方法“偷懒”，做不出来就逃避（因为算法太差了，只能寄希望于数学化简问题，而过度依赖数学方法导致算法熟练度也没提升orz死循环了），这就导致有的时候思路一卡就钻进死胡同出不来了，想换算法但又不会用，就比如这次的XXXXX（指针解法正在尝试中，防止忘交作业就先交了）

本次作业耗时比较长，被”超时“卡住的时候能感觉得到技巧性的东西越来越多，而不是能跑出来正确的output就行。希望自己能保持住学习的动力叭！

