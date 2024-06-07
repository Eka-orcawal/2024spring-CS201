数算 cheatsheet 在班内tlsq同学的基础上加入了不擅长的算法例题和会用但容易忘的内容

### 列表：

列表推导式：[声明变量 for 声明变量 in 某集合 if 共同满足的条件]

插入元素：list.insert(index,元素)；bisect库

删除已知元素：list.remove(元素)     删除已知索引的元素：del list[index]

倒序排序：list.sort(reverse=True)     指定顺序排序：list.sort(key= lambda s:排序指标（与s相关）

寻找索引：list.index(元素) 第一个元素，没有会触发ValueError 

 enumerate(列表，遍历开始的位置)返回为：位置，对应位置的值

### 字典：

半有序：Ordereddict()   添加值：dict[key]=value

遍历字典的键：for 元素 in dict() ； for 元素 in dict.keys()     遍历字典的值：for 元素 in dict.values()

删除键值对：del dict[键]      遍历键值对：for key,value in dict.items():

### 集合：

set.add()一个  set.update()多个

删除元素：set.remove() 或set.discard()（前者有KeyError风险，后者没有）

随机删除：set.pop()

运算：并集：set1 | set2   交集：set1 & set2  差集（补集）：set1 - set2  对称差集（补集之交）：set1^set2

## 库：import collections from

### math库

向上取整：math.ceil()   向下取整：math.floor()   阶乘：math.factoria()

数学常数：math.pi（圆周率），math.e（自然对数的底）

math.sqrt(x), math.pow(x,y), math.exp(x), math.log(真数，底数)（默认为自然对数）

### heapq库：（详见27947动态中位数

heapq.heapify(list)

heapq.heappush(堆名，被插元素)  heapq.heappop(堆名)

插入元素的同时弹出顶部元素：heapq.heappushpop(堆名，被插元素)  （或heapq.heapreplace(堆名，被插元素)）

·以上操作在最大堆中应换为“_X_max”（X是它们中的任意一个）

### itertools库：

整数集：itertools.count(x,y)（从x开始往大数的整数，间隔为y）

循环地复制一组变量：itertools.cycle(list)

所有排列：itertools.permutations(集合，选取个数)   所有组合：itertools.combinations

已排序列表去重：[i for i,_ in itertools.groupby(list)]（每种元素只能保留一个）

​      或者list(group)[:n]（group被定义为分组，保留每组的n个元素）

### collections库：

双端队列：

​	创建：a=deque(list)

​	从末尾添加元素：a.append(x)。从末尾删除元素：b=a.pop()

​	从开头添加元素：a.appendleft(x)。从开头删除元素：b=a.popleft()

默认值字典：a=defaultdict(默认值)，如果键不在字典中，会自动添加值为默认值的键值对，适合创建空字典

计数器：Counter(str)，返回以字符种类为键，出现个数为值的字典

### sys库：

sys.exit()用于及时退出程序   sys.setrecursionlimit()用于调整递归限制

### statistics库： 

1.mean(data)：平均值（均值）。2.harmonic_mean(data)：调和平均数。3.median(data)：中位数。

4.median_low(data)：低中位数。5.median_high(data)：高中位数。6.median_grouped(data, interval=1)：分组估计中位数。

7.mode(data)：计算众数。8.pstdev(data)：计算总体标准差。9.pvariance(data)：计算总体方差。10.stdev(data)：计算样本标准差。

11.variance(data)：计算样本方差。

## 数据处理：

二进制：bin()，八进制：oct()，十六进制：hex()

保留n位小数：round(原数字，n)；’%.nf’%原数字；’{:.nf}’.format(原数字)；

n位有效数字：’%.ng’%原数字；’{:.ng}’.format(原数字)

**ASCII转字符：chr();字符转ASCII：ord()** 

## 算法

### dp

步骤：1、定义矩阵（全零或负无穷）

2、遍历矩阵（顺带遍历可选项）

3、遇到可放入：状态转移方程：`a[i][j]=max(a[i-1][j-t]+value[t],a[i-1][j])`,t为物品对空间的占用

4、按情况输出矩阵的一个格子（通常是`a[-1][-1]`）

### dfs（深度优先搜索）

步骤：1、定义函数；

2、在函数内部，判断是否终了，若是，存下状态，return；

3、一层层地判断：符合条件就打标记，递归调用，进入下一层；记得在递归调用完后抹除标记，以便搜索其他分支；

4、在函数外调用函数，注意实参是否正确；

5、输出可能结果

#### 八皇后

```python
answer = []
def Queen(s):
    for col in range(1, 9):
        for j in range(len(s)):
            if (str(col) == s[j] or # 两个皇后不能在同一列
                    abs(col - int(s[j])) == abs(len(s) - j)): # 两个皇后不能在同一斜线
                break
        else:
            if len(s) == 7:
                answer.append(s + str(col))
            else:
                Queen(s + str(col))
Queen('')
```

```python
def is_safe(board, row, col):
    for i in range(row):# 检查同一列是否有皇后
        if board[i] == col:
            return False
    # 检查左上方是否有皇后
    i = row - 1
    j = col - 1
    while i >= 0 and j >= 0:
        if board[i] == j:
            return False
        i -= 1
        j -= 1
    # 检查右上方是否有皇后
    i = row - 1
    j = col + 1
    while i >= 0 and j < 8:
        if board[i] == j:
            return False
        i -= 1
        j += 1
    return True
def queen_dfs(board, row):
    if row == 8:
        # 找到第b个解，将解存储到result列表中
        ans.append(''.join([str(x+1) for x in board]))
        return
    for col in range(8):
        if is_safe(board, row, col):            
            board[row] = col# 当前位置安全，放置皇后
            queen_dfs(board, row + 1)# 继续递归放置下一行的皇后
            board[row] = 0# 回溯，撤销当前位置的皇后
ans = []
queen_dfs([None]*8, 0)
```

无向图有无连通回路：

```python
n,m=map(int,input().split())
edge=[[] for _ in range(n)]
for _ in range(m): #存储联通关系
    u,v=map(int,input().split())
    edge[u].append(v)
    edge[v].append(u)
cnt,flag=set(),False
def dfs(x,y): #x为当前点，y为父节点
    global cnt,flag
    cnt.add(x)
    for i in edge[x]: #遍历当前节点的邻居
        if i not in cnt: #对还没访问过的邻点继续dfs
            dfs(i,x)
        elif y !=i: #邻点已经被访问过，且不是父节点
            flag=True            
for i in range(n): #对每个点进行dfs
    cnt.clear() #清空之前访问的set
    dfs(i,-1)
    if len(cnt)==n: #联通的
        break
    if flag: #循环的
        break    
print("connected:"+("yes" if len(cnt)==n else "no"))
print("loop:"+("yes" if flag else "no"))
```

18160：最大连通域面积

```python
dire = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]
area = 0
def dfs(x,y):
    global area
    if matrix[x][y] == '.':return
    matrix[x][y] = '.'
    area += 1
    for i in range(len(dire)):
        dfs(x+dire[i][0], y+dire[i][1])
T=int(input())
for t in range(T):
    N,M=map(int,input().split()) 
    matrix = [['.' for _ in range(M+2)] for _ in range(N+2)]
    for i in range(1,N+1):
        matrix[i][1:-1] = input()        
    sur = 0
    for i in range(1,N+1):
        for j in range(1,M+1):
            if matrix[i][j] == 'W':
                area = 0 
                dfs(i, j)
                sur = max(sur, area)
    print(sur)
```

### bfs（广度优先搜索）

定义好所有方向，按方向遍历。记得标记好走过的地点。

#### 寻宝

Billy获得了一张藏宝图，图上标记了普通点（0），藏宝点（1）和陷阱（2）。按照藏宝图，Billy只能上下左右移动，每次移动一格，且途中不能经过陷阱。现在Billy从藏宝图的左上角出发，请问他是否能到达藏宝点？如果能，所需最短步数为多少？

**输入**

第一行为两个整数m,n，分别表示藏宝图的行数和列数。(m<=50,n<=50) 此后m行，每行n个整数（0，1，2），表示藏宝图的内容。 

**输出** 

如果不能到达，输出‘NO’。 如果能到达，输出所需的最短步数（一个整数）。 

```python
import heapq 
def bfs(x,y):
    d=[[-1,0],[1,0],[0,1],[0,-1]] 
	queue=[] 
	heapq.heappush(queue,[0,x,y])
	check=set() 
	check.add((x,y))
	while queue: 
		step,x,y=map(int,heapq.heappop(queue))
		if martix[x][y]==1: 
            return step
		for i in range(4): 
			dx,dy=x+d[i][0],y+d[i][1]
		if martix[dx][dy]!=2 and (dx,dy) not in check: 
			heapq.heappush(queue,[step+1,dx,dy])
			check.add((dx,dy))
	return "NO" 

m,n=map(int,input().split())
martix=[list(map(int,input().split())) for i in range(m)]
print(bfs(0,0))
```

20106：走山路

```python
from heapq import heappop, heappush

def bfs(sx, sy):
    q = [(0, sx, sy)]
    visited = set()
    while q:
        ans, x, y = heappop(q)
        if (x, y) in visited:	# 剪枝
            continue
        visited.add((x, y))
        if x == ex and y == ey:
            return ans
        for dx, dy in direc:
            nx, ny = x+dx, y+dy
            if 0 <= nx < m and 0 <= ny < n and \
                    Map[nx][ny] != '#' and (nx, ny) not in visited:
                new_ans = ans+abs(int(Map[nx][ny])-int(Map[x][y]))
                heappush(q, (new_ans, nx, ny))
    return 'NO'


m, n, p = map(int, input().split())
Map = [list(input().split()) for _ in range(m)]
direc= [(1, 0), (-1, 0), (0, 1), (0, -1)]
for _ in range(p):
    sx, sy, ex, ey = map(int, input().split())
    if Map[sx][sy] == '#' or Map[ex][ey] == '#':
        print('NO')
        continue
    print(bfs(sx,sy))
```

04115：鸣人和佐助

```python
from collections import deque

direc=[(0,1),(1,0),(0,-1),(-1,0)]
start,end=None,None

M,N,T=map(int,input().split())
Map=[list(input()) for i in range(M)]

for i in range(M):
    for j in range(N):
        if Map[i][j]=="@":
            start=(i,j)
def bfs():
    q=deque([start +(T,0)])
    visited=[[-1]*N for i in range(M)] #用来记录每个点的访问状态与剩余查克拉
    visited[start[0]][start[1]]=T
    while q:
        x,y,t,time=q.popleft()
        time+=1
        for dx,dy in direc:
            if 0<=x+dx<M and 0<=y+dy<N: #不能出地图
                if  (elem:=Map[x+dx][y+dy])=="*" and t>visited[x+dx][y+dy]: #路上没有大蛇丸手下
                    visited[x+dx][y+dy]=t
                    q.append((x+dx,y+dy,t,time))
                elif elem=="#" and t>0 and t-1>visited[x+dx][y+dy]:
                    #路上有大蛇丸手下，如果有富余查克拉可以走
                    visited[x+dx][y+dy] =t-1
                    q.append((x+dx,y+dy,t-1,time))
                elif elem =="+":
                    return time
    return -1
print(bfs())
```

### 二分查找

```python
import bisect
sorted_list = [1,3,5,7,9] #[(0)1, (1)3, (2)5, (3)7, (4)9]
position = bisect.bisect_left(sorted_list, 6)
print(position)  # 输出：3，因为6应该插入到位置3，才能保持列表的升序顺序

bisect.insort_left(sorted_list, 6)
print(sorted_list)  # 输出：[1, 3, 5, 6, 7, 9]，6被插入到适当的位置以保持升序顺序

sorted_list=(1,3,5,7,7,7,9)
print(bisect.bisect_left(sorted_list,7))
print(bisect.bisect_right(sorted_list,7)) # 输出：3 6
```

### 欧拉筛

```python
N=20
primes = []
is_prime = [True]*N
is_prime[0] = False;is_prime[1] = False
for i in range(2,N):
    if is_prime[i]:
        primes.append(i)
    for p in primes: #筛掉每个数的素数倍
        if p*i >= N:
            break
        is_prime[p*i] = False
        if i % p == 0: #这样能保证每个数都被它的最小素因数筛掉！
            break
print(primes)#得到的是20以内的素数
```

## 树

```python
class TreeNode():
   def __init__(self,val):
        self.value=val
        self.left=None
        self.right=None
N=int(input())
nodes=[TreeNode(i) for i in range(N)]#存放节点，防止重复创建实例
```

```python
# 前中建后：ps二叉搜索树的节点值从小到大排序就是中序遍历
def buildTree(preorder,inorder):
    if not preorder or not inorder: #没有节点
        return None
    root_value=preorder.pop(0)
    root=TreeNode(root_value) 
    root_index=inorder.index(root_value)
    root.right=buildTree(preorder,inorder[root_index+1:]) 
    root.left=buildTree(preorder,inorder[:root_index])   
    return root
```

```python
# 中后建前：
def buildTree(inorder,postorder):
    if not inorder or not postorder: #没有节点
        return None
    root_value=postorder.pop()
    root=TreeNode(root_value) 
    root_index=inorder.index(root_value)
    root.right=buildTree(inorder[root_index+1:],postorder) 
    root.left=buildTree(inorder[:root_index],postorder)
    
    return root
```

```python
#遍历
def preorder(root): 
    result=[]
    if root:
        result.append(root.value)
        result.extend(preorder(root.left))
        result.extend(preorder(root.right))
    return result
def postorder(root): 
    result=[]
    if root:
        result.extend(postorder(root.left))
        result.extend(postorder(root.right))
        result.append(root.value)
    return result
def level_order_traversal(root):
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left) #将下一深度的左节点放进去
        if node.right:
            queue.append(node.right)#将下一深度的右节点放进去
    return traversal
```

## 并查集

1.	初始化：将每一个节点的归属设为其自身；
2.	判断：判断两个节点是否同归属，在判断的过程中路径压缩；
3.	合并：将节点的归属按要求重新设置；
4.	关于路径压缩：只要节点的归属不是其父节点，就将其父节点设为总的根节点

```python
class Cola():
    def __init__(self,val):
       self.value=val
       self.parent=self
    def find(self):
        if self.parent==self:
            return self
        else:
            self.parent=self.parent.find()
            return self.parent
    def union(self,other):
        root_self = self.find()  # 找到当前对象所在集合的根节点
        root_other = other.find()  # 找到其他对象所在集合的根节点
        if root_self != root_other:  # 如果两个对象不属于同一个集合
            root_other.parent = root_self 
            print("No")
        else:
            print('Yes')
while True:
    try:
        n,m=map(int,input().split())
        colas=[0]+[Cola(i) for i in range(1,n+1)]
        for _ in range(m):
            x,y=map(int,input().split())
            colas[x].union(colas[y])
        counter=set()
        for i in range(1,n+1):
            counter.add(colas[i].find().value)
        counter=list(counter)
        counter.sort()
        counter=list(map(str,counter))
        print(len(counter))
        print(' '.join(counter))
    except EOFError:
        break
```

## 堆

27947动态中位数：

```python
import heapq
def dynamic_median(nums):
    # 维护小根和大根堆（对顶），保持中位数在大根堆的顶部
    min_heap = []  # 存储较大的一半元素，使用最小堆
    max_heap = []  # 存储较小的一半元素，使用最大堆
    median = []
    for i, num in enumerate(nums):
        # 根据当前元素的大小将其插入到对应的堆中
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)
        # 调整两个堆的大小差，使其不超过 1
        if len(max_heap) - len(min_heap) > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
        if i % 2 == 0:
            median.append(-max_heap[0])
    return median
T = int(input())
for _ in range(T):
    #M = int(input())
    nums = list(map(int, input().split()))
    median = dynamic_median(nums)
    print(len(median))
    print(*median)
```

## 单调栈：

```python
n=int(input())
lst=list(map(int,input().split()))
stack=[]

for i in range(len(lst)):
    while stack and lst[stack[-1]]<lst[i]:
        lst[stack.pop()]=str(i+1)
	stack.append(i)
while stack:
	lst[stack.pop()]='0'
print(' '.join(lst))#实际操作中，可以再单独开一个数组存放结果，以避免数据覆盖
```

快速堆猪：

```python
pigs_stacked=[]
lightest=[]
while True:
    try:
        operation=input().split()
        ope=operation[0]
        if not pigs_stacked:
            if ope=="push":
                pigs_stacked.append(int(operation[1]))
                lightest.append(pigs_stacked[-1])
            else:
                continue
        else:
            if ope=="pop":
               pigs_stacked.pop()
               if lightest:
                   lightest.pop()               
            elif ope=="min":
                print(lightest[-1])                
            else:
                pigs_stacked.append(int(operation[1]))
                lightest.append(min(pigs_stacked[-1],lightest[-1]))         
    except EOFError:
        break
```

其他栈的例题：

合法出栈顺序：

```python
def valid_pop(origin,output):
    if len(origin)!=len(output):
        return False
    else:
        stack=[]
        ori=list(origin)
        for char in output:         
            while (not stack or stack[-1]!=char) and ori:   #将原字符按顺序推进栈，直到栈顶output下当前char/全部推完 
                stack.append(ori.pop(0))
            # if stack[-1]==char
            #	stack.pop()
            #else:
            #	return False
            if  stack[-1]!=char #or not stack:
                return False    #原字符全部进栈时才会执行这一步，匹配不上，说明不合法（这里不用判断空栈，因为进行到这里，栈不可能为空，空栈说明所有字符都匹配弹出了）
            else: #只要在栈顶有char匹配，随时弹出,然后进下一轮
                stack.pop()            
        return True
origin=input()
```

约瑟夫问题2：

```python
while True:
    n,p,m=map(int,input().split())
    if n==0:
        break
    children_out=[]
    num=list(range(1,n+1))
    for i in range(p-1): #前面先洗队
        tmp=num.pop(0)
        num.append(tmp)
    count=0
    while num: #开始传球
        tmp=num.pop(0)
        count+=1
        if count==m:
            count=0
            children_out.append(tmp)
            continue
        num.append(tmp)       

    print(",".join(map(str,children_out)))
```

括号嵌套树：

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点
def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)
def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)
s = input().strip()
s = ''.join(s.split())  # 去掉所有空白字符
root = parse_tree(s)  # 解析整棵树
if root:
    print(preorder(root))  # 输出前序遍历序列
    print(postorder(root))  # 输出后序遍历序列
else:
    print("input tree string error!")
```

## 图

### Dijkstra：

```python
import heapq
def dijkstra(N, G, start):
	INF = float('inf')
	dist = [INF] * (N + 1) # 存储源点到各个节点的最短距离
	dist[start] = 0 # 源点到自身的距离为0
	pq = [(0, start)] # 使用优先队列，存储节点的最短距离
	while pq:
        d, node = heapq.heappop(pq) # 弹出当前最短距离的节点
        if d > dist[node]: # 如果该节点已经被更新过了，则跳过
            continue
		for neighbor, weight in G[node]: # 遍历当前节点的所有邻居节点
			new_dist = dist[node] + weight # 计算经当前节点到达邻居节点的距离
				if new_dist < dist[neighbor]: # 如果新距离小于已知最短距离，则更新最短距离
					dist[neighbor] = new_dist
					heapq.heappush(pq, (new_dist, neighbor)) # 将邻居节点加入优先队列
	return dist

N, M = map(int, input().split())
G = [[] for _ in range(N + 1)] # 图的邻接表表示
for _ in range(M):
    s, e, w = map(int, input().split())
	G[s].append((e, w))
start_node = 1 # 源点
shortest_distances = dijkstra(N, G, start_node) # 计算源点到各个节点的最短距离
print(shortest_distances[-1])
```

兔子与樱花

```python
import heapq
def dijkstra(adjacency, start):
    distances = {vertex: float('infinity') for vertex in adjacency}
    previous = {vertex: None for vertex in adjacency}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in adjacency[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous

def shortest_path_to(adjacency, start, end):
    distances, previous = dijkstra(adjacency, start)
    path = []
    current = end
    while previous[current] is not None:
        path.insert(0, current)
        current = previous[current]
    path.insert(0, start)
    return path, distances[end]

P = int(input())
places = {input().strip() for _ in range(P)}

Q = int(input())
graph = {place: {} for place in places}
for _ in range(Q):
    x, y, dist = input().split()
    dist = int(dist)
    graph[x][y] = dist
    graph[y][x] = dist 

R = int(input())
for Rs in range(R):   
    start,end = input().split()
    if start==end:
        print(start)
        continue    
    path, total_dist = shortest_path_to(graph, start, end)
    output = ""
    for i in range(len(path) - 1):
        output += f"{path[i]}->({graph[path[i]][path[i+1]]})->"
    output += f"{end}"
    print(output)
```

### Prim（最小生成树）：

```python
import heapq
def prim(graph):
    # 初始化最小生成树的顶点集合和边集合
	mst = set()
	edges = []
	visited = set()
	total_weight = 0
	start_vertex = list(graph.keys())[0] # 随机选择一个起始顶点
	mst.add(start_vertex)# 将起始顶点加入最小生成树的顶点集合中
	visited.add(start_vertex)	# 将起始顶点的所有边加入边集合中
	for neighbor, weight in graph[start_vertex]:
		heapq.heappush(edges, (weight, start_vertex, neighbor))	# 循环直到所有顶点都加入最小生成树为止
	while len(mst) < len(graph):
		weight, u, v = heapq.heappop(edges)		# 从边集合中选取权重最小的边
		if v in visited:		# 如果边的目标顶点已经在最小生成树中，则跳过
			continue
		mst.add(v)# 将目标顶点加入最小生成树的顶点集合中
		visited.add(v)
		total_weight += weight
		for neighbor, weight in graph[v]:# 将目标顶点的所有边加入边集合中
			if neighbor not in visited:
				heapq.heappush(edges, (weight, v, neighbor))
	return total_weight

n = int(input())
graph = {}
for _ in range(n - 1):
    alist = list(input().split())
    if alist[0] not in graph.keys():
		graph[alist[0]] = []
	for i in range(1, int(alist[1]) + 1):
		if alist[2 * i] not in graph.keys():
			graph[alist[2 * i]] = []
		graph[alist[0]].append((alist[2 * i], int(alist[2 * i + 1])))
		graph[alist[2 * i]].append((alist[0], int(alist[2 * i + 1])))
print(prim(graph))
```

兔子与星空：

```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [(cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))
    return mst

def solve():
    n = int(input())
    graph = {chr(i+65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star] = cost
            graph[to_star][star] = cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))
```

### 拓扑排序：

```python
from collections import deque, defaultdict
#实际应用中可能需要import heapq
def topological_sort(graph):
	indegree = defaultdict(int)
    result = []
    queue = deque()
	for u in graph:# 计算每个顶点的入度
		for v in graph[u]:
            indegree[v] += 1
	for u in graph:# 将入度为 0 的顶点加入队列
        if indegree[u] == 0:
            queue.append(u)
	while queue:
		u = queue.popleft()
		result.append(u)
		for v in graph[u]:
			indegree[v] -= 1
			if indegree[v] == 0:
				queue.append(v)
	if len(result) == len(graph):
        return result
	else:
        return None
```

舰队、海域出击：

```python
from collections import deque
def topo_sort(graph):
    in_degree={u:0 for u in range(1,N+1)} #记录入度
    for u in graph:
        for v in graph[u]:
            in_degree[v]+=1
    q=deque([u for u in in_degree if in_degree[u]==0]) #从入度为0的开始
    topo_order=[]
    while q:
        u=q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v]-=1 #将相连点的入度-1
            if in_degree[v]==0:
                q.append(v)
    if len(topo_order)!=len(graph):
        return 'Yes'
    return 'No' 

T=int(input())
for t in range(T):
    N,M=map(int,input().split())
    Map={i:[] for i in range(1,N+1)}
    for m in range(M):
        x,y=map(int,input().split())
        Map[x].append(y)
    print(topo_sort(Map))
```

