# 进阶修炼计划

# 1. 二分搜索技巧

## 1. 二分查找框架

```python
def search(nums:list, target):
    left, right = 0, ...
    while ...:
        mid = left + (right - left) / 2
        if nums[mid] == target:
            ...
        elif nums[mid] > target:
            right = ...
        elif nums[mid] < target:
            left = ...
    return ...
```

``...``出现的地方是容易出现问题的地方

另外计算mid时容易溢出，所以使用``mid = left + (right - left) / 2``的方法计算

## 2. 基本二分查找 找一个数

```python
def search(nums:list, target:int):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) / 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
    return -1
```

**ATTENTION：**

1. **为什么是`while left <= right:`?**

   当`right = len(nums) - 1`时，搜索区间为`[left, right]`；而当`right = len(nums)`时，搜索区间为`[left, right)`，因为`nums[right]`是越界的

   此时如果`while left < right`搜索区间`[left, left]`不为空，所以需要`=`

2. **为什么`left = mid + 1`，`right = mid - 1`？我看有的代码是`right = mid`或者`left = mid`，没有这些加加减减，到底怎么回事，怎么判断**？

   需要根据搜索区间去判断。当我们发现索引`mid`不是要找的`target`时，我们应该去搜索`[left, mid-1]`或者`[mid+1, right]`对不对？**因为`mid`已经搜索过，应该从搜索区间中去除**。

## 3. 寻找左侧边界的二分搜索

```python
def left_bound(nums:list, target:int):
    if len(nums) == 0: return -1
    left = 0
    right = len(nums)
    while left < right:
        mid = left + (right - left) / 2
        if nums[mid] > target:
            right = mid
        elif nums[mid] < target:
            left = mid + 1
        elif nums[mid] == target:
            right = mid
    return left
```

**ATTENTION：**

**1、为什么 while 中是****`<`而不是`<=`**?

答：用相同的方法分析，因为`right = nums.length`而不是`nums.length - 1`。因此每次循环的「搜索区间」是`[left, right)`左闭右开。

`while(left < right)`终止的条件是`left == right`，此时搜索区间`[left, left)`为空，所以可以正确终止。

**2、为什么没有返回 -1 的操作？****如果`nums`中不存在`target`这个值，怎么办**？

综上可以看出，函数的返回值（即`left`变量的值）取值区间是闭区间`[0, nums.length]`，所以我们简单添加两行代码就能在正确的时候 return -1：

```python
def left_bound(nums:list, target:int):
    if len(nums) == 0: return -1
    left = 0
    right = len(nums)
    while left < right:
        mid = left + (right - left) / 2
        if nums[mid] > target:
            right = mid
        elif nums[mid] < target:
            left = mid + 1
        elif nums[mid] == target:
            right = mid
    if left == len(nums): return -1
    return left if nums[left] == target else -1
```

如果想跟基础的二分搜索算法统一起来的话，可以这么写：

```python
def left_bound(nums:list, target:int):
    if len(nums) == 0: return -1
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + (right - left) / 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        elif nums[mid] == target:
            right = mid - 1
    # 检查left出界情况 target大于所有nums时
    if left >= len(nums) or nums[left] != target:
        return -1
    return left 
```

## 4. 寻找右侧边界的二分搜索

基础代码：

```python
def right_bound(nums:list, target:int):
    if len(nums) == 0: return -1
    left = 0
    right = len(nums)
    while left < right:
        mid = left + (right - left) / 2
        if nums[mid] > target:
            right = mid
        elif nums[mid] < target:
            left = mid + 1
        elif nums[mid] == target:
            left = mid + 1
    if left == 0: return -1
    return left - 1 if nums[left - 1] == target else -1
```

如果也写成跟基础二分搜索框架相同形式的话：

```python
def left_bound(nums:list, target:int):
    if len(nums) == 0: return -1
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + (right - left) / 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        elif nums[mid] == target:
            left = mid + 1
    # 这里改为检查right出界情况 target小于所有nums时
    if right < 0 or nums[right] != target:
        return -1
    return right 
```

# 2. 二分搜索运用

## 1. koko吃香蕉

LeetCode：875题 

首先，算法要求的是「`H`小时内吃完香蕉的最小速度」，我们不妨称为`speed`，**请问`speed`最大可能为多少，最少可能为多少呢？**

显然最少为 1，最大为`max(piles)`，因为一小时最多只能吃一堆香蕉。那么暴力解法就很简单了，只要从 1 开始穷举到`max(piles)`，一旦发现发现某个值可以在`H`小时内吃完所有香蕉，这个值就是最小速度：

```python
def minEatingSpeed(piles, h):
    maxpile = max(piles)
    for speed in range(1,maxpile):
        # 以 speed 是否能在 H 小时内吃完香蕉
        if canFinish(piles, speed, h):
            return speed
    return max
```

注意这个 for 循环，就是在**连续的空间线性搜索，这就是二分查找可以发挥作用的标志**

由于我们要求的是最小速度，所以可以用一个**搜索左侧边界的二分查找**来代替线性搜索，提升效率：

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def canFinish(piles, speed, h):
            time = 0
            for n in piles:
                t = 1 if (n % speed > 0) else 0
                time += (n // speed + t)
            return time <= h

        maxpile = max(piles)
        left = 1
        right = maxpile + 1
        while left < right:
            mid = left + (right - left) // 2
            if canFinish(piles, mid, h):
                right = mid
            else:
                left = mid + 1
        return left
```

借助二分查找技巧，算法的时间复杂度为 O(N*logN)。

## 2. 分割数组的最大值

LeetCode：410题

[二分查找算法如何运用？我和快手面试官进行了深入探讨… (qq.com)](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247487594&idx=1&sn=a8785bd8952c2af3b19890aa7cabdedd&chksm=9bd7ee62aca067742c139cc7c2fa9d11dc72726108611f391d321cbfc25ccb8d65bc3a66762b&scene=21#wechat_redirect)

**现在题目是固定了`m`的值，让我们确定一个最大子数组和；所谓反向思考就是说，我们可以反过来，限制一个最大子数组和`max`，来反推最大子数组和为`max`时，至少可以将`nums`分割成几个子数组**。

```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        def split(nums, t):
            # 至少可以分割的子数组数量
            count = 1
            # 记录每个字数组元素和
            s = 0
            for i in range(len(nums)):
                if (s + nums[i]) > t:
                    # 如果当前子数组和> max限制，则不能添加元素了
                    count += 1
                    s = nums[i]
                else:
                    s += nums[i]
            return count

        lo = max(nums)
        hi = sum(nums) + 1
        while lo < hi:
            mid = lo + (hi - lo) //2
            n = split(nums, mid)
            if n == m:
                hi = mid
            elif n < m:
                hi = mid
            elif n > m:
                lo = mid + 1
             
        return lo
```



# 3. 滑动窗口算法

https://mp.weixin.qq.com/s/ioKXTMZufDECBUwRRp3zaA  
滑动窗口的思路是这样的：  

1. 我们在字符串s中使用双指针中的左右指针技巧，初始化 left = right = 0，把索引左闭右开区间\[left,right)称为一个窗口
2. 我们先不断的增加right指针，扩大窗口\[left,right)，直到窗口中的字符串符合要求（包含了T中的左右字符）
3. 此时，我们停止增加right，转而不断增加left指针，缩小窗口\[left,right)，直到窗口中的字符串不在符合要求（不包含T中的所有字符了）。同时每次增加left，我们都要更新一轮结果
4. 重复2 和 3 步骤，直到right到达字符串s尽头

第 2 步相当于在寻找一个「可行解」，然后第 3 步在优化这个「可行解」，最终找到最优解  
needs和window相当于计数器，分别记录T中字符出现次数和「窗口」中的相应字符的出现次数    


```python
#**滑动窗口算法框架**

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # window 和 need 表示窗口中的字符和需要凑齐的字符
        need = collections.defaultdict(int)
        window = collections.defaultdict(int)
        for c in t:
            need[c] += 1
        left, right = 0, 0
        # valid 表示need中满足条件的字符个数
        valid = 0
      
        while right < len(s):
            # c 是要移入窗口的字符
            c = s[right]
            right += 1     #右移窗口
            #进行窗口内数据的更新
               #...
               
           #debug
           #print("window: [%d, %d]\n", left, right)
           
            #判断左侧窗口是否要收缩
            while valid == len(need):
                #这里更新最小子串
                if right - left < lens:
                    start = left
                    lens = right - left
                #d 是将移除的字符
                d = s[left]
                left += 1
                #进行窗口内数据的更新
               #...
```

## 1. 最小覆盖子串 

 LeetCode：76题


```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # window 和 need 表示窗口中的字符和需要凑齐的字符
        need = collections.defaultdict(int)
        window = collections.defaultdict(int)
        for c in t:
            need[c] += 1
        left, right = 0, 0
        # valid 表示need中满足条件的字符个数
        valid = 0
        # 记录最小覆盖子串的起始位置索引及长度
        start = 0
        lens = 10**6
        
        while right < len(s):
            # c 是要移入窗口的字符
            c = s[right]
            right += 1     #右移窗口
            #进行窗口内数据的更新
            if c in need:
                window[c] += 1
                if window[c] == need[c]:
                    valid += 1
            
            #判断左侧窗口是否要收缩
            while valid == len(need):
                #这里更新最小子串
                if right - left < lens:
                    start = left
                    lens = right - left
                #d 是将移除的字符
                d = s[left]
                left += 1
                if d in need:
                    if window[d] == need[d]:
                        valid -= 1
                    window[d] -= 1
        return s[start:start + lens] if lens != 10**6 else ""
```

## 2.  字符串序列 

LeetCode：567题 

1、本题移动left缩小窗口的时机是窗口大小大于t.size()时，因为排列嘛，显然长度应该是一样的。

2、当发现valid == need.size()时，就说明窗口中就是一个合法的排列，所以立即返回true。


```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        need = collections.defaultdict(int)
        window = collections.defaultdict(int)
        for i in s1:
            need[i] += 1

        left, right = 0, 0
        valid = 0
        while(right < len(s2)):
            c = s2[right]
            right += 1
            if c in need:
                window[c] += 1
                if window[c] == need[c]:
                    valid += 1
            
            while right - left >= len(s1):
                if valid == len(need):
                    return True
                d = s2[left]
                left += 1
                if d in need:
                    if window[d] == need[d]:
                        valid -= 1
                    window[d] -= 1
            
        return False
```

## 3. 无重复字符最长子串 

LeetCode:3题 

这就是变简单了，连need和valid都不需要，而且更新窗口内数据也只需要简单的更新计数器window即可。

**当window[c]值大于 1 时，说明窗口中存在重复字符，不符合条件，就该移动left缩小窗口了嘛。**

唯一需要注意的是，在哪里更新结果res呢？我们要的是最长无重复子串，哪一个阶段可以保证窗口中的字符串是没有重复的呢？


```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        window = collections.defaultdict(int)
        res = 0
        left, right = 0, 0
        
        while right < len(s):
            c = s[right]
            right += 1
            window[c] += 1
            while window[c] > 1:
                d = s[left]
                left += 1
                window[d] -= 1
            
            res = max(res, right - left)
        return res           
```

# 4. LRU（Least Recently Used）策略

LeetCode：146题

思路：哈希表+双向链表

如果要在O(1)时间复杂度内实现put和get操作，就必须要使用这样的数据结构

双向链表需要实现：`addToHead`, `removeNode`,`moveToHead`,`removeTail`这几个API。**其中`addToHead`这里面的操作要注意顺序，否则可能会报错**

```python
# 双向链表类
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = dict()
        # 使用伪头部和伪尾部节点
        self.tail = DLinkedNode()
        self.head = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0


    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 如果key存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value 

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            # 如果key不存在，创建一个新节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双链表头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果key存在，先通过哈希表定位，再修改value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)

    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        # 下面两部顺序很重要，不能随意更改
        self.head.next.prev = node
        self.head.next = node
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



# 5. 动态规划

## 1. 动态规划核心框架

**动态规划问题的一般形式就是求最值**。动态规划其实是运筹学的一种最优化方法，只不过在计算机问题上应用比较多，比如说让你求**最长**递增子序列呀，**最小**编辑距离呀等等。

既然是要求最值，核心问题是什么呢？**求解动态规划的核心问题是穷举**。

- 首先，动态规划的穷举有点特别，因为这类问题**存在「重叠子问题」**，如果暴力穷举的话效率会极其低下，所以需要「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。

- 而且，动态规划问题一定会**具备「最优子结构」**，才能通过子问题的最值得到原问题的最值。

只有列出**正确的「状态转移方程**」才能正确地穷举。

思维框架：

**明确「状态」 -> 定义 dp 数组/函数的含义 -> 明确「选择」-> 明确 base case**

### a.菲波齐纳数列

#### 1.暴力递归

数学表达形式就是递归：

```c++
int fib(int N) {
    if (N == 1 || N == 2) return 1;
    return fib(N - 1) + fib(N - 2);
}
```

这个算法的时间复杂度为 O(2^n)，指数级别，爆炸。

观察递归树，很明显发现了算法低效的原因：存在大量重复计算，比如`f(18)`被计算了两次，而且你可以看到，以`f(18)`为根的这个递归树体量巨大，多算一遍，会耗费巨大的时间。更何况，还不止`f(18)`这一个节点被重复计算，所以这个算法及其低效。

这就是动态规划问题的第一个性质：**重叠子问题**。下面，我们想办法解决这个问题。

#### 2.带备忘录的递归（自顶向下）

一般使用一个数组充当这个「备忘录」，当然你也可以使用哈希表（字典），思想都是一样的。

```c++
int fib(int N) {
    if (N < 1) return 0;
    // 备忘录全初始化为 0
    vector<int> memo(N + 1, 0);
    // 初始化最简情况
    return helper(memo, N);
}

int helper(vector<int>& memo, int n) {
    // base case 
    if (n == 1 || n == 2) return 1;
    // 已经计算过
    if (memo[n] != 0) return memo[n];
    memo[n] = helper(memo, n - 1) + 
                helper(memo, n - 2);
    return memo[n];
}
```

实际上，带「备忘录」的递归算法，把一棵存在巨量冗余的递归树通过「剪枝」，改造成了一幅不存在冗余的递归图，极大减少了子问题（即递归图中节点）的个数。

#### 3.dp数组的迭代解法（自底向上）

```c++
int fib(int N) {
    vector<int> dp(N + 1, 0);
    // base case
    dp[1] = dp[2] = 1;
    for (int i = 3; i <= N; i++)
        dp[i] = dp[i - 1] + dp[i - 2];
    return dp[N];
}
```

### b.凑零钱问题

先看下题目：给你`k`种面值的硬币，面值分别为`c1, c2 ... ck`，每种硬币的数量无限，再给一个总金额`amount`，问你**最少**需要几枚硬币凑出这个金额，如果不可能凑出，算法返回 -1 。算法的函数签名如下：

```c++
// coins 中是可选硬币面值，amount 是目标金额
int coinChange(int[] coins, int amount);
```

#### 1.暴力递归

**先确定「状态」**，也就是原问题和子问题中变化的变量。由于硬币数量无限，所以唯一的状态就是目标金额`amount`。

**然后确定`dp`函数的定义**：函数 dp(n)表示，当前的目标金额是`n`，至少需要`dp(n)`个硬币凑出该金额。

**然后确定「选择」并择优**，也就是对于每个状态，可以做出什么选择改变当前状态。具体到这个问题，无论当的目标金额是多少，选择就是从面额列表`coins`中选择一个硬币，然后目标金额就会减少：

**最后明确 base case**，显然目标金额为 0 时，所需硬币数量为 0；当目标金额小于 0 时，无解，返回 -1

```python
def coinChange(coins: List[int], amount: int):

	def dp(n):
        # base case
        if n < 0: return - 1
        if n == 0: return 0
        res = 10 ** 6
        for coin in coins:
            subproblem = dp(n - coin)
            # 子问题无解，跳过
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)
        return res if res != 10 ** 6 else -1
    return dp(amount)
```

#### 2.带备忘录的递归

```python
def coinChange(coins: List[int], amount: int):
	mem = {}
	def dp(n):
        if n in mem: return mem[n]
        
        # base case
        if n < 0: return - 1
        if n == 0: return 0
        res = 10 ** 6
        for coin in coins:
            subproblem = dp(n - coin)
            # 子问题无解，跳过
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)
        mem[n] = res if res != 10 ** 6 else -1
        return mem[n]
    return dp(amount)
```

#### 3.dp数组的迭代解法

```python
def coinChange(coins: List[int], amount: int):
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    
    for i in range(len(dp)):
        for coin in coins:
            if i - coin < 0:return -1
            dp[i] = min(dp[i], 1 + dp[i - coin])
    return dp[amount] if dp[amount] != amount + 1 else -1
```

## 2.动态规划base case问题

### 931题：最小下降路径和

#### dp数组迭代解法：

```python
class Solution:

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        dp = [[0] * n for _ in range(n)]
        dp[0] = matrix[0]

        for i in range(1, n):
            for j in range(n):
                if j == 0:
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j + 1]) + matrix[i][j]
                elif j == n - 1:
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]) + matrix[i][j]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1], dp[i - 1][j + 1]) + matrix[i][j]
        return min(dp[n - 1])
```

#### 备忘录解法：

```python
class Solution:

    def __init__(self):
        self.mem = []

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        res = 999999
        self.mem = [ [0] * n for _ in range(n)]
        for j in range(n):
            res = min(res, self.dp(matrix, n - 1, j))
        return res
    
    def dp(self, matrix, i, j):
        # 1. 索引合法检查
        if i < 0 or j < 0 or i >= len(matrix) or j >= len(matrix[0]):
            return 99999
        # 2. base case
        if i == 0:
            return matrix[0][j]
        # 3. 查备忘录
        if self.mem[i][j] != 0:
            return self.mem[i][j]
        # 状态转移方程
        self.mem[i][j] = matrix[i][j] + min(self.dp(matrix, i - 1, j), self.dp(matrix, i - 1, j - 1), self.dp(matrix, i - 1, j + 1))
        return self.mem[i][j]
       
```

## 3. 动态规划经典问题：

### 3.1 编辑距离

72题：编辑距离

备忘录解法：

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        mem = {}

        def dp(i, j):
            # base case
            if i == -1: return j + 1
            if j == -1: return i + 1
            if (i, j) in mem:
                return mem[(i, j)]
            if word1[i] == word2[j]:
                mem[(i, j)] = dp(i - 1, j - 1)
            else:
                mem[(i, j)] = min(dp(i - 1, j) + 1, dp(i, j - 1) + 1, dp(i - 1, j - 1) + 1)

            return mem[(i, j)]

        return dp(len(word1) - 1, len(word2) - 1)
```

dp数组解法：

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # base case
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        # 自底向上
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

        return dp[m][n]
```

### 3.2 最小路径和

备忘录解法：

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        mem = {}
        m, n = len(grid), len(grid[0])
        def dp(grid, i, j):
            # base
            if i < 0 or j < 0: return 10 ** 3 
            if i == 0 and j == 0: return grid[0][0]
            
            if (i, j) in mem:
                return mem[(i, j)]
            mem[(i, j)] = min(dp(grid, i - 1, j), dp(grid, i, j - 1)) + grid[i][j]
            return mem[(i, j)]
        return dp(grid, m - 1, n - 1)
```

dp数组解法：

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        # base case
        dp[0][0] = grid[0][0]
        for i in range(1, m):     #注意这里的起始点是1，不是0
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
    
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[m - 1][n - 1]
```

### 3.3 0-1背包问题

简单描述一下吧：

给你一个可装载重量为`W`的背包和`N`个物品，每个物品有重量和价值两个属性。其中第`i`个物品的重量为`wt[i]`，价值为`val[i]`，现在让你用这个背包装物品，最多能装的价值是多少？

套路：

第一步：明确状态与选择

**状态有两个，就是「背包的容量」和「可选择的物品」**。

所以框架是：

```python
for 状态1 in 状态1所有可能的值：
	for 状态2 in 状态2所有可能的值：
    	for ...
        	 dp[状态1][状态2]... = 择优(选择1，选择2...)
```

第二步：明确dp数组的定义

**`dp[i][w]`的定义如下：对于前`i`个物品，当前背包的容量为`w`，这种情况下可以装的最大价值是`dp[i][w]`。**

所以细化一下上面的框架：

```python
dp = [[0] * (W + 1) for _ in (N + 1)]
# base case
dp[0][...] = 0
dp[...][0] = 0

for i in [1,N]:
    for j in [1,W]:
        dp[i][j] = max(把物品i装进包，不把物品i装进包)
return dp[N][W]
```

第三步：根据选择思考状态转移的逻辑

如果没有把第i个物品装进包，则`dp[i][j] =dp[i-1][j]`。

如果把第i个物品装进包，则`dp[i][j] = max(dp[i - 1][j], dp[i - 1][w-wt[i - 1]] + val[i - 1])`

所以最终代码写为：

```python
def knapsack(W:int, N:int, wt:list, val:list):
    dp = [[0] * (W + 1) for _ in (N + 1)]
    for i in range(1, N + 1):
        for j in range(1, W + 1):
            if (w - wt[i - 1] < 0):
                # 当前背包装不下物品i
                dp[i][j] = dp[i - 1][j]
            else:
                # 装或者不装，择优
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][w - wt[i - 1]] + val[i - 1])
    return dp[N][W]
```

## 4. 如何推导状态转移方程

### 4.1 最长递增子序列

300题：最长递增子序列

> 注意子序列和子串的区别，子串必须是连续的，而子序列不需要

思路一：动态规划

首先明确dp数组定义

dp数组定义为：`dp[i]`是以`nums[i]`结尾的最长递增子序列长度

接下来推导状态转移方程

当`dp[i-1]`是以`nums[i-1]`结尾的最长子序列时，`dp[i]`是多少？

![](进阶修炼计划.assets/微信图片_20211115101859.gif)

可以看出**我们只要找到前面那些结尾比 3 小的子序列，然后把 3 接到最后，就可以形成一个新的递增子序列，而且这个新的子序列长度加一**。

伪代码如下：

```python
for j in range(0,i):
    if nums[i] > nums[j] :
        dp[i] = max(dp[i], dp[j] + 1)
```

整体代码如下：

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * (n + 1)

        for i in range(0, n):
            for j in range(0, i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j] + 1)
        return max(dp)
```

思路二：二分搜索

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        top = [0] * n
        piles = 0
        for i in range(0, n):
            poker = nums[i]

            left, right = 0, piles
            while left < right:
                mid = left + (right - left) // 2
                if top[mid] > poker:
                    right = mid
                elif top[mid] < poker:
                    left = mid + 1
                else:
                    right = mid

            if left == piles: piles += 1
            top[left] = poker
        return piles
```

思路三：动态规划+二分搜索

**降低复杂度切入点：** 解法一中，遍历计算 dp 列表需 O(N)，计算每个 dp[k] 需 O(N)。

- 动态规划中，通过线性遍历来计算 dp 的复杂度无法降低；
- 每轮计算中，需要通过线性遍历 [0,k)区间元素来得到 dp[k] 。我们考虑：是否可以通过重新设计状态定义，使整个 dp 为一个排序列表；这样在计算每个 dp[k]时，就可以通过二分法遍历 [0,k)区间元素，将此部分复杂度由 O(N) 降至 O(logN)。

**算法流程：**

- 状态定义：
  - tails[k]的值代表 长度为 k+1 子序列 的尾部元素值。

- 转移方程： 设 res 为 tails 当前长度，代表直到当前的最长上升子序列长度。设 j∈[0,res)，考虑每轮遍历 nums[k] 时，通过二分法遍历 [0,res) 列表区间，找出 nums[k] 的大小分界点，会出现两种情况：
  - 区间中存在 tails[i] > nums[k] ： 将第一个满足 tails[i] > nums[k]执行 tails[i] = nums[k] ；因为更小的 nums[k]后更可能接一个比它大的数字（前面分析过）。
  - 区间中不存在 tails[i] > nums[k] ： 意味着 nums[k] 可以接在前面所有长度的子序列之后，因此肯定是接到最长的后面（长度为 res ），新子序列长度为 res + 1。

- 初始状态：
  - 令 tails列表所有值 =0。

- 返回值：
  - 返回 res ，即最长上升子子序列长度。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        tails, res = [0] * n, 0
        for num in nums:
            i, j = 0, res
            while i < j:
                m = (i + j) // 2
                if tails[m] < num: i = m + 1
                else: j = m
            tails[i] = num
            if j == res: res += 1
        return res
```

### 4.2 最长递增子序列之嵌套信封问题

354题：

思路一：最长递增子序列问题

先把宽度按升序排列，再把宽度相同的按长度降序排列，之后再对长度求LIS问题

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0

        n = len(envelopes)
        # 按照宽度升序排序，如果宽度一样，则按高度降序排列
        envelopes.sort(key=lambda x: (x[0], -x[-1]))

        f = [1] * n
        for i in range(n):
            for j in range(i):
                if envelopes[i][1] > envelopes[j][1] :
                    f[i] = max(f[i], f[j] + 1)
        return max(f)

```

时间复杂度O(N^2)

思路二：用二分法优化时间复杂度

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0

        n = len(envelopes)
        # 按照宽度升序排序，如果宽度一样，则按高度降序排列
        envelopes.sort(key=lambda x: (x[0], -x[-1]))

        f = [envelopes[0][1]]
        for i in range(1, n):
            if (num := envelopes[i][1]) > f[-1]:
                f.append(num)
            else:
                index = bisect.bisect_left(f, num)
                f[index] = num
        return len(f)
```

> `:=`是海象运算符，在python3.8之后出现的算法，主要作用是，**把计算语句的结果赋值给变量，然后，变量可以在代码块里执行运用**

## 5. 动态规划套路

### 5.1 最大子数组和

思路：动态规划

第一步：dp数组定义：以nums[i]为结尾的最大子数组和

第二步：状态转移方程：

如果`dp[i-1]`是以`nums[i-1]`为结尾的最大子数组和，那么对于`nums[i]`有两个选择：

- 要么与前面的数组相连，形成一个数组和更大的数组
- 要么自成一派

所以状态转移方程是`dp[i] = max(nums[i], nums[i] + dp[i - 1])`

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums: return 0

        n = len(nums)
        dp = [0] * n
        # base case
        dp[0] = nums[0]
        # 状态转移方程
        for i in range(1, n):
            # 自成一派或者加上前面的子数组
            dp[i] = max(nums[i], nums[i] + dp[i - 1])
        return max(dp)
```

如果进行状态压缩，可以进一步减少空间复杂度

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # base case
        dp_0 = nums[0]
        dp_1 = 0
        res = dp_0
        
        # 状态转移方程
        for i in range(1, n):
            # 自成一派或者加上前面的子数组
            dp_1 = max(nums[i], nums[i] + dp_0)
            dp_0 = dp_1
            res = max(res, dp_1)
        return res
```

### 5.2 最长公共子序列系列问题

#### a. 最长公共子序列

1143题：

思路：动态规划

第一步：明确dp函数的定义

dp函数定义：dp(self, s1, i, s2, j)：表示计算s1[i]和s2[j]的最长公共子序列长度

第二步：状态转移方程

**如果`s1[i] == s2[j]`，说明这个字符一定在`lcs`中**：

​	`dp(s1,i,s2,j) = 1 + dp(s1, i + 1, s2, j + 1)`

**`s1[i] != s2[j]`意味着，`s1[i]`和`s2[j]`中至少有一个字符不在`lcs`中**：（两个都不在的情况不讨论，因为一定小于另外两种情况）

​	`dp(s1,i,s2,j) = max(dp(s1, i + 1, s2, j), dp(s1, i, s2, j + 1))`

>  注意：备忘录法去除重叠子问题

```python
class Solution:
    def __init__(self):
        self.memo = []

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        self.memo = [[-1] * n for _ in range(m)]
        return self.dp(text1, 0, text2, 0)
    
    # 定义计算s1[i]和s2[j]的最长公共子序列长度
    def dp(self, s1, i, s2, j):
        # base case
        if i == len(s1) or j == len(s2):
            return 0
        # 备忘录
        if self.memo[i][j] != -1:
            return self.memo[i][j]
        
        if s1[i] == s2[j]:
            self.memo[i][j] = 1 + self.dp(s1, i + 1, s2, j + 1)
        else:
            # s1[i] s2[j]至少有一个不在lcs中
            self.memo[i][j] = max(
                self.dp(s1, i + 1, s2, j),
                self.dp(s1, i, s2, j + 1)
            )
        return self.memo[i][j]
```

或者 dp数组法

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        # base case dp[0][...] = dp[...][0] = 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # i j 从1开始，所以要-1
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
```

#### b. 两个字符串的删除

583题：

思路一：动态规划（求最长公共子序列）

题目让我们计算将两个字符串变得相同的最少删除次数，那我们可以思考一下，最后这两个字符串会被删成什么样子？

删除的结果不就是它俩的最长公共子序列嘛！

那么，要计算删除的次数，就可以通过最长公共子序列的长度推导出来：

`ans = m + n - lcs * 2`

具体代码如下：

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        lcs = self.longestCommonSubsequence(word1, word2)
        return m + n - lcs * 2
    
    def longestCommonSubsequence(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
```

思路二：直接用动态规划

第一步：dp数组定义：`dp[i][j]`代表s1[:i], s2[:j]要变为相同字符串的最小删除次数

第二步：状态转移方程：

- s1[i] == s2[j]：`dp[i][j] = dp[i - 1][j - 1]`
- s1[i] != s2[j]：`dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j -  1] + 1)`代表至少一个删除 s1[i] 和 s2[j] 中的其中一个。

`dp[i][j]`代表上述方案的最小值

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 这里求的是最小删除次数，所以是min
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1)
        return dp[m][n]
```

#### c. 两个字符串的最小ASCII删除和

712题：

思路：动态规划

注意base case 的情况：

- 当s1到头，需要把s2剩下的都删掉
- 同理，当s2到头，需要把s1剩下的都删掉

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        # base case
        for i in range(1, m + 1): dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
        for j in range(1, n + 1): dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + ord(s1[i - 1]), dp[i][j - 1] + ord(s2[j - 1]))
        return dp[m][n]
```

### 5.3 最长回文子序列问题

516题：

思路：动态规划

第一步：dp数组定义

**在子串`s[i..j]`中，最长回文子序列的长度为`dp[i][j]`**

第二步：状态转移方程

如果`dp[i + 1][j - 1]`是回文子序列，那么`dp[i][j]=?`

- 如果两个字符相同`s[i] == s[j]`，那么长度直接 + 2`dp[i][j] = dp[i + 1][j - 1] + 2`
- 如果两个字符不同`s[i] != s[j]`，那么说明它俩**不可能同时**出现在`s[i..j]`的最长回文子序列中，那么把它俩**分别**加入`s[i+1..j-1]`中，看看哪个子串产生的回文子序列更长即可：`dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])`

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        # base case
        for i in range(n):
            dp[i][i] = 1
        # 反着遍历 保证正确的状态转移
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
        return dp[0][n - 1]
```

# 6. 回溯

解决一个回溯问题，实际上是一个决策树的遍历问题。需要考虑三个问题

1. 路径：也就是已经做出的选择
2. 选择列表：你当前可以做的选择
3. 结束条件：到达决策树的底层，无法再做选择的条件

回溯算法的框架：

```python
res = []
def backtrack(路径，选择):
    if 满足结束条件:
        res.append(路径)
        return 
    for 选择 in 选择列表:
        做选择
        backtrack(路径，选择列表)
        撤销选择
       
```



