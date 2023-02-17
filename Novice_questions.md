# 小白刷题攻略

# 1. 链表部分

## 1. 递归反转链表
### 剑指 Offer II 024. 反转链表


```python
#递归法解决，时间复杂度o(n),空间复杂度o(n)
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None: return head
        last = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return last
    
#迭代法解决，时间复杂度o(n),空间复杂度o(1)
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            temp = cur.next
            cur.next = pre
            pre, cur = cur, temp
        return pre
```

### 反转链表前n个节点
具体的区别：

1. base case 变为n == 1，反转一个元素，就是它本身，同时要记录后驱节点。

2. 刚才我们直接把``head.next``设置为 null，因为整个链表反转后原来的head变成了整个链表的最后一个节点。但现在head节点在递归反转之后不一定是最后一个节点了，所以要记录后驱successor（第 n + 1 个节点），反转之后将head连接上。


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def __init__(self):
    self.successor = None    #后驱节点
    
    #反转以head为起点的n个节点，返回新的头结点
    def reverseN(self, head: ListNode, n: int) -> ListNode:
        if n == 1:
            #记录第n+1个节点
            self.successor = head.next
            return head
        # 以head.next为起点，需要反转前n-1个节点
        last = self.reverseN(head.next, n - 1)
        head.next.next = head
        #让反转之后的head节点和后面的节点连起来
        head.next = self.successor
        return last
```

### 反转链表的一部分
现在解决我们最开始提出的问题，给一个索引区间[m,n]（索引从 1 开始），仅仅反转区间中的链表元素


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def __init__(self):
    self.successor = None    #后驱节点
    
    #反转以head为起点的n个节点，返回新的头结点
    def reverseN(self, head: ListNode, n: int) -> ListNode:
        if n == 1:
            #记录第n+1个节点
            self.successor = head.next
            return head
        # 以head.next为起点，需要反转前n-1个节点
        last = self.reverseN(head.next, n - 1)
        head.next.next = head
        #让反转之后的head节点和后面的节点连起来
        head.next = self.successor
        return last
    
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        #base case
        if m == 1:
            return self.reverseN(head, n)
        #前进到反转的起点触发 base case
        head.next = reverseBetween(head, m - 1, n - 1)
        return head
```

## 2. 递归思维：k个一组反转链表


```python
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# 反转以a为头结点的链表
def reverse(a: ListNode):
    pre = None
    cur = a
    nxt = a
    while cur:
        nxt = cur.next
        # 逐个节点反转
        cur.next = pre
        #更新指针位置
        pre = cur
        cur = nxt
    #返回反转后的头结点
    while pre

# 反转[a，b)之间的链表
def reverse(a: ListNode, b: ListNode):
    pre = None
    cur = a
    nxt = a
    #while 的终止条件更改一下就可以了
    while  cur != b:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    
    return pre

# k个一组反转链表
def reverseKGroup(head: ListNode, k: int):
    if head == None: return None
    a = head
    b = head
    for i in range(k):
        if b == None:
            return head
        b = b.next
    
    #反转前k个元素
    newHead = reverse(a, b)
    a.next = reverseKGroup(b, k)
    return newHead
```

## 3. 如何高效判断是否是回文链表


> 前提：经典面试题:最长回文子串
思路：
```python
for 0 <= i < len(s):
    找到以 s[i] 为中心的回文串
    找到以 s[i] 和 s[i+1] 为中心的回文串
    更新答案
```

具体代码实现：
```python
def palindrome(s: str, l: int, r: int):
    while(l >= 0 and r < len(s) and s[l] == s[r]):
        l -= 1
        r += 1
    return s[l + 1: r]

def longestPalindrome(s: str):
    res = ""
    for i in range(len(s)):
        s1 = palindrome(s, i, i)
        s2 = palindrome(s, i, i+1)
        if max(len(s1), len(s2)) >len(res):
            res = s1 if len(s1) > len(s2) else s2  
    return res
```

而判断回文单链表问题是：输入一个单链表的头结点，判断这个链表中的数字是不是回文  
这道题的难点在于，单链表无法倒着遍历，无法使用双指针技巧。  

其实，**借助二叉树后序遍历的思路，不需要显式反转原始链表也可以倒序遍历链表，下面来具体聊聊**  
对于二叉树的几种遍历方式：

```python
def traverse(root: TreeNode):
    #前序遍历代码
    traverse(root.left)
    #中序遍历代码
    traverse(root.righ)
    #后序遍历代码
```
对于链表也可以有前序遍历和后续遍历  
```python
def traverse(head: ListNode):
    #前序遍历代码
    traverse(head.next)
    #后序遍历代码
```
稍作修改就可以模仿双指针实现回文判断功能：  


```python
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def __init__(self):
        self.left = None
    def isPalindrome(head: ListNode) -> bool:
        self. left = head
        return traverse(head)
    def traverse(right: ListNode) -> bool:
        if right == None: return True
        res = traverse(right.next)
        res = res and (right.val == self.left.val)
        self.left = self.left.next
        return res
    
#优化空间复杂度
#1. 首先通过快慢指针找到链表中间点
slow = fast = head
while fast != None and fast.next!= None:
    fast = fast.next.next
    slow = slow.next
#2.如果fast没有指向null，说明链表长度为奇数，slow还要前进一步
if fast != None:
    slow = slow.next
#3.从slow开始反转后面的链表，现在就可以开始比较回文串了：
left = head
right = reverse(slow)

while right:
    if left.val != right.val:
        return False
    left = left.next
    right = right.next
return True
#其中的reverse函数：
def reverse(head: ListNode):
    pre = None
    cur = head
    nxt = head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    return pre
```

总结：  

寻找回文串是从中间往两边扩展，判断回文串是从两段向中间收缩。

对于单链表，无法直接倒序遍历，可以造一条新的反转链表，可以利用链表的后序遍历，也可以用栈的结构，倒序处理单链表。

具体到回文链表的判断，由于回文的特殊性，可以不完全反转链表，而是仅仅反转部分链表，将空间复杂度降到o(1)，不过要注意链表长度的奇偶。



# 2. 原地修改数组和双指针
## 1. 双指针技巧总结

### (1) 快慢指针的常见算法
#### a 判定链表中是否含有环     LeetCode 141题


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast :
                return True
        
        return False
```

#### b 已知链表中有环，返回这个环的起始位置
第一次相遇时，假设慢指针slow走了k步，那么快指针fast一定走了2k步  

fast一定比slow多走了k步，这多走的k步其实就是fast指针在环里转圈圈，所以k的值就是环长度的「整数倍」。

言归正传，设相遇点距环的起点的距离为m，那么环的起点距头结点head的距离为k - m，也就是说如果从head前进k - m步就能到达环起点。

所以，只要我们把快慢指针中的任一个重新指向head，然后两个指针同速前进，k - m步后就会相遇，相遇之处就是环的起点了。


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast :
                break
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow
```

#### c 寻找链表的中点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head: ListNode):
        slow = head
        fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        return slow
```

#### d 寻找链表的倒数第n个节点


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthNode(self, head: ListNode, n: int) :
        pre = ListNode(0, head)
        fast = head
        slow = pre
        for i in range(n):
            fast = fast.next
        # 如果此时快指针走到头，那么倒数第n个节点就是头结点
        # if fast == None:
        #     return head
        while fast:
            fast = fast.next
            slow = slow.next
        # slow.next 就是要删除的节点
        slow.next = slow.next.next
        return pre.next
```

### （2）左右指针的常用算法
#### a 二分查找
https://mp.weixin.qq.com/s/M1KfTfNlu4OCK8i9PSAmug


```python
def binarySearch(nums: list, target: int):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + (right - left)) // 2    # 与(left + right) / 2的结果一致，还可以有效防止相加溢出
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
    return -1
```

#### b 两数之和    LeetCode: 167题


```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        n = len(numbers)
        left, right = 0, n - 1
        while left < right:

            if numbers[left] + numbers[right] == target:
                return [left + 1, right + 1]
            if numbers[left] + numbers[right] > target:
                right -= 1
            if numbers[left] + numbers[right] < target: 
                left += 1
        return [-1, -1]
```

#### c 反转数组    LeetCode：344题


```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
```

#### d 滑动窗口算法
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

i. 最小覆盖子串  LeetCode：76题


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

ii. LeetCode：567题 字符串序列  
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

iii.LeetCode:3题 无重复字符最长子串  
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



## 2 Two Sum问题总结

### 1. 最基本的问题形式是：给你一个数组和一个整数target，可以保证数组中存在两个数的和为target，请你返回这两个数的索引。  


```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        other = 0
        index = {}
        ans = []
        # 构造一个哈希表，元素映射到对应索引
        for i in range(n):
            index[nums[i]] = i
        
        for i in range(n):
            other = target - nums[i]
            # 如果other存在且不是本身，则正确
            if other in index and index[other] != i:
                return list([i, index[other]])
        
        return list([-1, -1])
#这样，由于哈希表的查询时间为 O(1)，算法的时间复杂度降低到 O(N)，但是需要 O(N) 的空间复杂度来存储哈希表。
#不过综合来看，是要比暴力解法高效的
```

### 2. 稍微修改一下上面的问题，要求我们设计一个类，拥有两个 API：  

```python
class TwoSum:
    #向数据结构中添加一个数 number
    def add(number: int)
    #寻找当前数据结构中是否存在两个数的和为 value
    def find(value: int)

```


```python
class TwoSum:
    def __init__(self):
        self.freq = collections.defaultdict(int)
    def add(self, number):
        self.freq[number] += 1
    def find(self, value):
        for k in self.freq.keys():
            other = value
            if other == key and freq[k] > 1:
                return True
            if other != key and k in freq:
                return True
        return False
            
```

但是对于 API 的设计，是需要考虑现实情况的。比如说，我们设计的这个类，使用find方法非常频繁，那么每次都要 O(N) 的时间，岂不是很浪费费时间吗？对于这种情况，我们是否可以做些优化呢

是的，对于频繁使用find方法的场景，我们可以进行优化。我们可以参考上一道题目的暴力解法，借助哈希集合来针对性优化find方法：

```python
class TwoSum:
    def __init__(self):
        self.sum = set()
        self.nums = list()
    def add(self, number):
        #记录所有可能的和
        for n in self.nums:
            self.sum.add(n + number)
        self.nums.append(number)
    def find(self, value):
        return True if value in sum else False
```

对于 ``TwoSum`` 问题，一个难点就是给的数组无序。对于一个无序的数组，我们似乎什么技巧也没有，只能暴力穷举所有可能。

**一般情况下，我们会首先把数组排序再考虑双指针技巧。**``TwoSum ``启发我们，``HashMap ``或者 ``HashSet`` 也可以帮助我们处理无序数组相关的简单问题。


```python
def TwoSum(nums: list, target: int):
    nums.sort()
    left, right = 0, len(nums) - 1
    Sum = 0
    while left < right:
        Sum = nums[left] - nums[right]
        if Sum == target:
            return list([left, right])
        elif Sum > target:
            right -= 1
        elif Sum < target:
            left += 1
    return list([-1, -1])
        
```

## 3. 四道原地修改数组的算法题

### 1. 有序数组/链表去重问题
LeetCode：26题   有序数组去重


```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) == 0: return 0
        slow, fast = 0, 0
        while fast < len(nums):
            if nums[fast] != nums[slow]:
                slow += 1
                # 维护nums[0:slow]无重复
                nums[slow] = nums[fast]
            fast +=1  
        #无重复数组长度 slow + 1
        return slow + 1
```

LeetCode：83题   有序链表去重


```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None: return None
        slow = fast = head
        while fast:
            if fast.val != slow.val:
                # nums[slow] = nums[fast]
                slow.next = fast
                # slow += 1
                slow = slow.next
            # fast += 1
            fast = fast.next
        # 切断与后面重复元素的相连
        slow.next = None
        return head
```

### 2. 移除元素
原地移除数组中的元素，空间复杂度要是o(1)
LeetCode：27题


```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow, fast = 0, 0
        while fast < len(nums):
            if nums[fast] != val:
                # 注意这里与移除重复元素不一样
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
```

注意这里和有序数组去重的解法有一个重要不同，我们这里是先给``nums[slow]``赋值，  
然后再给slow+=1，这样可以保证``nums[0:slow-1]``是不包含值为val的元素的，最后的结果数组长度就是slow。

### 3. 移动0
LeetCode：283题 
照搬上题思路，相当于val=0。处理完之后在后面剩余的位置补0


```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums:
            return 0
        left, right = 0, 0
        while right < len(nums):
            if nums[right] != 0:
                nums[left] = nums[right]
                left += 1
            right += 1
        for i in range(left, len(nums)):
            nums[i] = 0

```

更简洁的写法，直接交换left、right位置数字，就可以把0移动至尾端


```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums:
            return 0
        left, right = 0, 0
        while right < len(nums):
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1

```

## 4. 给我 O(1) 时间，我能查找/删除数组中的任意元素
### 1. 实现随机集合
LeetCode：380题  O(1) 时间插入、删除和获取随机元素


```python
import random
class RandomizedSet:

    def __init__(self):
        # 储存元素的值
        self.nums = []
        # 记录每个元素对应在nums中的索引
        self.num2index = {}

    def insert(self, val: int) -> bool:
        # 若val已存在，不用再插入
        if val in self.num2index:
            return False
        # 若val不存在，插入到nums尾部，并记录对应索引值
        self.num2index[val] = len(self.nums)
        self.nums.append(val)
        return True


    def remove(self, val: int) -> bool:
        # 若val已存在，不用删除
        if val not in self.num2index:
            return False
        # 先拿到val索引
        index = self.num2index[val]
        # 将最后一个元素的索引改为index
        self.num2index[self.nums[-1]] = index
        # 交换val和最后一个元素的位置
        self.nums[index], self.nums[-1] = self.nums[-1], self.nums[index]
        # 删除最后一个元素
        self.nums.pop()
        # 删除字典中val的索引
        del self.num2index[val]
        return True


    def getRandom(self) -> int:
        #利用random.choice直接返回
        return random.choice(self.nums)
        # idx = random.randint(0, len(self.nums) - 1)
        # return self.nums[idx]


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

### 2. 避开黑名单中的随机数
LeetCode：710题

```python
import random
class Solution:

    def __init__(self, n: int, blacklist: List[int]):
        self.n = n
        self.blacklist = blacklist

        # 最终数组中的元素个数
        self.size = n - len(blacklist)
        
        self.mapping = {}
        # 把所有黑名单数字加入mapping
        for b in blacklist:
            self.mapping[b] = 666
        
        # 最后一个元素的索引
        self.last = n - 1
        for b in self.blacklist:
            # 如果b已经在数组尾端，可以直接忽略这个数
            if b >= self.size:
                continue
            while self.last in self.mapping:
                self.last -= 1
            self.mapping[b] = self.last
            self.last -= 1

        
 
    def pick(self) -> int:
        # 随机选取一个索引
        index = random.randint(0, self.size - 1)
        # 这个索引命中了黑名单，需要被映射到其他位置
        if index in self.mapping:
            return self.mapping[index]
        # 没有命中黑名单，直接返回
        return index



# Your Solution object will be instantiated and called as such:
# obj = Solution(n, blacklist)
# param_1 = obj.pick()
```


```python
class Solution:
    def __init__(self, N, blacklist):
        # 白名单的长度
        self.white_len = N - len(blacklist)
        # 把不在white_len位置内的数字映射到white_len内
        # black_lt ：在前white_len之内的黑名单中的数字
        # white_gt：在white_len之后的白名单中的数字
        black_lt = {i for i in blacklist if i < self.white_len}
        white_gt = {j for j in range(self.white_len, N)} - set(blacklist)
        self.map = dict(zip(black_lt, white_gt))
        
    def pick(self):
        res = random.randint(0, self.white_len - 1)
        if res in self.map:
            return self.map[res]
        else:
            return res
        
```

## 5 数组去重天花板难度
与上面题目不同，这道题没有一个全局的删除次数 k。而是对于每一个在字符串 s 中出现的字母 c 都有一个 k 值。这个 k 是 c 出现次数 - 1。

沿用上面的知识的话，我们首先要做的就是计算每一个字符的 k，可以用一个字典来描述这种关系，其中 key 为 字符 c，value 为其出现的次数。

具体算法：

- 建立一个字典。其中 key 为 字符 c，value 为其出现的剩余次数。
- 从左往右遍历字符串，每次遍历到一个字符，其剩余出现次数 - 1.
- 对于每一个字符，如果其对应的剩余出现次数大于 1，我们可以选择丢弃（也可以选择不丢弃），否则不可以丢弃。
- 是否丢弃的标准和上面题目类似。如果栈中相邻的元素字典序更大，那么我们选择丢弃相邻的栈中的元素。
还记得上面题目的边界条件么？如果栈中剩下的元素大于 n - k，我们选择截取前 n - k 个数字。然而本题中的 k 是分散在各个字符中的，因此这种思路不可行的。

不过不必担心。由于题目是要求只出现一次。我们可以在遍历的时候简单地判断其是否在栈上即可

时间复杂度：判断每个字符是否在栈上需要o(N)，总的为O(N^2)  
空间复杂度：用了额外的栈 O(N)  
**优化思路**  
查询给定字符是否在一个序列中存在的方法。根本上来说，有两种可能：

- 有序序列： 可以二分法，时间复杂度大致是 O(N)。
- 无序序列： 可以使用遍历的方式，最坏的情况下时间复杂度为 O(N)。我们也可以使用空间换时间的方式，使用 NN的空间 换取 O(1)的时间复杂度。  

由于本题中的 stack 并不是有序的，因此我们的优化点考虑空间换时间。而由于每种字符仅可以出现一次，这里使用 hashset 即可。


```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        seen = set()
        remain_counter = collections.Counter(s)

        for c in s:
            if c not in seen:
                while stack and c < stack[-1] and remain_counter[stack[-1]] > 0:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c)
            remain_counter[c] -= 1
        return ''.join(stack)
```



# 3. 数据结构

## 1 数据结构设计：队列实现栈，栈实现队列
### 队列实现栈
首先，队列的API如下


```python
class MyQueue:
    def __init__(self):
    # 添加元素到队尾
    def push(self, x):
    # 删除队头元素并返回
    def pop():
    #返回队头元素
    def peek():
    #判断队列是否为空
    def empty():   
```

<img src="xiaobaishuati.assets/image-20211012095709341.png" alt="image-20211012095709341" style="zoom:70%;" />


```python
class MyQueue:
    def __init__(self):
        self.s1 = []
        self.s2 = []
        
    # 添加元素到队尾
    def push(self, x):
        self.s1.append(x)
        
    # 删除队头元素并返回
    def pop():
        #先调用peek保证s2非空
        self.peek()
        return self.s2[-1]
        
    #返回队头元素
    def peek():
        if not self.s2:
            #把s1元素压入s2
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]
    
    #判断队列是否为空
    def empty():   
        return len(s1) == 0 and len(s2) == 0
```

## 2. 图文详解二叉堆、优先级队列
### 1. 二叉堆概述

因为，二叉堆其实就是一种特殊的二叉树（完全二叉树），只不过存储在数组里。一般的链表二叉树，我们操作节点的指针，而在数组里，我们把数组索引作为指针：

```python
#父节点的索引
def parent(root):
    return root / 2
#左孩子的索引
def left(root):
    return root * 2
#右孩子的索引
def right(root):
    return root * 2 + 1
```

![image-20211012100259590](xiaobaishuati.assets/image-20211012100259590.png)

你看到了，把 ``arr[1]`` 作为整棵树的根的话，每个节点的父节点和左右孩子的索引都可以通过简单的运算得到，这就是二叉堆设计的一个巧妙之处。为了方便讲解，下面都会画的图都是二叉树结构，相信你能把树和数组对应起来。

二叉堆还分为最大堆和最小堆。**最大堆的性质是：每个节点都大于等于它的两个子节点。**类似的，最小堆的性质是：每个节点都小于等于它的子节点。

### 2. 优先级队列概述

优先级队列这种数据结构有一个很有用的功能，**你插入或者删除元素的时候，元素会自动排序**，这底层的原理就是二叉堆的操作。

数据结构的功能无非增删查该，优先级队列有两个主要 API，分别是`insert`插入一个元素和`delMax`删除最大元素（如果底层用最小堆，那么就是`delMin`）。

下面我们实现了一个简化的优先级队列，代码框架如下：

```python
class MaxPQ:
    def __init__(self):
        self.pq = []    # 存储元素的数组
        self.N = 0    # 当前Priority Queue中元素个数
        
    def MaxPQ(self, cap):
        # 索引0不用，所以多分配一个空间
        pq = [None] * (cap + 1)
        
	# 返回当前队列中最大元素
    def max():
        return pq[1]
    
    # 插入元素e
    def insert(e):
        
    # 删除并返回当前队列中最大元素
    def delMax():
        
    # 上浮第 k 个元素，以维护最大堆性质
    def swim():
        
    # 下沉第 k 个元素，以维护最大堆性质
    def sink():
    
    # 交换数组的两个元素
    def exch(i, j):
        pq[i], pq[j] = pq[j], pq[i]
    
    # pq[i] 是否比 pq[j] 小
    def less(i, j):
        return True if pq[i] < pq else False
    # 还有left， right， parent 三个方法
```

### 3. 实现 swim 和 sink

对于最大堆，会破坏堆性质的有有两种情况：

1. 如果某个节点 A 比它的子节点（中的一个）小，那么 A 就不配做父节点，应该下去，下面那个更大的节点上来做父节点，这就是对 A 进行**下沉**。
2. 如果某个节点 A 比它的父节点大，那么 A 不应该做子节点，应该把父节点换下来，自己去做父节点，这就是对 A 的**上浮**。

```python
# 上浮代码的实现
def swim(k):
    # 如果浮到堆顶，就不能上浮了
    while k > 1 and self.less(parent(k), k):
		# 如果第 k 个元素比上层大，将 k 换上去
        self.exch(self.parent(k), k)
        k = self.parent(k)
```

![图片](xiaobaishuati.assets/6403.gif)

```python
# 下沉代码实现
def sink(k):
    # 如果沉到堆底，就不能沉了
    while (self.left(k) <= self.N):
        # 先假设左边节点大
        older = self.left(k)
        # 如果右边节点存在，比较大小
        if self.right(k) <= self.N and self.less(older, self.right(k)):
            older = self.right(k)
        # 节点k比older大，就比两个孩子都大，不必下沉
        if self.less(older, k): 
            break
        # 否则不符合最大堆结构，下沉k节点
        self.exch(k, older)
        k = older
```

![图片](xiaobaishuati.assets/6404.gif)

### 4. 实现 delMax 和 insert

这两个方法是建立在``swim``和``sink``上的

**`insert`方法先把要插入的元素添加到堆底的最后，然后让其上浮到正确位置。**

```python
def insert(e):
    self.N += 1
    self.pq[N] = e
	self.swim(N)
```

![图片](xiaobaishuati.assets/6401.gif)

**`delMax`方法先把堆顶元素 A 和堆底最后的元素 B 对调，然后删除 A，最后让 B 下沉到正确位置。**

```python
def delMax():
    # 最大堆顶就是最大元素
    max = self.pq[1]
    # 把这个最大元素换到最后，删除
    self.exch(1, N)
    self.pq[N] = None
    self.N -= 1
    # 把pq[1] 下沉到正确的位置
    self.sink(1)
    return max
```

## 3. 设计朋友圈时间线功能

LeetCode：355题

### 1. 题目及应用场景介绍

需要的API：

```python
class Twitter:

    def __init__(self):
	
    # user 发表一条tweet动态
    def postTweet(self, userId: int, tweetId: int) -> None:
	
    # 返回该user 关注的人（包括自己）最近的动态id，最多10条，而且动态按从新到旧排序
    def getNewsFeed(self, userId: int) -> List[int]:
	
    # follower 关注followee 如果id不存在则新建
    def follow(self, followerId: int, followeeId: int) -> None:

    #follower取关followee，如果id不存在则什么都不做
    def unfollow(self, followerId: int, followeeId: int) -> None:

# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
```

这几个 API 中大部分都很好实现，最核心的功能难点应该是 ``getNewsFeed``，因为返回的结果必须在时间上有序，但问题是用户的关注是动态变化的，怎么办？

**这里就涉及到算法了**：如果我们把每个用户各自的推文存储在链表里，每个链表节点存储文章 id 和一个时间戳 time（记录发帖时间以便比较），而且这个链表是按 time 有序的，那么如果某个用户关注了 k 个用户，我们就可以用合并 k 个有序链表的算法合并出有序的推文列表，正确地 ``getNewsFeed`` 了！

具体的算法等会讲解。不过，就算我们掌握了算法，应该如何编程表示用户 user 和推文动态 tweet 才能把算法流畅地用出来呢？**这就涉及简单的面向对象设计了**，下面我们来由浅入深，一步一步进行设计。

### 2. 面向对象设计

根据刚才的分析，我们需要一个 User 类，储存 user 信息，还需要一个 Tweet 类，储存推文信息，并且要作为链表的节点。

整体框架:

```python
class Twitter:
	class Tweet:
    class User:
        
    def __init__(self):
        self.timstemp = 0    # 时间戳
	
    # API 
    def postTweet(self, userId: int, tweetId: int) -> None:
    def getNewsFeed(self, userId: int) -> List[int]:
    def follow(self, followerId: int, followeeId: int) -> None:
    def unfollow(self, followerId: int, followeeId: int) -> None:
```

#### 1. Tweet类的实现

根据前面的分析，Tweet 类很容易实现：每个 Tweet 实例需要记录自己的 tweetId 和发表时间 time，而且作为链表节点，要有一个指向下一个节点的 next 指针。

```python
class Tweet:
    def __init__(self, id, time):
        self.id = id
        self.time = time
        self.next = Tweet()
```

<img src="xiaobaishuati.assets/6401.webp" alt="图片" style="zoom:50%;" />

#### 2. User类的实现

我们根据实际场景想一想，一个用户需要存储的信息有``userId``，关注列表，以及该用户发过的推文列表。其中关注列表应该用集合（``Hash Set``）这种数据结构来存，因为不能重复，而且需要快速查找；推文列表应该由链表这种数据结构储存，以便于进行有序合并的操作。画个图理解一下

<img src="xiaobaishuati.assets/6402.webp" alt="图片" style="zoom:50%;" />

除此之外，根据面向对象的设计原则，「关注」「取关」和「发文」应该是 User 的行为，况且关注列表和推文列表也存储在 User 类中，所以我们也应该给 User 添加 follow，unfollow 和 post 这几个方法：

```python
class User:
    def __init__(self, userId):
        self.id = userId
        self.followed = set()
        self.head = Tweet()    # 用户发表的推文的链表头结点
        self.follow(id)    # 关注一下自己
        
    def follow(userId):
        self.followed.add(userId)
    
    def unfollow(userId):
        # 不可以取关自己
        if userId != self.id:
            self.followed.remove(userId)
    
    def post(tweetId):
        twt = Tweet(tweetId, self.timestamp)
        self.timestamp += 1
        # 将新建的推文推入链表头
        twt.next = head
        head = twt
```

#### 3. 几个API方法的实现

```python
class Twitter:
	class Tweet:
        def __init__(self, id, time):
            self.id = id
            self.time = time
            self.next = Tweet()
            
    class User:
        def __init__(self, userId):
            self.id = userId
            self.followed = set()
            self.head = Tweet()    # 用户发表的推文的链表头结点
            self.follow(id)    # 关注一下自己

        def follow(userId):
            self.followed.add(userId)

        def unfollow(userId):
            # 不可以取关自己
            if userId != self.id:
                self.followed.remove(userId)

        def post(tweetId):
            twt = Tweet(tweetId, self.timestamp)
            self.timestamp += 1
            # 将新建的推文推入链表头
            twt.next = head
            head = twt

    def __init__(self):
        self.timstemp = 0    # 时间戳
        self.userMap = {}    # 需要一个映射将 userId 和 User 对象对应起来
	
     # user 发表一条tweet动态
    def postTweet(self, userId: int, tweetId: int) -> None:
        # 若userId不存在，则新建
        if userId not in self.userMap:
            self.userMap[userId] = User(userId)
        u = self.userMap[userId]
        u.post(tweetId)
	
    # follower 关注followee 如果id不存在则新建
    def follow(self, followerId: int, followeeId: int) -> None:
        # 若 follower 不存在，则新建
        if followerId not in self.userMap:
            u = User(followerId)
            self.userMap[followerId] = u
        # 若 followee 不存在，则新建
        if followeeId not in self.userMap:
            u = User(followeeId)
            self.userMap[followeeId] = u
        self.userMap[followerId].follow(followeeId)

    #follower取关followee，如果id不存在则什么都不做
    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId in self.userMap:
            flwer = self.userMap[followerId]
            flwer.unfollow(followeeId)
        
    # 返回该user 关注的人（包括自己）最近的动态id，最多10条，而且动态按从新到旧排序
    def getNewsFeed(self, userId: int) -> List[int]:
		# 需要算法设计
```

### 3. 算法设计

实现合并 k 个有序链表的算法需要用到优先级队列（Priority Queue），这种数据结构是「二叉堆」最重要的应用。

借助这种牛逼的数据结构支持，我们就很容易实现这个核心功能了。注意我们把优先级队列设为按 time 属性**从大到小降序排列**，因为 time 越大意味着时间越近，应该排在前面：

```python
def getNewsFeed(self, userId: int) -> List[int]:
    res = []
    if userId not in self.userMap:
        return res
    # 关注列表的用户id
    users = self.userMap[userId].followed
    # 自动通过time属性从大到小排序，容量为users的大小
    # pq = PriorityQueue(len(users))    # 还不知道该怎么修改
    
    # 现将所有链表头结点插入优先级队列
    for id in users:
        twt = userMap[id].head
        if twt == None:
            continue
        pq.add(twt)
    
    while (pq):
        # 最多返回10条就够
        if len(res) == 10:
            return break
        # 弹出 time 最大的
        twt = pq[-1]
        res.append(twt.id)
        # 将下一篇tweet插入进行排序
        if twt.next:
            pq.append(twt.next)
    return res
```

![图片](xiaobaishuati.assets/640.gif)

**针对LeetCode355题的python解法：很好的面向对象解决方案**

将推特用户抽象为User对象，所有动作的细节放进User对象中。
因此Twitter对象作为一个系统，只需要对User发号施令就行了。

```python
class User:

    def __init__(self, userId):
        self.userId = userId
        self.follows = []  # 用户的关注对象
        self.contents = []  # 存放该用户发的所有推特内容，推特内容形式为：（时间， 推特ID）

    def post(self, *tweet):
        self.contents.append(tweet)

    def get(self, users):
        recommends = []  # 将关注对象的推特和自己发的推特都存进去
        for i in self.follows:
            recommends.extend(users[i].contents)
        recommends.extend(self.contents)
        recommends = sorted(recommends, key=lambda x: x[0])  # 对推荐列表按时间排序
        return [i[1] for i in recommends[: 10]]  # 返回最新的10条推特

    def follow(self, followeeId):
        if followeeId != self.userId and followeeId not in self.follows:  # 验证
            self.follows.append(followeeId)
    
    def unfollow(self, followeeId):
        if followeeId != self.userId and followeeId in self.follows:  # 验证
            self.follows.remove(followeeId)


class Twitter:

    def __init__(self):
        self.users = {}  # 推特用户池
        self.time = 0  # 系统时间

    def valid_users(self, *ids):  # 验证id是否在用户池中，如果不在，注册一个！
        for i in ids:
            if i not in self.users:
                self.users[i] = User(i)   

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.valid_users(userId)  # 验证
        self.users[userId].post(self.time, tweetId)  # 推特内容形式为：（时间， 推特ID）
        self.time -= 1

    def getNewsFeed(self, userId: int) -> List[int]:
        self.valid_users(userId)  # 验证
        return self.users[userId].get(self.users)  # 从推特用户池中找到userID用户，发出get命令

    def follow(self, followerId: int, followeeId: int) -> None:
        self.valid_users(followerId, followeeId)  # 验证
        self.users[followerId].follow(followeeId)  # 从推特用户池中找到userID用户，发出follow命令
    
    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.valid_users(followerId, followeeId)  # 验证
        self.users[followerId].unfollow(followeeId)  # 从推特用户池中找到userID用户，发出unfollow命令
```

## 4. 单调栈结构

### 1. 单调栈模板

LeetCode：496题 下一个更大元素

- 当前项向右找第一个比自己大的位置 —— 从右向左维护一个单调递减栈

```python
class Solution2:
    def nextGreaterElement(self, nums1, nums2):
        dic, stack = {}, []

        for i in range(len(nums2) - 1, -1, -1):
            while stack and stack[-1] <= nums2[i]:
                stack.pop()
            if stack: dic[nums2[i]] = stack[-1]
            stack.append(nums2[i])

        return [dic.get(x, -1) for x in nums1]
```

- 从左到右维护单调递减栈， 找元素右侧区域，第一个比自己大的位置

```python
class Solution3:
    def nextGreaterElement(self, nums1, nums2):
        dic, stack = {}, []

        for i in range(len(nums2)):
            while stack and stack[-1] < nums2[i]:
                dic[stack.pop()] = nums2[i]
            stack.append(nums2[i])

        return [dic.get(x, -1) for x in nums1]
```

**单调栈总结：**

- 单调递增栈：从栈底到栈顶递增，栈顶大

- 单调递减栈：从栈底到栈顶递减，栈顶小

1. **什么时候用单调栈？**

   通常是一维数组，需要寻找任意元素右边（左边）第一个比自己大（小） 的元素，且要求时间复杂度o(n)

2. **模板套路**

   1） 当前位置向右找第一个比自己大的位置，从右往左维护单调递减栈：

   ```python
   def nextGreaterElement_01(nums:list):
       length = len(nums)
       res, stack = [-1] * length, []
       
       for i in range(length - 1, -1, -1):
           while stack and stack[-1] <= nums[i]:
               stack.pop()
           if stack :
               res[i] = stack[-1]
           stack.append(nums[i])
       return res
   ```

   从左往右维护：

   ```python
   def nextGreaterElement_01(nums:list):
       length = len(nums)
       res, stack = [-1] * length, []
       
       for i in range(length):
           while stack and nums[stack[-1]] < nums[i]:
               idx = stack.pop()
               res[idx] = nums[i]
           stack.append(i)
       return res
   ```

   2）当前位置向右找第一个比自己小的位置，从右往左维护单调递增栈

   ```python
   def nextGreaterElement_01(nums:list):
       length = len(nums)
       res, stack = [-1] * length, []
       
       for i in range(length - 1, -1, -1):
           while stack and stack[-1] >= nums[i]:
               stack.pop()
           if stack:
               res[i] = stack[-1]
           stack.append(nums[i])
       return res
   ```
   从左往右

   ```python
   def nextGreaterElement_01(nums:list):
       length = len(nums)
       res, stack = [-1] * length, []
       
       for i in range(length):
           while stack and nums[stack[-1]] > nums[i]:
               idx = stack.pop()
               res[idx] = nums[i]
           stack.append(i)
       return res
   ```

   3）当前位置向左找第一个比自己大的元素，从左往右维护单调递减栈
   
   ```python
   def nextGreaterElement_01(nums:list):
       length = len(nums)
       res, stack = [-1] * length, []
       
       for i in range(length):
           while stack and stack[-1] < nums[i]:
               stack.pop()
           if stack:
               res[i] = nums[i]
           stack.append(nums[i])
       return res
   ```
   
   4）当前位置向左找第一个比自己小的元素，从左往右维护单调递增栈：
   
   ```python
   def nextGreaterElement_01(nums:list):
       length = len(nums)
       res, stack = [-1] * length, []
       
       for i in range(length):
           while stack and stack[-1] >= nums[i]:
               stack.pop()
           if stack:
               res[i] = nums[i]
           stack.append(nums[i])
       return res
   ```
   
### 2. 问题变形

   LeetCode：1118题，第一个比自己大的元素，距离自己有多远

   ```python
   def dailyTemperatures(T: list):
       length = len(T)
       res, stack = [-1] * length, []
   
       for i in range(length - 1, -1, -1):
           while stack and T[stack[-1]] <= T[i]:
               stack.pop()
           if stack:
               res[i] = stack[-1] - i
           stack.append(i)
       return res
   ```

  ### 3. 环形数组时怎么解决

LeetCode：503题

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res, stack = [-1] * n, []
        # 假设这个数组长度翻倍了
        for i in range(2*n-1, -1, -1):
            while stack and stack[-1] <= nums[i % n]:
                stack.pop()
            if stack:
                res[i % n] = stack[-1]
            stack.append(nums[i % n])
        return res
```

## 5. 单调队列

LeetCode：239题

```python
# 单调队列的实现 从大到小
class MonotonicQueue:
    def __init__(self):
        self.q = []    # 用list实现单调队列

    # 如果push的数值大于入口的数值，那么就将队列后端的数值弹出，知道push的值小于队列入口元素的值
    # 这样保证队列里的数值是单调从大到小的
    def push(self, n: int):
        while self.q and self.q[-1] < n:
            self.q.pop()
        self.q.append(n)
    
    # 查询队列里的最大值，直接返回队列的头就可以
    def max(self):
        return self.q[0]
    
    # 每次弹出是，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出，同时要判断队列是否为空
    def pop(self, n):
        if self.q and n == self.q[0]:
            self.q.pop(0)

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        window = MonotonicQueue()
        res = []

        for i in range(len(nums)):
            if i < k - 1:
                window.push(nums[i])    # 先填满窗口前 k - 1
            else:
                window.push(nums[i])    # 窗口向前滑动，加入新数字
                res.append(window.max())    # 记录当前窗口最大值
                window.pop(nums[i - k + 1])     # 移除旧数字
        return res
```



# 4. 二叉树

## 二叉树第一期

### 1. 二叉树的重要性

举个例子，比如说我们的经典算法「快速排序」和「归并排序」，对于这两个算法，你有什么理解？**如果你告诉我，快速排序就是个二叉树的前序遍历，归并排序就是个二叉树的后续遍历，那么我就知道你是个算法高手了**。

```c++
void sort(int[] nums, int lo, int hi) {
    /****** 前序遍历位置 ******/
    // 通过交换元素构建分界点 p
    int p = partition(nums, lo, hi);
    /************************/

    sort(nums, lo, p - 1);
    sort(nums, p + 1, hi);
}
```

先构造分界点，然后去左右子数组构造分界点，你看这不就是一个二叉树的前序遍历吗？

```c++
void sort(int[] nums, int lo, int hi) {
    int mid = (lo + hi) / 2;
    sort(nums, lo, mid);
    sort(nums, mid + 1, hi);

    /****** 后序遍历位置 ******/
    // 合并两个排好序的子数组
    merge(nums, lo, mid, hi);
    /************************/
}
```

先对左右子数组排序，然后合并（类似合并有序链表的逻辑），你看这是不是二叉树的后序遍历框架？另外，这不就是传说中的分治算法嘛，不过如此呀。

### 2 写递归算法的秘诀

**写递归算法的关键是要明确函数的「定义」是什么，然后相信这个定义，利用这个定义推导最终结果，绝不要试图跳入递归**。

比如说让你计算一棵二叉树共有几个节点：

```python
# 定义：count(root) 返回以 root 为根的树有多少节点
def count(root) :
    # base case
    if (root == None):
        return 0
    # 自己加上子树的节点数就是整棵树的节点数
    return 1 + count(root.left) + count(root.right)
```

**写树相关的算法，简单说就是，先搞清楚当前`root`节点该做什么，然后根据函数定义递归调用子节点**，递归调用会让孩子节点做相同的事情。

### 3 算法实践

#### 1. LeetCode：226题 反转二叉树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        # base case
        if root == None:
            return None
        # 每个节点会干的事情
        root.left, root.right = root.right, root.left
        # 左右节点也反转
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

值得一提的是，如果把交换左右子节点的代码放在后序遍历的位置也是可以的，但是放在中序遍历的位置是不行的，请你想一想为什么？

> 中序遍历换节点 根据左根右的遍历顺序 相当于左侧节点交换了两次 右侧节点没换  因为遍历根的时候交换了左右节点 遍历右侧的时候还是之前那个左节点

首先讲这道题目是想告诉你，**二叉树题目的一个难点就是，如何把题目的要求细化成每个节点需要做的事情**。

#### 2. LeetCode：116题 **填充二叉树节点的右侧指针**

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root == None:
            return None
        self.connectTwoNode(root.left, root.right)
        return root

    # 将传入的两个节点连接
    def connectTwoNode(self, node1, node2):
        if node1 == None or node2 == None:
            return None
        # 主要实现步骤
        node1.next = node2

        # 连接相同父节点的两个节点
        self.connectTwoNode(node1.left, node1.right)
        self.connectTwoNode(node2.left, node2.right)
        # 连接跨父节点的两个节点
        self.connectTwoNode(node1.right, node2.left)
```

回想刚才说的，**二叉树的问题难点在于，如何把题目的要求细化成每个节点需要做的事情**，但是如果只依赖一个节点的话，肯定是没办法连接「跨父节点」的两个相邻节点的。

那么，我们的做法就是增加函数参数，一个节点做不到，我们就给他安排两个节点，「将每一层二叉树节点连接起来」可以细化成「将每两个相邻节点都连接起来」

#### 3. LeetCode：114题 **将二叉树展开为链表**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        # base case 
        if root == None:
            return None
        
        # 1. 先把左右两个字数捋直
        self.flatten(root.left)
        self.flatten(root.right)
		
        # 后序遍历位置
        # 2. 把左子树接到右边
        left = root.left
        right = root.right
        root.left = None
        root.right = left

        # 3. 把原来的右子树接到现在右子树后
        p = root
        while p.right != None:
            p = p.right
        p.right = right
        return root
```

我们尝试给出这个函数的定义：

**给`flatten`函数输入一个节点`root`，那么以`root`为根的二叉树就会被拉平为一条链表**。

我们再梳理一下，如何按题目要求把一棵树拉平成一条链表？很简单，以下流程：

1、将`root`的左子树和右子树拉平。

2、将`root`的右子树接到左子树下方，然后将整个左子树作为右子树。

你看，这就是递归的魅力，你说`flatten`函数是怎么把左右子树拉平的？不容易说清楚，**但是只要知道`flatten`的定义如此，相信这个定义，让`root`做它该做的事情，然后`flatten`函数就会按照定义工作。**

另外注意递归框架是后序遍历，因为我们要先拉平左右子树才能进行后续操作。

**二叉树题目的难点在于如何通过题目的要求思考出每一个节点需要做什么，这个只能通过多刷题进行练习了。**

## 二叉树第二期

### 1. 构造最大二叉树

LeetCode：654题 构造最大二叉树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        
        if not nums:
            return 
        n = len(nums)
        if n == 1:
            return TreeNode(nums[0])

        maxIndex = nums.index(max(nums))
        root = TreeNode(nums[maxIndex])
        root.left = self.constructMaximumBinaryTree(nums[:maxIndex])
        root.right = self.constructMaximumBinaryTree(nums[maxIndex+1:])
        return root
```

**对于每个根节点，只需要找到当前`nums`中的最大值和对应的索引，然后递归调用左右数组构造左右子树即可**。

### 2. 从前序遍历和中序遍历结果中构造二叉树

LeetCode：105题

**类似上一题，我们肯定要想办法确定根节点的值，把根节点做出来，然后递归构造左右子树即可**。

我们先来回顾一下，前序遍历和中序遍历的结果有什么特点？

前文 [二叉树就那几个框架](http://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485871&idx=1&sn=bcb24ea8927995b585629a8b9caeed01&chksm=9bd7f7a7aca07eb1b4c330382a4e0b916ef5a82ca48db28908ab16563e28a376b5ca6805bec2&scene=21#wechat_redirect) 写过，这样的遍历顺序差异，导致了`preorder`和`inorder`数组中的元素分布有如下特点：

<img src="xiaobaishuati.assets/6406.webp" style="zoom:67%;" />

**关键之处在于左子树和右子树的起止索引：**

但是其实按照对应关系可以知道：

左子树的起止索引：``preorder[1: index + 1]   inorder[:index]``

右子树的起止索引：``preorder[index + 1: ]   inorder[index + 1:]``

<img src="xiaobaishuati.assets/6403.webp" style="zoom:67%;" />

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder and not inorder:
            return
        # 构造根节点
        root = TreeNode(preorder[0])
        # 根节点在中序遍历数字中的索引
        index = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:index + 1], inorder[:index])
        root.right = self.buildTree(preorder[index + 1:], inorder[index + 1:])
        return root
```

### 3. 从中序与后序遍历中构造二叉树

方法同上题，只不过略有改动

<img src="xiaobaishuati.assets/640.webp" alt="图片" style="zoom:67%;" />

## 二叉树序列化与反序列化

LeetCode：297题

### 1. 先序遍历

二叉树先序遍历（迭代法）：

- 根节点入栈
- 弹出栈顶元素，加入结果
- 右子树，左子树，根节点依次入栈

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return None
        stack, res = [root], []
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
                
        return res
```



```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        def ser(root):
            if not root:
                return res.append('null')
            else:
                res.append(str(root.val))
                ser(root.left)
                ser(root.right)
        ser(root)
        res = ','.join(res)
        return res

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return []
        data_s = data.split(',')

        def deser(nodes):
            val = nodes[0]
            del nodes[0]
            if val == 'null':
                return None
            val = int(val)
            root = TreeNode(val)
            root.left = deser(nodes)
            root.right = deser(nodes)
            return root
        root = deser(data_s)
        return root
```

**ATTENTION:**

**序列化时使用列表先暂时储存每一个字符串的值，最后拼接起来，可以防止最后多一位``','``**

**反序列化时，取出第一个节点后，要删除它，这样就可以直接把剩下的列表直接传到递归函数**

### 2. 后序遍历

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        def ser(root):
            if not root:
                return res.append('null')
            else:
                ser(root.left)
                ser(root.right)
                res.append(str(root.val))
        ser(root)
        res = ','.join(res)
        return res
    

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return []
        data_s = data.split(',')

        def deser(nodes):
            val = nodes[-1]
            del nodes[-1]
            if val == 'null':
                return None
            val = int(val)
            root = TreeNode(val)
            root.right = deser(nodes)
            root.left = deser(nodes)
            return root
        root = deser(data_s)
        return root
```

**ATTENTION:**

**反序列化时，先从右子树开始，因为节点按左右根顺序排列**

### 3. 层序遍历

- 先熟悉层序遍历：LeetCode：102题 二叉树层序遍历代码

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        # 根节点入queue
        queue = [root]
        res = []
        while queue:
            res.append([node.val for node in queue])
            # 当前层孩子节点的列表
            childs = []
            # 对当前层每个节点遍历
            for node in queue:
                if node.left:
                    childs.append(node.left)
                if node.right:
                    childs.append(node.right)
            queue = childs
            
        return res
```

- LeetCode：107题 二叉树层序遍历代码2 从底层往上层序遍历

只要将结果改为``return res[::-1]``即可

- LeetCode：199题 二叉树层序遍历的右视图

将``res.append([node.val for node in queue])``改为``res.append([node.val for node in queue][-1])``即可

回到297题，代码如下：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root: return ''
        # 类似于层序遍历，但空节点也要保存
        queue = collections.deque()
        queue.append(root)
        res = []
        while queue:
            # 每次拿一个节点出来操作
            node = queue.popleft()
            # node 为空，用null表示
            if not node:
                res.append('null')
            # 否则自身加入res，左右节点加入双端队列
            else:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)

        return ','.join(res)       

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return []
        data_s = data.split(',')
        # 用指针表示当前进行到的节点
        idx = 1
        root = TreeNode(int(data_s[0]))
        queue = collections.deque()
        queue.append(root)
        while queue:
            # 每次弹出一个节点，对其加入左右孩子
            node = queue.popleft()
            if data_s[idx] != 'null':
                node.left = TreeNode(int(data_s[idx]))
                queue.append(node.left)
            idx += 1
            if data_s[idx] != 'null':
                node.right = TreeNode(int(data_s[idx]))
                queue.append(node.right)
            idx += 1

        return root
```

## 二叉树第三期

LeetCode：652题 寻找重复子树

对于这道题，**你需要知道以下两点**：

**1、以我为根的这棵二叉树（子树）长啥样**？

**2、以其他节点为根的子树都长啥样**？

首先第一个问题：以我为根的这颗二叉树长啥样？

​	其实看到这个问题，就可以判断本题要使用「后序遍历」框架来解决：为什么？很简单呀，我要知道以自己为根的子树长啥样，是不是得先知道我的左右子树长啥样，再加上自己，就构成了整棵子树的样子？

```python
def dfs(root):
    # 对于空节点，用'#'表示
    if not root:
        return '#'
    # 将左右子树序列化成字符串
    left = dfs(root.left)
    right = dfs(root.right)
    # 左右子树加上自己，就是以自己为根的二叉树序列化结果
    subTree = left + ',' + right + ',' + str(root.val)
    return subTree
```

接下来第二个问题：怎么知道其他的子树什么样。

​	放到字典里，key是子树序列，value是出现次数。当``value == 1``时将key加入结果，大于1也不用管。

全部代码如下：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        mem = {}
        res = []
        def dfs(root):
            if not root:
                return '#'
            left = dfs(root.left)
            right = dfs(root.right)
            subTree = left + ',' + 'right' + ',' + str(root.val)
            freq = mem.get(subTree, 0)
            if freq == 1:
                res.append(root)
            mem[subTree] = freq + 1
            return subTree
       	dfs(root)
        return res
```

# 5. 二叉搜索树

## 1. 二叉搜索树第一期

### 1. 寻找第k小的元素

LeetCode：230题

直接中序遍历就可以，返回``res[k - 1]``即可。不需要像C++那样设置个rank

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        if not root:
            return  None
        res = []

        def dfs(root):
            if not root:
                return
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)
        dfs(root)
        return res[k - 1]
```

那么回到这个问题，想找到第`k`小的元素，或者说找到排名为`k`的元素，如果想达到对数级复杂度，关键也在于每个节点得知道他自己排第几。

比如说你让我查找排名为`k`的元素，当前节点知道自己排名第`m`，那么我可以比较`m`和`k`的大小：

1、如果`m == k`，显然就是找到了第`k`个元素，返回当前节点就行了。

2、如果`k < m`，那说明排名第`k`的元素在左子树，所以可以去左子树搜索第`k`个元素。

3、如果`k > m`，那说明排名第`k`的元素在右子树，所以可以去右子树搜索第`k - m - 1`个元素。

这样就可以将时间复杂度降到`O(logN)`了

那么，如何让每一个节点知道自己的排名呢？

这就是我们之前说的，需要在二叉树节点中维护额外信息。**每个节点需要记录，以自己为根的这棵二叉树有多少个节点**。

### 2. 二叉搜索树转变为累加树

LeetCode：53题

对于这道题来说，二叉树的通用思路在这里用不了。即：先思考每个节点该做什么剩下的交给递归函数。

对于一个节点来说，确实右子树都是比它大的元素，但问题是它的父节点也可能是比它大的元素，我们没有到达父节点的指针，所以不能这么用。

但是刚刚我们说了，利用中序遍历的特性，可以直接升序打印BST：

```python
def dfs(root):
    if not root:
        return None
    dfs(root.left)
    print(root.val)
    dfs(root.right)
```

那么我们想降序打印BST怎么办？很简单，改一下递归顺序就可：先遍历右子树，就是降序打印

```python
def dfs(root):
    if not root:
        return None
    dfs(root.right)
    print(root.val)
    dfs(root.left)
```

那么我们可以维护一个外部累加变量sum，把sum的值赋给每一个节点就可以了

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        sum = 0
        def dfs(root):
            if not root:
                return None
            dfs(root.right)
            nonlocal sum
            sum += root.val
            root.val = sum
            dfs(root.left)
        dfs(root)
        return root
```

## 2. 二叉搜索树第二期

本文来实现 BST 的基础操作：判断 BST 的合法性、增、删、查。其中「删」和「判断合法性」略微复杂。

### 1. 判断BST的合法性

这里是有坑的，如果按照上文的思路，判断每个节点的左子树和右子树跟自己的大小关系，代码如下：

```python
def isValidBST(root):
    if not root:
        return True
    if root.left and root.left.val >= root.val: return False
    if root.right and root.right.val <= root.val: return False
    return isValidBST(root.left) and isValidBST(root.right)
```

但是这个算法出现了错误，BST 的每个节点应该要小于右边子树的所有节点，下面这个二叉树显然不是 BST，因为节点 10 的右子树中有一个节点 6，但是我们的算法会把它判定为合法 BST：

![](xiaobaishuati.assets/6404.webp)

**出现问题的原因在于，对于每一个节点`root`，代码值检查了它的左右孩子节点是否符合左小右大的原则；但是根据 BST 的定义，`root`的整个左子树都要小于`root.val`，整个右子树都要大于`root.val`**。

 正确的代码：

```python
def isValidBST(root, min, max):
    if not root:
        return True
    # 若 root.val 不符合 max 和 min 的限制，说明不是合法 BST
    if min and root.val <= min.val: return False
    if max and root.val >= max.val: return False
    # 限定左子树的最大值是 root.val，右子树的最小值是 root.val
    return isValidBST(root.left, min, root) and isValidBST(root.right, root, max)
```

**我们通过使用辅助函数，增加函数参数列表，在参数中携带额外信息，将这种约束传递给子树的所有节点，这也是二叉树算法的一个小技巧吧**。

实例：LeetCode：98题 验证二叉搜索树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def isBST(root, min, max):
            if not root:
                return True
            if min and root.val <= min.val: return False
            if max and root.val >= max.val: return False
            return isBST(root.left, min, root) and isBST(root.right, root, max)
        return isBST(root, None, None)
```

### 2. 搜索一个数 & 针对 BST 的遍历框架

我们在普通的二叉树中搜索一个数，可以这么写：

```python
def search(root, target):
    if not root:
        return False
    if root.val == target:
        return True
    # 当前节点没找到就递归地去左右子树寻找
    return search(root.left, target) or search(root.right, target)
```

但是在BST中，可以使用二分搜索

```python
def isInBST(root, target):
    if not root:
        return False
    if root.val == target:
        return True
    elif root.val < target:
        return isInBST(root.right, target)
    elif root.val > target:
        return isInBST(root.left, target)
```

这样我们就抽象出一套**针对 BST 的遍历框架**：

```python
def BST(root, target):
    # base case
    if not root:
    	# do something
    if root.val == target:
        # find the target, do something
    elif root.val < target:
        BST(root.right, target)
    elif root.val > target:
        BST(root.left, target)
```

使用这个框架解决LeetCode：700题 二叉搜索树中的搜索

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return None
        if root.val == val:
            return root
        if root.val > val:
            return self.searchBST(root.left, val)
        if root.val < val:
            return self.searchBST(root.right, val)
```

### 3. BST中插入一个数

上一个问题，我们总结了 BST 中的遍历框架，就是「找」的问题。直接套框架，加上「改」的操作即可。**一旦涉及「改」，函数就要返回`TreeNode`类型，并且对递归调用的返回值进行接收**。

实例：LeetCode：701 二叉搜索树中的插入操作

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        # 如果当前节点为空，也就意味着val找到了合适的位置，此时创建节点直接返回。
        if not root:
            return TreeNode(val)  
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)    # 递归创建左子树
        elif root.val > val:
            root.left = self.insertIntoBST(root.left, val)     # 递归创建右子树
        return root
```

### 4. BST中删除一个数

跟插入操作类似，先找，然后删代码框架如下：

```python
def delBST(root, val):
    if root.val == val:
        # find the val, delete it
    if root.val < val:
        root.right = delBST(root.right, val)
    if root.val > val:
        root.left = delBST(root.left, val)
    return root
```

**找到的节点A可能有三种情况，不能直接删除！**

**情况 1：**`A`恰好是末端节点，两个子节点都为空，那么它可以当场去世了。


```python
if not root.left and not root.right:
    return None
```

**情况 2**：`A`只有一个非空子节点，那么它要让这个孩子接替自己的位置。

<img src="xiaobaishuati.assets/6407.webp" style="zoom:67%;" />

```python
if not root.left: return root.right
if not root.right: return root.left
```

**情况 3**：`A`有两个子节点，麻烦了，为了不破坏 BST 的性质，`A`必须找到左子树中最大的那个节点，或者右子树中最小的那个节点来接替自己。我们以第二种方式讲解。

![](xiaobaishuati.assets/6408.webp)

```python
if root.left and root.right:
    # 找到右子树的最小节点
    midNode = getMin(root.right)
    # 把root改成minNode
    root.val = minNode.val
    # 转而删除minNode
    root.right = deleteNode(root.right, minNode.val)
```

实例：LeetCode：450 删除二叉树中的一个节点

完整代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:

        def getMin(node):
            # BST 最左边的就是最小的
            while node.left:
                node = node.left
            return node

        if not root:
            return None

        if root.val == key:
            # 这两个 if 把情况 1 和 2 都正确处理了
            if not root.left: return root.right
            if not root.right: return root.left
            # 处理情况3
            minNode = getMin(root.right)
            root.val = minNode.val
            root.right = self.deleteNode(root.right, minNode.val)

        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        return root
```

**总结**:

通过这篇文章，我们总结出了如下几个技巧：

1、如果当前节点会对下面的子节点有整体影响，可以通过辅助函数增长参数列表，借助参数传递信息。

2、在二叉树递归框架之上，扩展出一套 BST 代码框架：

```python
def BST(root, target):
    if root.val == target:
        # do something
    if root.val > target:
        BST(root.left, target)
    if root.val < target:
        BST(root.right, target)
```

## 3. 二叉搜索树第三期

### 1. 二叉搜索树的数量

LeetCode：96题：给你输入一个正整数`n`，请你计算，存储`{1,2,3...,n}`这些值共有有多少种不同的 BST 结构。

这就是一个正宗的**穷举问题**，那么什么方式能够正确地穷举合法 BST 的数量呢？

我们前文说过，不要小看「穷举」，这是一件看起来简单但是比较有技术含量的事情，问题的关键就是不能数漏，也不能数多，你咋整？

举个例子，比如给算法输入`n = 5`，也就是说用`{1,2,3,4,5}`这些数字去构造 BST。

- 首先，这棵 BST 的根节点总共有几种情况？显然有 5 种情况对吧，**因为每个数字都可以作为根节点**。

- 比如说我们固定`3`作为根节点，这个前提下能有几种不同的 BST 呢？

  根据 BST 的特性，根节点的左子树都比根节点的值小，右子树的值都比根节点的值大。

  所以如果固定`3`作为根节点，左子树节点就是`{1,2}`的组合，右子树就是`{4,5}`的组合。

  **左子树的组合数和右子树的组合数乘积**就是`3`作为根节点时的 BST 个数。也就是4种

我们可以写这样一个函数：

```python
# count(l,r) 可以计算出[l, r]之间存在的BST数量
def count(l, r):
```

然后为了消除重叠子的情况，使用备忘录法``mem={}``中存入(l, r)之间的BST数量，之后可以直接用。

完整代码如下：

```python
class Solution:
    def numTrees(self, n: int) -> int:
        mem = collections.defaultdict()
        # count计算[l,r]之间BST数量
        def count(l, r):
            # base case
            if l > r: return 1
            if mem.get((l,r), None):
                return mem[(l, r)]
            res = 0
            for i in range(l, r + 1):
                # i 的值作为root节点
                left = count(l, i - 1)
                right = count(i + 1, r)
                res += left * right
            mem[(l,r)] = res
            return res
        return count(1, n)
```

### 2. 构造出所有可能的BST

LeetCode：95题：构建出所有合法的 BST

**明白了上道题构造合法 BST 的方法，这道题的思路也是一样的**：

1、穷举`root`节点的所有可能。

2、递归构造出左右子树的所有合法 BST。

3、给`root`节点穷举所有左右子树的组合。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0: return []
        
        # build(l, r) 可以返回(l,r)之间所有可能的BST
        def build(l, r):
            res = []    # 储存结果
            # base case
            if l > r:
                return [None]
            # 1. 穷举root节点的所有可能
            for i in range(l, r + 1):
                # 2. 递归构造左右子树的合法BST
                leftTree = build(l, i - 1)
                rightTree = build(i + 1, r)
                # 3. 穷举root节点左右可能的左右子树组合
                for left in leftTree:
                    for right in rightTree:
                        root = TreeNode(i)
                        root.left = left
                        root.right = right
                        res.append(root)
            return res
        return build(1, n)
```

## 美团面试：二叉搜索树后序遍历

LeetCode：1373题 二叉搜索子树的最大键值和

**那么我们想计算子树中 BST 的最大和，站在当前节点的视角，需要做什么呢**？

1. 肯定得知道左右子树是不是合法的BST，如果这俩儿子有一个不是BST，以我为根的这棵树肯定就不是BST了
2. 如果左右子树都是合法的BST，我需要判断左右子树加上我还是不是合法BST了。我需要大于左子树的最大值，小于右子树的最小值。
3. 因为题目要计算最大节点值和，如果第二点满足，我还需要我们的这一整颗BST的所有节点值只合适多少，方便比较出最大值

**根据以上三点，那么当前节点需要知道以下具体信息：**

1. 左右子树是否BST
2. 左子树的最大值和右子树的最小值
3. 左右子树的所有节点值之和

按照上面所需的信息，如果按照前序遍历的方式，伪代码如下：

```python
def maxSumBST(self, root: TreeNode) -> int:
    # 维护一个全局变量，保存BST节点值之和的最大值
    maxSum = 0
    def traverse(root):
        if not root: return None
        # 前序遍历
        # 1. 左右子树是否BST
        if isBST(root.left) or isBST(root.right):
            goto next
        # 2. 左子树的最大值和右子树的最小值
        left = findMax(root.left)
        right = findMin(root.right)
        # 判断加上root后BST合法性
        if root.val < left or root.val > right:
            goto next
        # 计算当前节点值之和
        leftSum = findSum(root.left)
        rightSum = findSum(root.right)
        rootSum = leftSum + rightSum + root.val
        maxSum = max(rootSum, maxSum)
        
        next:
            traverse(root.left)
            traverse(root.right)
        # 需要额外的辅助函数 
        def findMax(root)
        def findMin(root)
        def findSum(root)
        def isBST(root)
```

可以看到，如果用前序遍历，时间复杂度会非常麻烦。

但是我们可以将前序遍历变为后序遍历，让递归函数完成辅助函数的功能：

让递归函数返回一个大小为4的数组``res=[0]*4``

`res[0]` 记录以 `root` 为根的二叉树是否是 BST，若为 1 则说明是 BST，若为 0 则说明不是 BST；

`res[1]` 记录以 `root` 为根的二叉树所有节点中的最小值；

`res[2]` 记录以 `root` 为根的二叉树所有节点中的最大值；

`res[3]` 记录以 `root` 为根的二叉树所有节点值之和。

其实这就是把之前分析中说到的几个值放到了 `res` 数组中，**最重要的是，我们要试图通过 `left` 和 `right` 正确推导出 `res` 数组**。

代码实现：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxSumBST(self, root: TreeNode) -> int:
        maxSum = 0
        def traverse(root):
            nonlocal maxSum
            if not root:
                return [1, 10**5, -10**5, 0]

            left = traverse(root.left)
            right = traverse(root.right)
            # res = [a, b, c, d] a:是否是合法BST，b:BST中最小值，c:BST中最大值，d:BST最大节点值和
            res = [0] * 4
            # 判断左右子树BST合法性以及左右根BST合法性
            if left[0] == 1 and right[0] == 1 and root.val > left[2] and root.val < right[1]:
                res[0] = 1
                res[1] = min(root.val, left[1])
                res[2] = max(root.val, right[2])
                res[3] = left[3] + right[3] + root.val
                maxSum = max(res[3], maxSum)
            else:
                res[0] = 0
            return res
        traverse(root)
        return maxSum
```

也不是所有题目都要使用后序遍历，就像中序遍历BST会返回有序序列一样。**如果当前节点要做的事情需要通过左右子树的计算结果推导出来，就要用到后序遍历**。

你计算以 `root` 为根的二叉树的节点之和，是不是可以通过左右子树的和加上 `root.val` 计算出来？

你计算以 `root` 为根的二叉树的最大值/最小值，是不是可以通过左右子树的最大值/最小值和 `root.val` 比较出来？

你判断以 `root` 为根的二叉树是不是 BST，是不是得先判断左右子树是不是 BST？是不是还得看看左右子树的最大值和最小值？

## 一道递归题

LeetCode：341 扁平化嵌套列表迭代器

[题目不让我做什么，我就偏要去做什么🤔 (qq.com)](https://mp.weixin.qq.com/s/uEmD5YVGG5LHQEmJQ2GSfw)

```python
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """
#
#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """
#
#    def getList(self) -> [NestedInteger]:
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """

class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.lists = []
        self.point = 0

        def get_list(n):
            for i in n:
                if i.isInteger():
                    self.lists.append(i.getInteger())
                else:
                    get_list(i.getList())

        get_list(nestedList)
        
    def next(self) -> int:
        self.point += 1
        return self.lists[self.point - 1]
    
    def hasNext(self) -> bool:
        if self.point < len(self.lists):
            return True
        else:
            return False

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
```

## 二叉树的最近公共祖先 & 递归函数三问

**遇到任何递归型的问题，无非就是灵魂三问**：

**1、这个函数是干嘛的**？

**2、这个函数参数中的变量是什么的是什么**？

**3、得到函数的递归结果，你应该干什么**

LeetCode：236题

**第一个问题，这个函数是干嘛的**？

​	给该函数输入三个参数`root`，`p`，`q`，它会返回一个节点。

​	情况 1，如果`p`和`q`都在以`root`为根的树中，函数返回的即使`p`和`q`的最近公共祖先节点。

​	情况 2，那如果`p`和`q`都不在以`root`为根的树中怎么办呢？函数理所当然地返回`null`呗。

​	情况 3，那如果`p`和`q`只有一个存在于`root`为根的树中呢？函数就会返回那个节点。

**第二个问题，这个函数的参数中，变量是什么**

​	函数参数中的变量是`root`，因为根据框架，`lowestCommonAncestor(root)`会递归调用`root.left`和`root.right`；至于`p`和`q`，我们要求它俩的公共祖先，它俩肯定不会变化的。

**第三个问题，得到函数的递归结果，你该干嘛**？

​	先想 base case，如果`root`为空，肯定得返回`null`。如果`root`本身就是`p`或者`q`，比如说`root`就是`p`节点吧，如果`q`存在于以`root`为根的树中，显然`root`就是最近公共祖先；即使`q`不存在于以`root`为根的树中，按照情况 3 的定义，也应该返回`root`节点。

​	情况 1，如果`p`和`q`都在以`root`为根的树中，那么`left`和`right`一定分别是`p`和`q`（从 base case 看出来的）。

​	情况 2，如果`p`和`q`都不在以`root`为根的树中，直接返回`null`。

​	情况 3，如果`p`和`q`只有一个存在于`root`为根的树中，函数返回该节点。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        def traverse(root, p, q):
            if not root: return None
            if root == p or root == q: return root

            left = traverse(root.left, p, q)
            right = traverse(root.right, p, q)
            if left and right:
                return root
            if not left and not right:
                return None
            return right if not left else left
        return traverse(root, p, q)

```

## 计算二叉树的节点数

首先，二叉树分为：满二叉树，完全二叉树，和普通二叉树

![](xiaobaishuati.assets/complete.png)

对于普通二叉树：时间复杂度是o(N)

```python
def count(root):
    if not root:
        return 0
    return 1 + count(root.left) + count(root.right)
```

对于满二叉树(perfect binary tree)：时间复杂度o(logN)

```python
def count(root):
    h = 0
    # 计算树的高度
    while root:
        root = root.left
        h += 1
    return pow(2,h) - 1
```

对于完全二叉树：时间复杂度是 O(logN*logN)。

```python
def count(root):
    l = root
    r = root
    # 记录左右子树的高度
    hl = hr = 0
    while l:
        l = l.left
        hl += 1
    while r:
        r = r.left
        hr += 1
    # 如果左右子树高度相同，就是满二叉树
    if hl == hr:
        return pow(2,hl) - 1
    else:
        # 否则跟普通二叉树一样
        return 1 + count(root.left) + count(root.right)
```

# 6. 图

## 图的遍历

图和多叉树的区别是，图可能包含环。如果包含环，在遍历的时候需要visited数组进行辅助

```python
visited = []
# 图遍历框架
def traverse(graph,s):
    if visited[s]: return
    visited[s] = True
    for neighbor in graph[s]:
        traverse(graph, neighbor)
    visited[s] = False
```

LeetCode：797题 所有可能的路径

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        res = []  
        n = len(graph)
        def traverse(graph, s, path):
            if s == n - 1:
                # path[:]相当于新建一个 避免使用引用
                res.append(path[:])
            
            for v in graph[s]:
                path.append(v)
                # 从开始继续遍历
                traverse(graph, v, path)
                # 返回上一个节点继续遍历
                path.pop()
        traverse(graph, 0, [0])
        return res
```

## 判断图中是否有环 & 拓扑排序

LeetCode：207 

LeetCode：210 

尚未解决

