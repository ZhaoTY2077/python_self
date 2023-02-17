# 牛客网算法100题

## 链表

### BM1 反转链表

思路：迭代法 + 优化空间复杂度

```python
class Solution:
    def ReverseList(self , head: ListNode) -> ListNode:
        # write code here
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre
```

### BM2 链表指定区间反转

思路一：四个指针

优点：思路简单，清晰

分别用pre, left, right, succ 四个指针记录：开始反转前一个节点，开始反转节点，结束反转节点，结束反转后一个节点

![image-20220302154903302](牛客网算法100题.assets/image-20220302154903302.png)

```python
class Solution:
    def reverseBetween(self , head: ListNode, m: int, n: int) -> ListNode:
        # write code here
        dummynode = ListNode(-1)
        dummynode.next = head
        pre = dummynode
        # 第1步 从head虚拟头结点走m-1步 到达需要反转前节点pre
        for _ in range(m - 1):
            pre = pre.next
        # 第2步 从pre节点往前走n-m+1步 到达right节点
        right = pre
        for _ in range(n - m + 1):
            right = right.next
        # 第3步 分割链表
        left = pre.next
        succ = right.next
        # 切断链接
        pre.next = None
        right.next = None
        # 第4步 反转left，right之间链表
        def reverse(head):
            pre, cur = None, head
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
        reverse(left)
        # 第5步 连接回原来的链表
        pre.next = right
        left.next = succ
        return dummynode.next
```

思路二：头插法

上个方法在left和right距离非常远时，时间复杂度是o(2n)

但是头插法只需要一次遍历即可完成

```python
class Solution:
    def reverseBetween(self , head: ListNode, m: int, n: int) -> ListNode:
        # write code here
        dummynode = ListNode(-1)
        dummynode.next = head
        pre = dummynode
        for _ in range(m - 1):
            pre = pre.next
        cur = pre.next
        for _ in range(n - m):
            nxt = cur.next
            cur.next = nxt.next
            nxt.next = pre.next
            pre.next = nxt
        return dummynode.next
```

### BM3 k个一组反转链表

```python
class Solution:
    # 反转[a, b)区间的链表
    def reverse(self, head, tail):
        pre, cur = None, head
        while cur != tail:
            nxt = cur.next
            cur.next= pre
            pre = cur
            cur = nxt
        return pre
    
    def reverseKGroup(self , head: ListNode, k: int) -> ListNode:
        # write code here
        if not head: return None
        a = b = head
        for i in range(k):
            # base case 不足k个 直接返回head
            if not b: return head
            b = b.next
        newHead = self.reverse(a, b)
        # 递归反转链表，然后拼接
        a.next = self.reverseKGroup(b, k)
        return newHead
```

### BM4 合并两个有序链表

```python
class Solution:
    def Merge(self , pHead1: ListNode, pHead2: ListNode) -> ListNode:
        # write code here
        pre = ListNode(0)
        cur = pre
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val: 
                cur.next = pHead1
                pHead1 = pHead1.next
            else:
                cur.next = pHead2
                pHead2 = pHead2.next
            cur = cur.next
        cur.next = pHead1 if pHead1 is not None else pHead2
        return pre.next
```

### BM5 合并k个有序链表

思路一：分冶法

时间复杂度：o(k * n * log(k))      分冶法合并k个链表o(k*log(k)),遍历链表o(n)

空间复杂度：o(log(k))

```python
class Solution:
    def mergeKLists(self , lists: List[ListNode]) -> ListNode:
        # write code here
        if not lists: return None
        n = len(lists)
        return self.mergeSort(lists, 0, n - 1)
    
    def mergeSort(self, lists, l, r):
        if l == r:
            return lists[l]
        m = (l + r) // 2
        s1 = self.mergeSort(lists, l, m)
        s2 = self.mergeSort(lists, m + 1, r)
        return self.merge(s1, s2)
    
    def merge(self, l1, l2):
        pre = ListNode(0)
        cur = pre
        while l1 and l2:
            if l1.val <= l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 if l1 is not None else l2
        return pre.next
```

思路二：堆

时间复杂度：o(k * n * log(k))        堆中的元素最多是k个，所以插入和删除的时间代价是O(log(k))，一共有kn个元素。

空间复杂度：o(k)

```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        import heapq
        minHeap = []
        for listi in lists:
            while listi:
                heapq.heappush(minHeap, listi.val)
                listi = listi.next
        pre = ListNode(0)
        cur = pre
        while minHeap:
            cur.next = ListNode(heapq.heappop(minHeap))
            cur = cur.next
        return pre.next
```

### BM6 判断链表中是否有环

双指针法

```python
class Solution:
    def hasCycle(self , head: ListNode) -> bool:
        slow = fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

### BM7 链表中环的入口

双指针法

快慢指针法，相遇后，将slow放回head节点，然后再次相遇时，就是环的入口

```python
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        slow = fast = pHead
        while True:
            if not (fast and fast.next): return None
            slow = slow.next
            fast = fast.next.next
            if slow == fast: break
        slow = pHead
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow
```

### BM8 倒数第k个节点

双指针法

```python
class Solution:
    def FindKthToTail(self , pHead: ListNode, k: int) -> ListNode:
        # write code here
        slow = fast = pHead
        for _ in range(k):
            if not fast: return None
            fast = fast.next
        while fast:
            slow, fast = slow.next, fast.next
        return slow
```

### BM9 删除链表倒数第k个节点

双指针法

```python
class Solution:
    def removeNthFromEnd(self , head: ListNode, n: int) -> ListNode:
        # write code here
        dummy = ListNode(0)
        dummy.next = head
        pre = cur = dummy
        for _ in range(n):
            cur = cur.next
        while cur.next:
            pre, cur = pre.next, cur.next
        pre.next = pre.next.next
        return dummy.next
```

### BM10 两个链表的第一个共同节点

双指针法

注：遍历链表时，应是 `a = a.next if a`而不是`a = a.next if a.next`

```python
class Solution:
    def FindFirstCommonNode(self , pHead1 , pHead2 ):
        # write code here
        if pHead1 is None or pHead2 is None:
            return None
        node1, node2 = pHead1, pHead2
        while node1 != node2:
            node1 = node1.next if node1 else pHead2
            node2 = node2.next if node2 else pHead1
        return node1
```

### BM11 链表相加

栈

注：使用头插法可以不用再反序遍历链表一次

```python
class Solution:
    def addInList(self , head1: ListNode, head2: ListNode) -> ListNode:
        # write code here
        if not head1: return head2
        if not head2: return head1
        
        def listnode2stack(head):
            res = []
            while head:
                res.append(head.val)
                head = head.next
            return res
        
        stack1, stack2 = listnode2stack(head1), listnode2stack(head2)
            
        dummy = ListNode(0)
        pre = dummy
        carry = 0
        
        while stack1 or stack2 or carry:
            n1 = stack1.pop() if stack1 else 0
            n2 = stack2.pop() if stack2 else 0
            
            s = n1 + n2 + carry
            
            carry = s // 10
            div = s % 10
            # 头插法
            node = ListNode(div)
            node.next = dummy.next
            dummy.next = node
        return dummy.next
```

反转链表法

```python
class Solution:
    def addInList(self , head1: ListNode, head2: ListNode) -> ListNode:
        # write code here
        if not head1: return head2
        if not head2: return head1
        
        def reverse(node):
            if not node:
                return None
            pre = None
            cur = node
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            return pre
        
        newHead1, newHead2 = reverse(head1), reverse(head2)

        dummy = ListNode(0)
        pre = dummy
        carry = 0
        
        while newHead1 or newHead2 or carry:
            n1 = newHead1.val if newHead1 else 0
            n2 = newHead2.val if newHead2 else 0
            newHead1 = newHead1.next if newHead1 else None
            newHead2 = newHead2.next if newHead2 else None
            s = n1 + n2 + carry
            
            carry = s // 10
            div = s % 10
            
            node = ListNode(div)
            node.next = dummy.next
            dummy.next = node
        return dummy.next
```

### BM12 链表排序

方法一：辅助数组（不推荐）

```python
class Solution:
    def sortInList(self , head: ListNode) -> ListNode:
        # write code here
        res = []
        cur = head
        while cur:
            res.append(cur.val)
            cur = cur.next
        res.sort()
        newHead = ListNode(0)
        cur = newHead
        for i in res:
            cur.next = ListNode(i)
            cur = cur.next
        return newHead.next
```

方法二：归并排序

模块化写法：

```python
class Solution:
    def sortInList(self , head: ListNode) -> ListNode:
        # 递归结束条件
        if not head or not head.next:
            return head

        # 找到链表中间节点并断开链表、递归
        midnode = self.cut(head)
        rightnode = midnode.next
        midnode.next = None

        left = self.sortInList(head)
        right = self.sortInList(rightnode)
        # 合并有序链表
        return self.merge(left, right)

    # 876. 链表的中间结点
    def cut(self, head):
        if not head or not head.next:
            return None
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        return slow
    
    # 21. 合并两个有序链表
    def merge(self, left, right):
        dummy = ListNode(0)
        cur = dummy
        while left and right:
            if left.val < right.val:
                cur.next = left
                left = left.next
            else:
                cur.next = right
                right = right.next
            cur = cur.next
        cur.next = left if left else right
        return dummy.next
```

一体式写法：

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param head ListNode类 the head node
# @return ListNode类
#
class Solution:
    def sortInList(self , head: ListNode) -> ListNode:
        if not head or not head.next: return head
        
        # cut the Linked at the mid index
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            
        # save and cut
        mid, slow.next = slow.next, None
        
        # recursive for cutting
        left, right = self.sortInList(head), self.sortInList(mid)
        
        # merge left and right linked list and return it 
        h = res = ListNode(0)
        while left and right:
            if left.val < right.val : 
                h.next, left = left, left.next
            else:
                h.next, right = right, right.next
            h = h.next
        h.next = left if left else right
        return res.next
```

### BM13 判断回文链表

双指针法

1. 先通过快慢指针法找到链表中点，若是奇数长度（`fast != None`），再向前一位
2. 反转中间节点后链表
3. 然后与头节点开始比较

```python
class Solution:
    def isPail(self , head: ListNode) -> bool:
        def reverse(node):
            if not node: return None
            pre, cur = None, node
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            return pre
        # write code here
        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        if fast: 
            slow = slow.next
        left = head
        right = reverse(slow)
        while right:
            if left.val != right.val:
                return False
            left, right = left.next, right.next
        return True
```

笔试可以用的方法：

```python
class Solution:
    def isPail(self , head: ListNode) -> bool:
        # write code here
        res = []
        cur = head
        while cur:
            res.append(cur.val)
            cur = cur.next
        return True if res == res[::-1] else False
```

想不出来再用

### BM14 链表的奇偶重排

```python
class Solution:
    def oddEvenList(self , head: ListNode) -> ListNode:
        # write code here
        if not head or not head.next:
            return head
        dummy = ListNode(0)
        n1 = head
        n2 = head.next
        dummy.next = n1
        pre = n2
        while n1 and n2 and n2.next:
            n1.next = n1.next.next
            n2.next = n2.next.next
            n1 = n1.next
            n2 = n2.next
        n1.next = pre
        return dummy.next
```

### BM15 删除有序链表中的重复项

方法：双指针法

- 沿着一样的找，找到不一样的赋值

```python
class Solution:
    def deleteDuplicates(self , head: ListNode) -> ListNode:
        # write code here
        if not head: return None
        slow = fast = head
        while fast:
            if fast.val != slow.val:
                slow.next = fast
                slow = slow.next
            fast = fast.next
        slow.next = None
        return head
```

- 沿着不一样的找，遇到一样的跳过，找下一个不一样的赋值

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 
# @param head ListNode类 
# @return ListNode类
#
class Solution:
    def deleteDuplicates(self , head: ListNode) -> ListNode:
        # write code here
        dummy = pre = head
        while pre and pre.next:
            if pre.val == pre.next.val:
                cur = pre.next
                while cur and cur.next and cur.val == cur.next.val:
                    cur = cur.next
                pre.next = cur.next
            pre = pre.next
        return dummy
```

### BM16 删除有序链表中的重复项2

```python
class Solution:
    def deleteDuplicates(self , head: ListNode) -> ListNode:
        # write code here
        if not head: return head
        dummy = ListNode(0)
        dummy.next = head
        cur = dummy
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                x = cur.next.val
                while cur.next and cur.next.val == x:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next
```

```python
class Solution:
    def deleteDuplicates(self , head: ListNode) -> ListNode:
        # write code here
        if not head: return head
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        cur = head
        while cur:
            # 跳过当前的重复节点，使得cur指向当前重复元素的最后一个位置
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if pre.next == cur:
                # pre和cur之间没有重复节点，pre后移
                pre = pre.next
            else:
                # pre->next指向cur的下一个位置（相当于跳过了当前的重复元素）
                # 但是pre不移动，仍然指向已经遍历的链表结尾
                pre.next = cur.next
            cur = cur.next
        return dummy.next
```

## 二分查找/排序

### BM17 二分查找

```python
class Solution:
    def search(self , nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                return mid
        return -1
```

### BM18 二维矩阵搜索

从右上角开始搜索

注意j的取值`j = n - 1 `和取值范围`j >= 0`

```python
class Solution:
    def Find(self , target: int, array: List[List[int]]) -> bool:
        # write code here
        if len(array) == 0 or len(array[0]) == 0:
            return False
        
        m, n = len(array), len(array[0])
        i, j = 0, n - 1
        while i < m and j >= 0:
            if array[i][j] < target:
                i += 1
            elif array[i][j] > target:
                j -= 1
            else:
                return True
        return False
```

### BM19 寻找峰值

方法：二分法

二分法的关键在于分段性，而不是单调性，在这题就可以体现。

```python
class Solution:
    def findPeakElement(self , nums: List[int]) -> int:
        # write code here
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            elif nums[mid] > nums[mid + 1]:
                right = mid
        return left
```

### BM20 数组中的逆序对

思路：归并排序

同剑指52

```python
class Solution:
    def InversePairs(self , data: List[int]) -> int:
        # write code here
        def mergesort(l, r):
            if l >= r : return 0
            m = (l + r) // 2
            res = mergesort(l, m) + mergesort(m + 1, r)
            i, j = l, m + 1
            temp[l: r + 1] = data[l: r + 1]
            for k in range(l, r + 1):
                if i == m + 1:
                    data[k] = temp[j]
                    j += 1
                elif j == r + 1 or temp[i] <= temp[j]:
                    data[k] = temp[i]
                    i += 1
                else:
                    data[k] = temp[j]
                    j += 1
                    res += m - i + 1
            return res
        temp = [0] * len(data)
        return mergesort(0, len(data) - 1) % 1000000007
```

### BM21 旋转数组中的最小值

思路：二分法

看到时间复杂度logn，就往二分法考虑。

这题的关键是没有给明确的target，那么可以考虑端点，在这题中，因为始终有右端点 <= 左端点，所以mid与right比

```python
class Solution:
    def minNumberInRotateArray(self , rotateArray: List[int]) -> int:
        # write code here
        left, right = 0, len(rotateArray) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if rotateArray[mid] > rotateArray[right]:
                left = mid + 1
            elif rotateArray[mid] < rotateArray[right]:
                right = mid
            else:
                right -= 1
        return rotateArray[left]
```

### BM22 比较版本号

思路：模拟

分割字符串后，较短的补零，然后逐位比较

```python
class Solution:
    def compare(self , version1: str, version2: str) -> int:
        # write code here
        v1_list = version1.split('.')
        v2_list = version2.split('.')
        v1_n = [int(i) for i in v1_list]
        v2_n = [int(i) for i in v2_list]
        
        for i in range(max(len(v1_n), len(v2_n))):
            n1 = v1_n[i] if i < len(v1_n) else 0
            n2 = v2_n[i] if i < len(v2_n) else 0
            if n1 > n2:
                return 1
            elif n1 < n2:
                return -1
            else:
                continue
        return 0
```

## 二叉树

### BM23 二叉树前序遍历

思路：递归

```python
class Solution:
    def preorderTraversal(self , root: TreeNode) -> List[int]:
        # write code here
        res = []
        def preoerder(root):
            if not root:
                return None
            res.append(root.val)
            preoerder(root.left)
            preoerder(root.right)
        preoerder(root)
        return res
```

思路：迭代

```python
class Solution:
    def preorderTraversal(self , root: TreeNode) -> List[int]:
        # write code here
        stack, res = [root], []
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                # 先插右子树，因为pop是从
                if node.right:
                    stack.append(node.right)
                if node.left:
                    stack.append(node.left)
        return res
```

### BM24 中序遍历

递归

注意：默认的递归深度不够用，需要设置一下

```python
import sys
sys.setrecursionlimit(100000)
class Solution:
    def inorderTraversal(self , root: TreeNode) -> List[int]:
        # write code here
        res = []
        def inorder(root):
            if not root:
                return None
            inorder(root.left)
            res.append(root.val)
            inorder(root.right)
        inorder(root)
        return res
```

### BM25 后序遍历

```python
class Solution:
    def postorderTraversal(self , root: TreeNode) -> List[int]:
        # write code here
        res = []
        def postorder(root):
            if not root:
                return None
            postorder(root.left)
            postorder(root.right)
            res.append(root.val)
        postorder(root)
        return res
```

### BM26 层序遍历

思路：队列

```python
class Solution:
    def levelOrder(self , root: TreeNode) -> List[List[int]]:
        # write code here
        if not root: return []
        queue, res = [root], []
        while queue:
            res.append([node.val for node in queue])
            childs = []
            for node in queue:
                if node.left:
                    childs.append(node.left)
                if node.right:
                    childs.append(node.right)
            queue = childs
        return res
```

### BM27 之字形打印二叉树

思路：层序遍历

加上一个对res长度的判断，如果`len(res) % 2`是0，那么就按正常顺序；否则，就逆序加入res

```python
class Solution:
    def Print(self , pRoot: TreeNode) -> List[List[int]]:
        # write code here
        if not pRoot: return None
        import collections
        queue = collections.deque()
        res = []
        queue.append(pRoot)
        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                temp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(temp[::-1] if len(res) % 2 else temp)
        return res
```

### BM28 二叉树的最大深度

思路：递归+动态规划

方法一：

```python
class Solution:
    def maxDepth(self , root: TreeNode) -> int:
        # write code here
        ans = 0
        def dfs(root):
            if not root: return 0
            left = dfs(root.left)
            right = dfs(root.right)
            return max(left, right) + 1
        ans = dfs(root)
        return ans
```

方法二：一行写法

```python
class Solution:
    def maxDepth(self , root: TreeNode) -> int:
        # write code here
        if not root: return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```

### BM29 二叉树中和为某一值的路径

思路：回溯

当tar减为0，且没有左右子树的时候，这条路径就是我们要的。

```python
class Solution:
    def hasPathSum(self , root: TreeNode, s: int) -> bool:
        res, path = [], []
        def dfs(node, tar):
            if not node:
                return None
            path.append(node.val)
            tar -= node.val
            if tar == 0 and not node.left and not node.right:
                res.append(list(path))
            dfs(node.left, tar)
            dfs(node.right, tar)   
            path.pop()
        dfs(root, s)
        return True if res else False
```

### BM30 二叉搜索树与双向链表

方法一：中序遍历+辅助数组

```python
class Solution:
    def Convert(self , pRootOfTree ):
        # write code here
        def dfs(root, res):
            if not root:
                return None
            dfs(root.left, res)
            res.append(root)
            dfs(root.right, res)
        res = []
        if not pRootOfTree: return None
        dfs(pRootOfTree, res)
        for i in range(len(res) - 1):
            res[i].right = res[i + 1]
            res[i + 1].left = res[i]
        return res[0]
```

但是空间复杂度是O(N)，占用了大小为n的辅助数组

方法二：中序遍历+原地修改

```python
class Solution:
    def __init__(self):
        self.pre = None
        self.root = None
        
    def Convert(self , pRootOfTree ):
        if not pRootOfTree: return None
        #递归左节点
        self.Convert(pRootOfTree.left)
        # 处理节点
        if not self.root:
            # 最左节点作为根节点
            self.root = pRootOfTree
        if self.pre:
            pRootOfTree.left = self.pre
            self.pre.right = pRootOfTree
        self.pre = pRootOfTree
        # 递归右节点
        self.Convert(pRootOfTree.right)
        return self.root
```

### BM31 对称二叉树

思路：递归判断

```python
class Solution:
    def isSymmetrical(self , pRoot: TreeNode) -> bool:
        # write code here
        def dfs(L, R):
            if not L and not R :
                return True
            if not L or not R or L.val != R.val: return False
            return dfs(L.left, R.right) and dfs(L.right, R.left)
        return dfs(pRoot.left, pRoot.right) if pRoot else True
```

### BM32 合并二叉树

思路：递归

```python
class Solution:
    def mergeTrees(self , t1: TreeNode, t2: TreeNode) -> TreeNode:
        # 如果t1是空就返回t2，如果t2是空就返回t1
        if not t1:
            return t2
        if not t2:
            return t1
        # 当前节点求和
        t1.val += t2.val
        # 递归合并左右子树
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1
```

### BM33 二叉树的镜像

思路：递归

```python
class Solution:
    def Mirror(self , root: TreeNode) -> TreeNode:
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root
```

### BM34 判断二插搜索树

思路：递归

```python
class Solution:
    def isValidBST(self , root: TreeNode) -> bool:
        # write code here
        def isBST(root, min, max):
            if not root:return True
            if min and root.val <= min.val: return False
            if max and root.val >= max.val: return False
            return isBST(root.left, min, root) and isBST(root.right, root, max)
        return isBST(root, None, None)
```

思路：中序遍历

```python
class Solution:
    def isValidBST(self , root: TreeNode) -> bool:
        # write code here
        res = []
        def dfs(root):
            if not root:return None
            dfs(root.left)
            res.append(root.val)
            dfs(root.right)
        dfs(root)
        for i in range(len(res) - 1):
            if res[i] >= res[i + 1]:
                return False
        return True
```

### BM35 判断是否是完全二叉树

思路：层序遍历

当层序遍历二叉树时，遇到第一个空节点，将flag变为True；若在之后再遇到非空节点，可以返回False，否则，返回True

```python
class Solution:
    def isCompleteTree(self , root: TreeNode) -> bool:
        # write code here
        if not root:
            return None
        queue = [root]
        end = False
        while queue:
            for _ in range(len(queue)):
                node = queue.pop(0)
                if not node:
                    end = True
                else:
                    if end: return False
                    queue.append(node.left)
                    queue.append(node.right)
        return True
```

### BM36 判断平衡二叉树

思路：递归

在计算二叉树深度的程序上修改，在后序遍历位置加上对左右子树深度的判断即可。

```python
class Solution:
    def IsBalanced_Solution(self , root: TreeNode) -> bool:
        # write code here
        isbalance = True
        def deepth(root):
            nonlocal isbalance
            if not root: 
                return 0
            left = deepth(root.left)
            right = deepth(root.right)
            if abs(left - right) > 1:
                isbalance = False
            return 1 + max(left, right)
        deepth(root)
        return isbalance
```

### BM37 二叉搜索树的最近公共节点

思路：递归

利用二叉搜索树的有序性质，若p q都小于当前节点，则公共节点在root左边；若p q都大于当前节点，则公共节点都在root右边；

若p < root < q 则当前节点就是公共节点

```python
class Solution:
    def lowestCommonAncestor(self , root: TreeNode, p: int, q: int) -> int:
        # write code here
        if not root: return None
        if p > q: return self.lowestCommonAncestor(root, q, p)
        if p <= root.val <= q:
            return root.val
        if root.val > q:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return self.lowestCommonAncestor(root.right, p, q)
```

### BM38 二叉树的最近公共祖先

思路：递归

比二插搜索树更加复杂，不可以简单的用跟root.val的大小作比较，需要按照后序遍历模板来进行更改。

递归函数 `lowestCommonAncestor(root, p, q)`

函数描述：提供给lowestCommonAncestor三个参数 root p q，它会返回一个节点

情况1：p q 都在以root为根的树中，函数返回的即是p q 的最近公共祖先

情况2：p q 都不在以root为根的树中，函数返回None

情况3： p或q在以root为根的树中，函数返回当前节点

按照二叉树遍历模板

```python
def traverse(root):
    def traverse(root.left)
    def traverse(root.right)
```

根据问题进行改进

```python
def lowestCommonAncestor(root, p, q):
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
```

接下来讨论base case情况：

1. 若root为空， 返回None
2. 若 root == p 或root == q，则返回root。（root就是最近公共祖先）

```python
def lowestCommonAncestor(root, p, q):
    # base case
    if not root: return None
    if root == p or root == q: return root
    
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
```

再根据三种情况分别处理当前节点情况。

```python
class Solution:
    def lowestCommonAncestor(self , root: TreeNode, o1: int, o2: int) -> int:
        # write code here
        if not root: return None
        if root.val == o1 or root.val == o2: return root.val
        
        left = self.lowestCommonAncestor(root.left, o1, o2)
        right = self.lowestCommonAncestor(root.right, o1, o2)
        # 情况1
        if left and right:
            return root.val
        # 情况2
        if not left and not right:
            return None
        # 情况3
        return left if left else right
```

注：情况1中，因为是二叉树的后续遍历位置，所以从p q往上走，第一次相遇的节点就肯定是最近的公共祖先

### BM39 二叉树的序列化与反序列化

思路：前序遍历 + 根据前序遍历生成二叉树

```python
class Solution:
    def Serialize(self, root):
        # write code here
        res = []
        def dfs(root):
            if not root: return res.append('#')
            res.append(str(root.val))
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return ','.join(res)
    
    def Deserialize(self, s):
        # write code here
        if not s:
            return []
        s_s = s.split(',')
        
        def dfs(nodes):
            val = nodes[0]
            del nodes[0]
            if val == '#':
                return None
            root = TreeNode(int(val))
            root.left = dfs(nodes)
            root.right = dfs(nodes)
            return root
        root = dfs(s_s)
        return root  
```

**ATTENTION:**

**序列化时使用列表先暂时储存每一个字符串的值，最后拼接起来，可以防止最后多一位``','``**

**反序列化时，取出第一个节点后，要删除它，这样就可以直接把剩下的列表直接传到递归函数**

### BM40 从前序遍历和中序遍历结构中构建二叉树

思路：递归判断

```python
class Solution:
    def reConstructBinaryTree(self , pre: List[int], vin: List[int]) -> TreeNode:
        # write code here
        if not pre and not vin:
            return None
        root = TreeNode(pre[0])
        index = vin.index(pre[0])
        root.left = self.reConstructBinaryTree(pre[1: index + 1], vin[: index])
        root.right = self.reConstructBinaryTree(pre[index + 1:], vin[index + 1 :])
        return root
```

### BM41 输出二叉树的右视图

思路：层序遍历，res中只添加每一层的最后一个元素

```python
class Solution:
    def solve(self , xianxu: List[int], zhongxu: List[int]) -> List[int]:
        # write code here
        root = self.reConstruct(xianxu, zhongxu)
        queue = [root]
        ans = []
        while queue:
            ans.append([node.val for node in queue][-1])
            childs = []
            for node in queue:
                if node.left: childs.append(node.left)
                if node.right: childs.append(node.right)
            queue = childs
        return ans
        
    def reConstruct(self, preorder, inorder):
        if not preorder and not inorder:
            return None
        root = TreeNode(preorder[0])
        index = inorder.index(preorder[0])
        root.left = self.reConstruct(preorder[1 : index + 1], inorder[:index])
        root.right = self.reConstruct(preorder[index + 1:], inorder[index + 1:])
        return root
```

## 堆/栈/队列

### BM42 用两个栈实现队列

```python
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)
        
    def pop(self):
        # return xx
        if not self.stack1:
            return -1
        else:
            self.move(self.stack1, self.stack2)
            t = self.stack2.pop()
            self.move(self.stack2, self.stack1)
            return t
    
    def move(self, s1, s2):
        while s1:
            node = s1.pop()
            s2.append(node)
```

### BM43 包含min的栈

思路：辅助栈

stack：是主要栈，存储全部元素

m：辅助栈，存储单调非增元素

```python
class Solution:
    def __init__(self):
        self.stack = []
        self.m = []
        
    def push(self, node):
        # write code here
        self.stack.append(node)
        if not self.m or self.m[-1] >= node:
            self.m.append(node)
            
    def pop(self):
        # write code here
        if self.stack.pop() == self.m[-1]:
            self.m.pop()
        
    def top(self):
        # write code here
        return self.stack[-1]
    
    def min(self):
        # write code here
        return self.m[-1]
```

### BM44 有效括号序列

思路：栈

```python
class Solution:
    def isValid(self , s: str) -> bool:
        stack = []
        for i in range(len(s)):
            if s[i] in ['{','[','(']:
                stack.append(s[i])
            else:
                if stack and stack[-1] + s[i] in ['()','[]','{}']:
                    stack.pop()
                else:
                    return False
        return not stack
```

更简洁的写法：

```python
class Solution:
    def isValid(self , s: str) -> bool:
        n = len(s)
        if n % 2 != 0: return False
        stack = []
        pairs = {
            ')':'(',
            ']':'[',
            '}':'{'
        }
        for i in s:
            if stack and i in pairs:
                if stack[-1] == pairs[i]: stack.pop()
                else: return False
            else:
                stack.append(i)
        return not stack
```

### BM45 滑动窗口的最小值

思路：单调队列（类似43题的单调栈）

如果使用暴力解法，时间复杂度是o(NK)， N是数组长度，K是窗口长度，因为每次移动窗口后最差的情况都要O(K)的时间复杂度来找到最大值。

优化思路就是使用O(1)的时间复杂度来找到窗口内的最大值。

优化方法是使用单调队列。保证单调队列始终是单调递减的，每次返回队列头即可。

再往单调队列中添加时，进行判断，若队尾小于要添加元素，则弹出队尾，直至保证队列的单调性。

```python
class Solution:
    def maxInWindows(self , num: List[int], size: int) -> List[int]:
        import collections
        if size > len(num) or size == 0: return None
        deque = collections.deque()
        for i in range(size):
            while deque and deque[-1] < num[i]:
                deque.pop()
            deque.append(num[i])
        res = [deque[0]]
        for i in range(size, len(num)):
            if deque[0] == num[i - size]:
                deque.popleft()
            while deque and deque[-1] < num[i]:
                deque.pop()
            deque.append(num[i])
            res.append(deque[0])
        return res
```

### BM46 最小的k个数

思路：排序（快速排序，归并排序，堆排序）

1. 快速排序

快速排序可以优化，不用全部排完，只要找到哨兵节点左边k-1长度的数组就可以直接返回

所以当哨兵排序玩后，当k < i 则返回左半边，当 k > i 则返回右半边

```python
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        if k >= len(arr): return arr
        def quick_sort(nums, l, r):
            if l >= r: return
            i, j = l, r
            while i < j:
                while i < j and nums[j] >= nums[l]: j -= 1
                while i < j and nums[i] <= nums[l]: i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[l], nums[i] = nums[i], nums[l]
            if k < i: quick_sort(nums, l, i - 1)
            if k > i: quick_sort(nums, i + 1, r)
        quick_sort(arr, 0, len(arr) - 1)
        return arr[:k]
```

2. 归并排序

```python
class Solution:
    def GetLeastNumbers_Solution(self , input: List[int], k: int) -> List[int]:
        # write code here
        if k >= len(input) : return input
        return self.mergesort(input)[:k]
    
    def mergesort(self, nums):
        if len(nums) <= 1: return nums
        mid = len(nums) // 2
        left = self.mergesort(nums[:mid])
        right = self.mergesort(nums[mid:])
        return self.merge(left, right)
    
    def merge(self, l1, l2):
        res = []
        i = j = 0
        while i < len(l1) and j < len(l2):
            if l1[i] <= l2[j]:
                res.append(l1[i])
                i += 1
            else:
                res.append(l2[j])
                j += 1
        if i == len(l1): res.extend(l2[j:])
        if j == len(l2): res.extend(l1[i:])
        return res
```

### BM47 寻找第K大

思路一：快速排序

```python
class Solution:
    def findKth(self , a: List[int], n: int, K: int) -> int:
        # write code here
        def quick_sort(a, l, r):
            if l >= r: return None
            i, j = l, r
            while i < j:
                while i < j and a[j] <= a[l]: j -= 1
                while i < j and a[i] >= a[l]: i += 1
                a[i], a[j] = a[j], a[i]
            a[l], a[i] = a[i], a[l]
            quick_sort(a, l, i - 1)
            quick_sort(a, i + 1, r)

        quick_sort(a, 0, len(a) - 1)
        return a[K - 1]
```

但是实际上我们只需要找到第k大的数即可，不关心第k大左右两边的子数组是否有序，因此可以使用快速选择法。

思路二：快速选择法

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def qick_sort(nums, l, r):
            # 注意要随机选择pivot，否则可能会导致最差的O(N^2)
            piovt = random.randint(l, r)
            nums[l], nums[piovt] = nums[piovt], nums[l]
            i, j = l, r
            while i < j:
                while i < j and nums[j] <= nums[l]: j -= 1
                while i < j and nums[i] >= nums[l]: i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[l], nums[i] = nums[i], nums[l]
            if i == k - 1:
                return nums[i]
            elif i < k - 1:
                return qick_sort(nums, i + 1, r)
            else:
                return qick_sort(nums, l, i - 1)
        return qick_sort(nums, 0, len(nums) - 1)
```

**注意要随机选择pivot，否则可能会导致最差的O(N^2)**

思路三：小顶堆

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = [x for x in nums[:k]]
        heapq.heapify(heap)    # 构造大小为k的小顶堆
        n = len(nums)
        for i in range(k, n):
            if heap[0] < nums[i]:
                heapq.heappop(heap)
                heapq.heappush(heap, nums[i])
        return heap[0]
```

### BM48 数据流的中位数

思路：堆

用小根堆（A）存较大的一半数据，用大根堆（B）存较小的一半数据

当有一个新数据来时，若小根堆（A）大根堆（B）大小一样，先向大根堆（B）中添加，然后把大根堆（B）堆顶弹出，放入小根堆（A）中

若小根堆（A）大根堆（B）大小不一样，则先向小根堆（A）中添加，然后把（A）堆顶弹出，放入大根堆（B）中

```python
from heapq import *
class Solution:
    def __init__(self):
        self.A = [] # 小顶堆，存较大的一半
        self.B = [] # 大顶堆，存较小的一半
        
    def Insert(self, num):
        if len(self.A) != len(self.B):
            heappush(self.B, -heappushpop(self.A, num))
        else:
            heappush(self.A, -heappushpop(self.B, -num))

    def GetMedian(self):
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0
```

### BM49 表达式求值

思路：栈

```python
class Solution:
    def solve(self , s: str) -> int:
        # write code here
        n = len(s)
        stack = []
        preSign = '+'
        num = 0
        i = 0
        while i < n:
            if s[i] != ' ' and s[i].isdigit():
                num = 10* num + int(s[i])
            if s[i] == '(':
                cnt = 0
                j = i
                while i < n:
                    if s[i] == '(':
                        cnt += 1
                    elif s[i] == ')':
                        cnt -= 1
                    if cnt == 0:
                        break
                    i += 1
                num = self.solve(s[j +1 : i])
            if i == n - 1 or s[i] in '+-*/':
                if preSign == '+':
                    stack.append(num)
                elif preSign == '-':
                    stack.append(-num)
                elif preSign == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                preSign = s[i]
                num = 0
            i += 1
        return sum(stack)
```

更加整洁的写法：

- 遇到`(`开始递归，遇到`)`结束递归

```python
class Solution:
    def solve(self , s: str) -> int:
        def helper(s):
            stack = []
            num = 0
            sign = '+'
            while len(s) > 0:
                c = s.popleft()
                if c.isdigit():
                    num = 10 * num + int(c)
                if c == '(':
                    num = helper(s)
                if (not c.isdigit() and c != ' ') or len(s) == 0:
                    if sign == '+' :
                        stack.append(num)
                    elif sign == '-':
                        stack.append(-num)
                    elif sign == '*':
                        stack[-1] = stack[-1] * num
                    elif sign == '/':
                        stack[-1] = int(stack[-1] / float(num))
                    num = 0
                    sign = c
                if c == ')': break
            return sum(stack)
        import collections
        return helper(collections.deque(s))
```

## 哈希

### BM50 两数之和

思路：哈希

```python
class Solution:
    def twoSum(self , numbers: List[int], target: int) -> List[int]:
        index = {}
        for i in range(len(numbers)):
            index[numbers[i]] = i
        for i, num in enumerate(numbers):
            j = index.get(target - num)
            if j is not None and j != i:
                return [i + 1, j + 1]
```

### BM51 超过数组长度一半的数

思路：哈希

手写哈希字典时，注意初始化方法

```python
class Solution:
    def MoreThanHalfNum_Solution(self , nums: List[int]) -> int:
        # write code here
        if not nums:
            return None
        if len(nums) == 1:
            return nums[0]
        count = {}
        for i in nums:
            if i not in count:
                count[i] = 1
            else:
                count[i] += 1
                if count[i] > len(nums) / 2:
                    return i
        return None
```

使用库函数

```python
class Solution:
    def MoreThanHalfNum_Solution(self , nums: List[int]) -> int:
        import collections
        count = collections.Counter(nums)
        return max(count.keys(), key=count.get)
```

思路二：排序

```python
class Solution:
    def MoreThanHalfNum_Solution(self , nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums) // 2]
```

思路三：摩尔投票法

```python
class Solution:
    def MoreThanHalfNum_Solution(self , nums: List[int]) -> int:
        if not nums:
            return None
        res = nums[0]
        times = 1
        for i in nums[1:]:
            if times == 0:
                res = i
                times += 1
            elif i == res:
                times += 1
            else:
                times -= 1
        return res
```

更简洁的写法：

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        for num in nums:
            if votes == 0:
                x = num
            votes += 1 if num ==x else -1
        return x
```

### BM52 只出现一次的两个数字

思路：位运算

将数组分为两个部分

一个部分是与异或后的结果做与运算为0的部分；一部分是与异或结果做与运算为1 的部分

```python
class Solution:
    def FindNumsAppearOnce(self , nums: List[int]) -> List[int]:
        # write code here
        xorsum = 0
        for num in nums:
            xorsum ^= num
        lsb = xorsum & (-xorsum)
        type1 = type2 = 0
        for num in nums:
            if num & lsb:
                type1 ^= num
            else:
                type2 ^= num
        return [type1, type2] if type1 < type2 else [type2, type1]
```

思路二：哈希统计次数

```python
class Solution:
    def FindNumsAppearOnce(self , array: List[int]) -> List[int]:
        # write code here
        import collections
        count = collections.Counter(array)
        res = [k for k, v in count.items() if v == 1]
        res.sort()
        return res
```

### BM53 缺失的第一个正整数

思路一：原地哈希

因为要求空间复杂度是O(1)，所以不能另外新建哈希表，只能原地哈希

先把所有不在`[1,n]`范围的数变成 N + 1，去除影响

然后如果遇到`[1,n]`区间的数x，就把数组中`x-1`位置的数变为负数，这样第一个遇到正数的索引`i`再 `+ 1`就可以得到缺失的第一个正整数

```python
class Solution:
    def minNumberDisappeared(self , nums: List[int]) -> int:
        # write code here
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n + 1
        for i in range(n):
            num = abs(nums[i])
            if num <= n:
                nums[num - 1] = -abs(nums[num - 1])
        
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1
```

思路二：置换

原地更改数组，将第i位数字x放在索引x-1的位置上，遍历所有数组，若索引i的数字不是i+1，则返回i+1

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1
```

### BM54 三数之和

思路：双指针法

在两数之和的基础上，先穷举第一个数`nums[k]`，然后剩下的按照两数之和的方法

```python
class Solution:
    def threeSum(self , nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []
        for k in range(n - 2):
            if nums[k] > 0: break # 1. because nums[j] > nums[i] > nums[k]
            if k > 0 and nums[k] == nums[k - 1]: continue # 2. skip the same nums[k]
            i, j = k + 1, n - 1
            while i < j:
                s = nums [i] + nums[j] + nums[k]
                if s < 0:
                    i += 1
                    while i < j and nums[i] == nums[i - 1]: i += 1
                elif s > 0:
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]: j -= 1
                else:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]: i += 1
                    while i < j and nums[j] == nums[j + 1]: j -= 1
        return res
```

## 递归 / 回溯

### BM55 没有重复项的全排列

思路：回溯

最基础的回溯模板

```python
class Solution:
    def permute(self , nums: List[int]) -> List[List[int]]:
        # write code here
        def backtrack(first=0):
            if first == n:
                return res.append(nums[:])
            for i in range(first, n):
                nums[first], nums[i] = nums[i], nums[first]
                backtrack(first + 1)
                nums[first], nums[i] = nums[i], nums[first]
        n = len(nums)
        res = []
        backtrack()
        return res
```

### BM56 有重复项的全排列

思路：回溯

加上一个剪枝的操作

```python
class Solution:
    def permuteUnique(self , nums: List[int]) -> List[List[int]]:
        # write code here
        nums.sort()
        res = []
        check = [0 for _ in range(len(nums))]
        def backtrack(sol, nums, check):
            if len(sol) == len(nums):
                res.append(sol)
            for i in range(len(nums)):
                if check[i] == 1:
                    continue
                if i > 0 and nums[i] == nums[i - 1] and check[i - 1] == 0:
                    continue
                check[i] = 1
                backtrack(sol + [nums[i]], nums, check)
                check[i] = 0
        backtrack([], nums, check)
        return res
```

用集合来做剪枝

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        def backtrack(x):
            if x == len(nums):
                res.append(nums[:])
            dic = set()
            for i in range(x, len(nums)):
                if nums[i] in dic: continue
                dic.add(nums[i])
                nums[i], nums[x] = nums[x], nums[i]
                backtrack(x + 1)
                nums[i], nums[x] = nums[x], nums[i]
        backtrack(0)
        return res
```

### BM57 岛屿数量

思路：递归

每遇到一个岛屿，就将其淹没，并数量 + 1

```python
class Solution:
    def solve(self , grid: List[List[str]]) -> int:
        # write code here
        def dfs(grid, i, j):
            # 超出索引边界
            if i < 0 or i >= m or j < 0 or j >= n:
                return 
            # 已经是岛屿
            if grid[i][j] == '0': 
                return
            # 淹没当前位置
            grid[i][j] = '0'
            # 淹没周围所属岛屿
            dfs(grid, i + 1, j)
            dfs(grid, i - 1, j)
            dfs(grid, i, j + 1)
            dfs(grid, i, j - 1)
            
        if not grid: return 0
        ans = 0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                # 遇到岛屿
                if grid[i][j] == '1':
                    ans += 1
                    dfs(grid, i, j)
        return ans
```

### BM58 字符串的排列

思路：回溯

用集合来判断重复字符

```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        c, res = list(s), []
        def dfs(x):
            if x == len(c) - 1:
                res.append(''.join(c))    # 添加排列方案
                return
            dic = set()
            for i in range(x, len(c)):
                if c[i] in dic: continue    # 重复，剪枝
                dic.add(c[i])
                c[i], c[x] = c[x], c[i]    # 交换，将c[i]固定在第x位
                dfs(x + 1)                 # 开启固定第x + 1位
                c[i], c[x] = c[x], c[i]    # 恢复交换
        dfs(0)
        return res
```

### BM59 N皇后问题

类似： LeetCode51 题，但是LC需要输出每一种棋盘的结果

思路：回溯

需要模拟建立一个棋盘，然后对每一种下法进行判断（isvalid函数）

```python
# LeetCode 51题
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if not n : return []
        board = [['.'] * n for _ in range(n)]
        res = []
        def isvalid(board, row, col):
            # 判断同一列是否冲突
            for i in range(len(board)):
                if board[i][col] == 'Q':
                    return False
            # 判断左上角是否冲突
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            # 判断右上角是否冲突
            i, j = row - 1, col + 1
            while i >= 0 and j < len(board):
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True

        def backtrack(board, row, n):
            if row == n:
                temp_res = []
                for temp in board:
                    temp_str = "".join(temp)
                    temp_res.append(temp_str)
                res.append(temp_res)
            for col in range(n):
                if not isvalid(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtrack(board, row + 1, n)
                board[row][col] = '.'
        backtrack(board, 0, n)
        return res
```

```python
# 牛客网58题
class Solution:
    def Nqueen(self , n: int) -> int:
        # write code here
        if not n : return []
        board = [['.'] * n for _ in range(n)]
        self.res = 0
        
        def isvalid(board, row, col):
            # 判断同一列是否冲突
            for i in range(len(board)):
                if board[i][col] == 'Q':
                    return False
            # 判断左上角是否冲突
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            # 判断右上角是否冲突
            i, j = row - 1, col + 1
            while i >= 0 and j < len(board):
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True
        
        def backtrack(board, row, n):
            if row == n:
                self.res += 1
            for col in range(n):
                if not isvalid(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtrack(board, row + 1, n)
                board[row][col] = '.'
        backtrack(board, 0, n)
        return self.res
```

### BM60  括号生成

思路：回溯

```python
class Solution:
    def generateParenthesis(self , n: int) -> List[str]:
        # write code here
        res = []
        
        def backtrack(S, left, right):
            if len(S) == 2 * n:
                res.append(''.join(S))
            if left < n:
                S.append('(')
                backtrack(S, left + 1, right)
                S.pop()
            if right < left :
                S.append(')')
                backtrack(S, left, right + 1)
                S.pop()
        backtrack([], 0, 0)
        return res 
```

### BM61 矩阵最长递增路径

思路：递归 深度优先搜索

matrix 为储存原本数字的矩阵
创建一个新矩阵 store，每一个格子 (i,j) 表示在 matrix 上走到格子 (i,j)，最长的递增路径长度
 
旁白：每次当我们看到一个新格子，我们问什么问题？
 
小明问：我能从哪里走到这个格子？
小红答：上、下、左、右四个格子中，matrix 中储存的数字小于我这个格子的地方，都可以走到我现在这个格子
 
小明问：那我在能选的选项里选哪个？
小红答：选上、下、左、右中 store值 最大的
 
小明问：那我现在这个格子的 store值 应该存多少？
小红答：就是我选择的上一个格子储存的 store值 +1 呗
 
小明问：那我怎么算上、下、左、右每个格子储存了多长的路径？
小红答：重复上面相同的搜索方式(查看这些格子各自的上下左右)，直到我遇到一个格子，这个格子上下左右比它 matrix值 小的格子的 store值 都已知了
 
小明问：可是我没有初始化我的 store 矩阵，怎么可能一开始就深搜到一个周围 store值 都已知的格子呢？
小红说：当你搜到一个格子，这个格子里的 matrix 数字很小，上下左右格子都比它大，你说我现在这个格子的 store值 的计算，还需要看上下左右的 store值 吗？这个格子的 store值 直接填1就可以啦
 
小红补充说：从某个格子开始的深搜，不能保证整个 matrix 都被覆盖，所以你每次 matrix 里遇到一个还没有 store值 的格子，就从它开始往底下深搜就行啦。
小明说：啦啦啦啦啦啦

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        record = [[0] * n for _ in range(m)]
        res = 0

        def dfs(i, j):
            nonlocal res
            compare = []
            for dx, dy in [[1, 0], [0, 1], [-1, 0],[0, -1]]:
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n and matrix[x][y] < matrix[i][j]:
                    if record[x][y]:
                        compare.append(record[x][y])
                    else:
                        compare.append(dfs(x, y))
            record[i][j] = max(compare) + 1 if compare else 1
            res = max(res, record[i][j])
            return record[i][j]
        
        for i in range(m):
            for j in range(n):
                if not record[i][j]:
                    dfs(i, j)
        return res
```

### BM74 将字符串转换为IP地址

思路：回溯 + 剪枝

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        SEG_COUNT = 4
        ans = []
        segments = [0] * SEG_COUNT

        def dfs(segId, segStart):
            # 如果找到了 4 段 ip地址并且遍历完字符串，那么就是一种答案
            if segId == SEG_COUNT:
                if segStart == len(s):
                    ipAddr = '.'.join(str(seg) for seg in segments)
                    ans.append(ipAddr)
                return 

            # 如果遍历完了字符串 但还没有找到 4 段ip地址，提前回溯
            if segStart == len(s):
                return
            
            # 由于不能有前导 0 ，如果当前数字为 0 ，那么这段 ip 地址只能为 0 
            if s[segStart] == '0':
                segments[segId] = 0
                dfs(segId + 1, segStart + 1)
            
            # 一般情况
            addr = 0
            for segEnd in range(segStart, len(s)):
                addr = addr * 10 + (ord(s[segEnd]) - ord('0'))
                if 0 < addr <= 255:
                    segments[segId] = addr
                    dfs(segId + 1, segEnd + 1)
                else:
                    break
        dfs(0, 0)
        return ans
```

## 动态规划

### BM62 斐波那契数列

思路：动态规划

这个问题可以用递归，但是时间复杂度会很高达到o(2^n)

所以还是使用动态规划思路比较好

```python
class Solution:
    def Fibonacci(self , n: int) -> int:
        if n <= 2:
            return 1
        dp = [1] * n
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n - 1]
```

时间复杂度：O(N)

空间复杂度：O(N)

空间复杂度还可以优化

```python
class Solution:
    def Fibonacci(self , n: int) -> int:
        if n <= 2:
            return 1
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
```

空间复杂度：O(1)

### BM63 跳台阶

思路：动态规划

```python
class Solution:
    def jumpFloor(self , n: int) -> int:
        if n == 1: return 1
        dp = [1] * n
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n - 1]
```

空间复杂度优化至O(1)

```python
class Solution:
    def jumpFloor(self , n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a
```

### BM64 最小代价爬楼梯

思路：动态规划

在青蛙跳台阶的基础上加上权重

```python
class Solution:
    def minCostClimbingStairs(self , cost: List[int]) -> int:
        n = len(cost)
        if n == 1 or n == 2: return min(cost)
        dp = [0] * (n + 1)
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        return dp[n]
```

因为当 i > 2 时，dp[i] 只与 dp[i - 1] 和 dp[i - 2] 有关，所以可以用 pre 和 cur 代替进行空间复杂度优化

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        if n <= 2: return min(cost)
        pre, cur = 0, 0
        for i in range(2, n + 1):
            nxt = min(cur + cost[i - 1], pre + cost[i - 2])
            pre, cur = cur, nxt
        return cur
```

### BM65 最长公共子序列（二）

思路：动态规划

要记录下最长子序列的结果

```python
class Solution:
    def LCS(self , s1: str, s2: str) -> str:
        # write code here
        m, n = len(s1), len(s2)
        if m == 0 or n == 0: return -1
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        ans = []
        i, j = m, n
        while i >= 1 and j >= 1:
            if s1[i - 1] == s2[j - 1]:
                ans.append(s1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] < dp[i][j - 1]:
                j -= 1
            else:
                i -= 1
        if ans == []:
            return '-1'
        else:
            ans.reverse()
            return ''.join(ans)
```

```python
class Solution:
    def LCS(self , s1: str, s2: str) -> str:
        # write code here
        m, n = len(s1), len(s2)
        if m == 0 or n == 0: return -1
        dp = [[''] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + s1[i - 1]
                else:
                    dp[i][j] = dp[i - 1][j] if len(dp[i - 1][j]) > len(dp[i][j - 1]) else dp[i][j - 1]
        if dp[m][n] == '':
            return '-1'
        else:
            return dp[m][n]
```

### BM66 最长公共子串

思路：动态规划

```python
class Solution:
    def LCS(self , str1: str, str2: str) -> str:
        # write code here
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        last_index = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 如果遇到了更长的子串，要更新，记录最长子串的长度，
                # 以及最长子串最后一个元素的位置
                if str1[i- 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        last_index = i
                else:
                    dp[i][j] = 0
        return str1[last_index - max_length: last_index]
```

但是python的时间超了，java能过

```java
public class Solution {
    public String LCS (String str1, String str2) {
        int maxLength = 0;
        int lastIndex = 0;
        int[][] dp = new int[str1.length() + 1][str2.length() + 1];
        for (int i = 0; i < str1.length(); i++){
            for (int j = 0; j < str2.length(); j++){
                if (str1.charAt(i) == str2.charAt(j)){
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                    if (dp[i + 1][j + 1] > maxLength){
                        maxLength = dp[i + 1][j + 1];
                        lastIndex = i;
                    }
                } else {
                    dp[i + 1][j + 1] = 0;
                }
            }
        }
        return str1.substring(lastIndex - maxLength + 1, lastIndex + 1);
    }
}
```

思路二：滑动窗口

```python
class Solution:
    def LCS(self , str1: str, str2: str) -> str:
        # write code here
        m, n = len(str1), len(str2)
        res = ''
        left = 0
        for i in range(m + 1):
            if str1[left : i + 1] in str2:
                res = str1[left : i + 1]
            else:
                left += 1
        return res
```

### BM67 不同路径的数目

思路：动态规划

```python
class Solution:
    def uniquePaths(self , m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m +1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if i == 1 or j == 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m][n]
```

时间复杂度：O(MN)

空间复杂度：O(MN)

空间复杂度优化：

```python
class Solution:
    def uniquePaths(self , m: int, n: int) -> int:
        pre = [1] * n
        cur = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                cur[j] = pre[j] + cur[j - 1]
            pre = cur[:]
        return pre[-1]
```

思路二：排列组合

比如，`m=3, n=2`，我们只要向下 1 步，向右 2 步就一定能到达终点。

所以有 C_{m+n-2}^{m-1}

```python
class Solution:
    def uniquePaths(self , m: int, n: int) -> int:
        import math
        return int(math.factorial(m + n - 2) / math.factorial(m - 1) / math.factorial(n - 1))
```

### BM68 矩阵的最小路径和

思路：动态规划表

注：关键是搞清楚`dp[i][j]`的定义

```python
class Solution:
    def minPathSum(self , matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = matrix[0][0]
        
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + matrix[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + matrix[0][j]
        for i in range(1, m):
            for j in range(1, n):
                    dp[i][j] = min(dp[i - 1][j] + matrix[i][j], 
                                   dp[i][j - 1] + matrix[i][j])
        return dp[m - 1][n - 1]
```

### BM69 把数字翻译成字符串

思路：动态规划

>  注意：0在这里不能表示字母，所以需要特别注意 与**剑指offer46题**有区别

```python
class Solution:
    def solve(self , nums: str) -> int:
        if not nums or len(nums) == 0 or nums == '0': return 0
        dp = [0] * (len(nums) + 1)
        dp[0] = 1
        dp[1] = 0 if nums[0] == '0' else 1
        for i in range(2, len(nums) + 1):
            # 第i位 无法独立编码也无法组合编码
            if nums[i-1] == '0' and (nums[i - 2] == '0' or nums[i - 2] > '2'):
                return 0
            # 第i位 只能跟第i-1位组合编码
            elif nums[i - 1] == '0':
                dp[i] = dp[i - 2]
            # 第i位 只能单独编码
            elif nums[i - 2] == '0' or nums[i - 2] > '2' or nums[i - 2] == '2' and nums[i - 1] > '6':
                dp[i] = dp[i - 1]
            # 第i位 单独编码 组合编码都可以
            else:
                dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]
```

### BM70 兑换零钱一

思路：动态规划

```python
class Solution:
    def minMoney(self , coins: List[int], aim: int) -> int:
        dp = [float('inf')] * (aim + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin, aim + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[aim] if dp[aim] != float('inf') else -1
```

### BM71 最长递增子序列

思路：动态规划

难点在于如何找到arr[i]前面的递增序列，可以用两层循环来寻找

```python
class Solution:
    def LIS(self , arr: List[int]) -> int:
        if not arr: return 0
        # base case
        dp = [1] * len(arr)
        for i in range(len(arr)):
            for j in range(i):
                if arr[i] > arr[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

时间复杂度：O(N^2)

思路二：用二分法优化

解题思路有点像蜘蛛纸牌，只把点数小的数放在点数大的数下，若现有牌堆没有，就新建一堆，最后牌堆数就是最长子序列数

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0
        n = len(nums)
        top = [0] * n
        piles = 0    # 牌堆初始化 0
        for i in range(n):
            poker = nums[i]
            
            left, right = 0, piles
            while left < right:
                mid = (left + right) // 2
                if top[mid] > poker:
                    right = mid
                elif top[mid] < poker:
                    left = mid + 1
                else:
                    right = mid
            
            # 没找到合适的牌堆，新建一堆
            if left == piles : piles += 1
            top[left] = poker
        return piles
```

时间复杂度：O(NlogN) 遍历每个数要O(N)，每个数找到自己要放的位置需要O(logN)

### BM72 最大子数组和

思路：动态规划

dp[i] 定义为以 nums[i] 为结尾的最大子数组和

那么 dp[i] 有两种选择，要么是与前面的数组组成更大和的子数组，要么是自己单独成一个子数组。

```python
class Solution:
    def FindGreatestSumOfSubArray(self , nums: List[int]) -> int:
        n = len(nums)
        if n == 0: return 0
        dp = [0] * n
        for i in range(n):
            dp[i] = max(nums[i], nums[i] + dp[i - 1])
        return max(dp)
```

空间复杂度优化：

```python
class Solution:
    def FindGreatestSumOfSubArray(self , nums: List[int]) -> int:
        tempsum = 0
        ans = nums[0]
        for num in nums:
            tempsum = max(tempsum + num, num)
            ans = max(tempsum, ans)
        return ans
```

### BM73 最长回文子串

思路：动态规划

```python
class Solution:
    def getLongestPalindrome(self , A: str) -> int:
        n = len(A)
        if n < 2:
            return n
        maxLen = 1
        dp = [[0] * n for _ in range(n)]
        for right in range(1, n):
            for left in range(right + 1):
                # 如果两个字符不同，肯定不能构成回文子串
                if A[left] != A[right]:
                    continue
                # 下面是A[left] == A[right] 的情况
                # 字符相同的情况下
                # 如果只有一个字符，肯定是
                if right == left:
                    dp[left][right] = True
                # 类似于 'aa' 和 'aba'，肯定是
                elif right - left <= 2:
                    dp[left][right] = True
                # 需要判断中间部分是否是回文子串
                else:
                    dp[left][right] = dp[left + 1][right - 1]
                if dp[left][right] and (right - left + 1 > maxLen):
                    maxLen = right - left + 1
        return maxLen
```

### BM75 编辑距离

思路：动态规划

```python
class Solution:
    def editDistance(self , str1: str, str2: str) -> int:
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + 1

        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + 1
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[m][n]
```

### BM76 正则表达式匹配

思路：动态规划

```python
class Solution:
    def match(self , s: str, p: str) -> bool:
        m, n = len(s) + 1, len(p) + 1
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True
        for j in range(2, n, 2):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
        for i in range(1, m):
            for j in range(1, n):
                if p[j - 1] == '*':
                    if dp[i][j - 2]: dp[i][j] = True
                    elif dp[i - 1][j] and s[i - 1] == p[j - 2]: dp[i][j] = True
                    elif dp[i - 1][j] and p[j - 2] == '.': dp[i][j] = True
                else:
                    if dp[i - 1][j - 1] and s[i - 1] == p[j - 1]: dp[i][j] = True
                    elif dp[i - 1][j - 1] and p[j - 1] == '.': dp[i][j] = True
        return dp[-1][-1]
```

### BM77 最长括号子串

思路：栈 + 动态规划

```python
class Solution:
    def longestValidParentheses(self , s: str) -> int:
        stack = []
        # dp[i] 的定义：记录以 s[i-1] 结尾的最长合法括号子串长度
        dp = [0] * (len(s) + 1)
        for i in range(len(s)):
            if s[i] == '(':
                # 遇到左括号，记录索引
                stack.append(i)
                # 左括号不可能是合法括号子串的结尾
                dp[i + 1] = 0
            else:
                # 遇到右括号
                if stack:
                    # 配对的左括号对应索引
                    leftindex = stack.pop()
                    # 以这个右括号结尾的最长子串长度
                    length = i - leftindex + 1 + dp[leftindex]
                    dp[i + 1] = length
                else:
                    # 没有配对的左括号
                    dp[i + 1] = 0
        return max(dp)
```

思路二：栈

```python
class Solution:
    def longestValidParentheses(self , s: str) -> int:
        if not s: return 0
        n = len(s)
        stack = [-1]
        ans = 0
        for i in range(n):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    ans = max(ans, i - stack[-1])
        return ans
```

### BM78 打家劫舍一

思路：动态规划 

- 备忘录 + 自顶向下

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        memo = [-1] * (len(nums) + 2)
		# 返回 dp[start..] 能抢到的最大值
        def dp(nums, start):
            if start >= len(nums):
                return 0
            if memo[start] != -1:
                return memo[start]
            memo[start] = max(dp(nums, start + 1), nums[start] + dp(nums, start + 2))
            return memo[start]
        # 强盗从第 0 间房子开始抢劫
        return dp(nums, 0)
```

- 自底向上

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        # 从第 x 间开始抢 最多可以抢到的钱
        dp = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            dp[i] = max(dp[i + 1], nums[i] + dp[i + 2])
        return dp[0]
```

- 空间复杂度优化

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        # 从第 x 间开始抢 最多可以抢到的钱
        dp_0 = 0
        dp_1, dp_2 = 0, 0
        for i in range(n - 1, -1, -1):
            dp_0 = max(dp_1, nums[i] + dp_2)
            dp_2 = dp_1
            dp_1 = dp_0
        return dp_0
```

### BM79 打家劫舍二

思路：动态规划

环形数组的话，第一间和最后一间只有三种情况，都不抢；只抢第一间；只抢最后一间。其中第一种肯定是比后两种情况小的，所以只考虑后两种情况即可。

```python
class Solution:
    def rob(self , nums: List[int]) -> int:
        n = len(nums)
        if n == 1: return nums[0]
        
        # 计算区间[start, end] 的最优结果
        def robRange(nums, start, end):
            dp_i, dp_i_1, dp_i_2 = 0, 0, 0
            for i in range(end, start - 1, -1):
                dp_i = max(dp_i_1, dp_i_2 + nums[i])
                dp_i_2 = dp_i_1
                dp_i_1 = dp_i
            return dp_i
        return max(robRange(nums, 0, n - 2), robRange(nums, 1, n - 1))
```

**扩展： LeetCode 337打家劫舍三**

思路：动态规划 + 后序遍历

难点在于，如果从上往下看这棵树，是无法在遍历到某一个节点时决定【偷或不偷】这个节点的收益的。
因此，我们要想办法从下往上看，于是就想到了后序遍历。

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        def dfs(root):
            if not root: 
                return 0, 0
            left_steal, left_nosteal = dfs(root.left)
            right_steal, right_nosteal = dfs(root.right)
            # 当前节点偷
            steal = root.val + left_nosteal + right_nosteal
            # 当前节点不偷
            nosteal = max(left_steal, left_nosteal) + max(right_steal, right_nosteal)
            return steal, nosteal
        return max(dfs(root))
```

### BM80 买卖股票的最佳时机 一

思路：动态规划

k = 1 时的情况

- 标准模板

```python
class Solution:
    def maxProfit(self , prices: List[int]) -> int:
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        for i in range(n):
            if i - 1 == -1:
                dp[i][0] = 0
                dp[i][1] = -prices[i]
                continue
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
                dp[i][1] = max(dp[i - 1][1], - prices[i])
        return dp[n - 1][0]
```

- 空间复杂度优化

```python
class Solution:
    def maxProfit(self , prices: List[int]) -> int:
        n = len(prices)
        dp_0, dp_1 = 0, float('-inf')
        for i in range(n):
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, - prices[i])
        return dp_0
```

### BM81 买卖股票的最佳时机 二

思路：动态规划

k 为无穷大时的情况

- 标准模板

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        for i in range(n):
            if i == 0:
                dp[i][0] = 0
                dp[i][1] = -prices[i]
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[n - 1][0]
```

- 空间复杂度优化

```python
class Solution:
    def maxProfit(self , prices: List[int]) -> int:
        n = len(prices)
        dp_0, dp_1 = 0, float('-inf')
        for i in range(n):
            temp = dp_0
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, temp - prices[i])
        return dp_0
```

### BM82 买卖股票的最佳时机 三

思路：动态规划

k = 2 的情况

- 标准模板

注意要加入 k 的状态循环，与之前的 k = 1 与 k = 无穷 情况不一样

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_k = 2
        n = len(prices)
        dp = [[[0] * 2 for _ in range(max_k + 1)] for _ in range(n)]
        for i in range(n):
            for k in range(max_k, 0, -1):
                if i == 0:
                    dp[i][k][0] = 0
                    dp[i][k][1] = -prices[i]
                else:
                    dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
                    dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
        return dp[n - 1][max_k][0]
```

- 空间复杂度优化

```python
class Solution:
    def maxProfit(self , prices: List[int]) -> int:
        dp_i10, dp_i11, dp_i20, dp_i21 = 0, float('-inf'), 0, float('-inf')
        for price in prices:
            dp_i20 = max(dp_i20, dp_i21 + price)
            dp_i21 = max(dp_i21, dp_i10 - price)
            dp_i10 = max(dp_i10, dp_i11 + price)
            dp_i11 = max(dp_i11, -price)
        return dp_i20
```

### 买卖股票最佳时机 四 LeetCode 188题

k 取 无穷的情况

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        k = min(k, n // 2)
        if n <= 0:
            return 0
        dp = [[[0] * 2 for _ in range(k + 1)] for _ in range(n)]
        for i in range(n):
            dp[i][0][0] = 0
            dp[i][0][1] = float('-inf')
        for i in range(n):
            for j in range(k, 0, -1):
                if i == 0:
                    dp[i][j][0] = 0
                    dp[i][j][1] = -prices[i]
                else:
                    dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                    dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])
        return dp[n - 1][k][0]
```

### **买卖股票最佳时机带手续费 LeetCode 714题**

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        for i in range(n):
            if i == 0:
                dp[i][0] = 0
                dp[i][1] = -prices[i] - fee
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
                dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee)
        return dp[n - 1][0]
```

- 空间复杂度优化

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        dp_0, dp_1 = 0, float('-inf')
        for i in range(n):
            temp = dp_0
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, temp - prices[i] - fee)
        return dp_0
```

### 买卖股票最佳时机手续费和冷静期 LeetCode 309题

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        for i in range(n):
            if i == 0:
                dp[i][0] = 0
                dp[i][1] = -prices[i]
                continue
            elif i - 2 < 0:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
                dp[i][1] = max(dp[i - 1][1], -prices[i])
            else:
                dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
                dp[i][1] = max(dp[i - 1][1], dp[i - 2][0] - prices[i])
        return dp[n - 1][0]
```

[一个方法团灭 LeetCode 股票买卖问题 :: labuladong的算法小抄 (gitee.io)](https://labuladong.gitee.io/algo/3/27/96/)

## 字符串

### BM83 字符串变形

```python
class Solution:
    def trans(self , s: str, n: int) -> str:
        s_l = s.split(' ')
        ans = []
        for word in s_l:
            temp = []
            for i in range(len(word)):
                if word[i].islower():
                    temp.append(word[i].upper())
                else:
                    temp.append(word[i].lower())
            ans.append(''.join(temp))
        return ' '.join(ans[::-1])
```

注：

python 对字符串操作的函数中，有 `swapcase()`， 可以直接反转大小写不需要判断

```python
class Solution:
    def trans(self , s: str, n: int) -> str:
        s_l = s.split(' ')
        s_l = s_l[::-1]
        ans = ''
        for word in s_l:
            word = word.swapcase()
            ans += word
            ans += ' '
        return ans[0:len(ans) - 1]
```

### BM84 最长公共前缀

思路：sort()

```python
class Solution:
    def longestCommonPrefix(self , strs: List[str]) -> str:
        if not strs:
            return ""
        strs.sort()
        a, b = strs[0], strs[-1]
        for i in range(len(a)):
            if a[i] != b[i]:
                return a[:i]
        return a
```

### BM85 验证IP地址

```python
class Solution:
    def solve(self , IP: str) -> str:
        if '.' in IP:
            for ip in IP.split('.'):
                if (ip.isdigit() is False) or ip == '' or ip[0] == '0' or (not 0 <= int(ip) <= 255) or IP[-1]=='.':
                    return 'Neither'
            return 'IPv4'
        if ':' in IP:
            for ip in IP.split(':'):
                if ip == '' or (len(ip) > 1 and len(ip) == ip.count('0')) or len(ip) > 4 or IP[-1]==':':
                    return 'Neither'
                try:
                    int(ip, 16)
                except:
                    return 'Neither'
            return 'IPv6'
```

### BM86 大数加法

模拟

```python
class Solution:
    def solve(self , s: str, t: str) -> str:
        m, n = len(s), len(t)
        maxlen = max(m, n)
        res = ''
        carry = 0
        for i in range(maxlen):
            x = int(s[m - i - 1]) if i < m else 0
            y = int(t[n - i - 1]) if i < n else 0
            sum = x + y + carry
            res += str(sum % 10)
            carry = sum // 10
        if carry: res += '1'
        return res[::-1]
```

## 双指针

### BM87 合并两个有序数组

```python
class Solution:
    def merge(self , A, m, B, n):
        while m - 1 >= 0 and n - 1 >= 0:
            if A[m - 1] >= B[n - 1]:
                A[m + n - 1] = A[m - 1]
                m -= 1
            else:
                A[m + n - 1] = B[n - 1]
                n -= 1
        if n >= 1:
            A[m: m + n] = B[:n]
```

### BM88 判断回文字符串

```python
class Solution:
    def judge(self , s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] == s[j]: 
                i += 1
                j -= 1
            else:
                return False
        return True
```

### BM89 合并区间

```python
class Solution:
    def merge(self , intervals: List[Interval]) -> List[Interval]:
        if not intervals:
            return []
        # 按照区间左边界的值进行排序
        intervals = sorted(intervals, key=lambda interval: interval.start)
        ans = []
        ans.append(intervals[0])
        for i in range(1, len(intervals)):
            # 用 ans 的最后一个区间， 前面的都有序了
            last = ans[-1]
            # 当前区间的左右边界
            cur_left, cur_right = intervals[i].start, intervals[i].end
            # 如果没有重合，把区间加入 ans
            if last.end < cur_left:
                ans.append(intervals[i])
            else:
                # 重合了， 比较右边界，取较大的
                last.end = max(cur_right, last.end)
        return ans
```

### BM90 最小覆盖子串

思路：滑动窗口法

```python
class Solution:
    def minWindow(self , S: str, T: str) -> str:
        import collections
        need = collections.defaultdict(int)
        window = collections.defaultdict(int)
        for c in T:
            need[c] += 1
        left, right = 0, 0
        valid = 0
        start = 0
        lens = 10 ** 6
        while right < len(S):
            c = S[right]
            right += 1
            if c in need:
                window[c] += 1
                if window[c] == need[c]:
                    valid += 1
            while valid == len(need):
                if right - left < lens:
                    start = left
                    lens = right - left
                d = S[left]
                left += 1
                if d in need:
                    if window[d] == need[d]:
                        valid -= 1
                    window[d] -= 1
        return S[start: start + lens] if lens != 10 ** 6 else ''
```

### BM91 反转字符串

```python
class Solution:
    def solve(self , s: str) -> str:
        j = len(s) - 1
        ans = ''
        while j >= 0:
            ans += s[j]
            j -= 1
        return ans
```

### BM92 最长无重复子数组

思路：滑动窗口法

```python
class Solution:
    def maxLength(self , arr: List[int]) -> int:
        import collections
        window = collections.defaultdict(int)
        left, right = 0, 0
        ans = 0
        
        while right < len(arr):
            c = arr[right]
            right += 1
            window[c] += 1
            
            while window[c] > 1:
                d = arr[left]
                left += 1
                window[d] -= 1
                
            ans = max(ans, right - left)
        return ans
```

### BM93 盛水最多的容器

思路：双指针法

```python
class Solution:
    def maxArea(self , height: List[int]) -> int:
        left, right = 0, len(height) - 1
        ans = 0
        while left < right:
            ans = max(ans, (right - left) * min(height[left], height[right]))
            if height[left] < height[right]:
                 left += 1
            else:
                right -= 1
        return ans
```

### BM94 接雨水

思路一：暴力解法

时间复杂度：O(N^2)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        ans = 0
        for i in range(n):
            l_max, r_max = 0, 0
            for j in range(i, n):
                r_max = max(r_max, height[j])
            for j in range(i, -1, -1):
                l_max = max(l_max, height[j])
            ans += min(l_max, r_max) - height[i]
        return ans
```

思路二：备忘录优化

时间复杂度：O(N)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height: return 0
        n = len(height)
        ans = 0
        l_max, r_max = [0] * n, [0] * n
        # 初始化 base case
        l_max[0] = height[0]
        r_max[n - 1] = height[n -1]
        # 从左向右计算l_max
        for i in range(n):
            l_max[i] = max(l_max[i - 1], height[i])
        # 从右向左计算r_max
        for i in range(n - 2, -1, -1):
            r_max[i] = max(r_max[i + 1], height[i])
        for i in range(1, n - 1):
            ans += min(l_max[i], r_max[i]) - height[i]
        return ans
```

思路三：双指针法

空间复杂度：O(1)

`l_max` 是 `height[0..left]` 中最高柱子的高度，`r_max` 是 `height[right..end]` 的最高柱子的高度。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        
        if not height or len(height) <= 2: return 0
        
        left, right = 0, len(height) - 1
        l_max, r_max = 0, 0
        ans = 0
        
        while left < right:
            l_max = max(l_max, height[left])
            r_max = max(r_max, height[right])
            if l_max < r_max:
                ans += l_max - height[left]
                left += 1
            else:
                ans += r_max - height[right]
                right -= 1
        return ans
```

## 贪心

### BM95 分糖果问题

思路：两次遍历

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        left = [1 for _ in range(len(ratings))]
        right = left[:]
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                left[i] = left[i - 1] + 1
        count = left[-1]
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                right[i] = right[i + 1] + 1
            count += max(left[i], right[i])
        return count
```

### BM96 主持人调度

思路：贪心 + 优先级队列

堆中只存活动结束时间即可，每次与下一个活动开始时间相比，不重合则pop，重合则加入堆中，最后返回堆的大小即可。

```python
class Solution:
    def minmumNumberOfHost(self , n: int, startEnd: List[List[int]]) -> int:
        import heapq
        startEnd.sort()
        queue = []
        heapq.heappush(queue, float('-inf'))
        for i in  range(n):
            if queue[0] <= startEnd[i][0]:
                heapq.heappop(queue)
            heapq.heappush(queue, startEnd[i][1])
        return len(queue)
```

## 模拟

### BM97 旋转数组

```python
class Solution:
    def solve(self , n: int, m: int, a: List[int]) -> List[int]:
        if n == 0 or m == 0: return a
        m = m % n
        res = a[::-1]
        self.reverse(res, 0, m - 1)
        self.reverse(res, m, n - 1)
        return res
    def reverse(self, res, start, end):
        while start < end:
            temp = res[start]
            res[start] = res[end]
            res[end] = temp
            start += 1
            end -= 1
```

### BM98 螺旋数组

四条边界逐渐收缩

```python
class Solution:
    def spiralOrder(self , matrix: List[List[int]]) -> List[int]:
        if not matrix: return []
        m, n = len(matrix), len(matrix[0])
        upper_bound, lower_bound = 0, m - 1
        left_bound, right_bound = 0, n - 1
        res = []
        while len(res) < m * n:
            # 在顶部从左往右遍历
            if upper_bound <= lower_bound :
                for i in range(left_bound, right_bound +1):
                    res.append(matrix[upper_bound][i])
                # 上边界收缩
                upper_bound += 1
            # 在右侧从上往下遍历
            if left_bound <= right_bound:
                for i in range(upper_bound, lower_bound + 1):
                    res.append(matrix[i][right_bound])
				# 右边界收缩
                right_bound -= 1
            # 在底部从右往左遍历
            if upper_bound <= lower_bound:
                for i in range(right_bound, left_bound - 1, -1):
                    res.append(matrix[lower_bound][i])
                # 下边界收缩
                lower_bound -= 1
            # 在左侧从下往上遍历
            if left_bound <= right_bound:
                for i in range(lower_bound, upper_bound - 1, -1):
                    res.append(matrix[i][left_bound])
                # 左边界收缩
                left_bound += 1
        return res
```

**注：LeetCode 59题 螺旋矩阵二**

方法同上，稍加改变

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for _ in range(n)]
        upper_bound, lower_bound = 0, n - 1
        left_bound, right_bound = 0, n - 1
        num = 1

        while num <= n * n:
            if upper_bound <= lower_bound:
                for i in range(left_bound, right_bound + 1):
                    matrix[upper_bound][i] = num
                    num += 1
                upper_bound += 1
            if left_bound <= right_bound:
                for i in range(upper_bound, lower_bound + 1):
                    matrix[i][right_bound] = num
                    num += 1
                right_bound -= 1
            if upper_bound <= lower_bound:
                for i in range(right_bound, left_bound - 1, -1):
                    matrix[lower_bound][i] = num
                    num += 1
                lower_bound -= 1
            if left_bound <= right_bound:
                for i in range(lower_bound, upper_bound - 1, -1):
                    matrix[i][left_bound] = num
                    num += 1
                left_bound += 1
        return matrix
```

**注2：LeetCode 885题 螺旋矩阵三**

同样的思想，但是是从中间开始发散

```python
class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        res = []
        around = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        left, right, upper, lower = cStart - 1, cStart + 1, rStart - 1, rStart + 1
        x, y, num, Dir = rStart, cStart, 1, 0 # (x, y)为当前节点，num为当前查找的数字，Dir是当前方向
        while num <= rows * cols:
            if 0 <= x < rows and 0 <= y < cols:  # (x, y)在矩阵中
                res.append([x, y])
                num += 1
            if Dir == 0 and y == right:    # 到右边界后
                Dir += 1
                right += 1
            elif Dir == 1 and x == lower:    # 到下边界后
                Dir += 1
                lower += 1
            elif Dir == 2 and y == left:    # 到左边界后
                Dir += 1
                left -= 1
            elif Dir == 3 and x == upper:    # 到上边界后
                Dir = 0
                upper -= 1
            x += around[Dir][0]
            y += around[Dir][1]
        return res
```

### BM99 顺时针旋转矩阵

先沿对角线镜像一下矩阵，然后在每行反转一下矩阵，就得到了结果

```python
class Solution:
    def rotateMatrix(self , mat: List[List[int]], n: int) -> List[List[int]]:
        # 沿对角线镜像
        for i in range(n):
            for j in range(i, n):
                mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
        # 每行反转矩阵
        for i in range(n):
            mat[i] = mat[i][::-1]
        return mat
```

### BM100 LRU缓存结构

思路：双向链表 + 字典

**LeetCode 146题**

```python
# 双向链表类
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class Solution:

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

    def set(self, key: int, value: int) -> None:
        if key not in self.cache:
            # 如果key不存在，创建一个新节点
            node = DLinkedNode(key, value)
            # 添加进哈希表, !! 是把节点加入哈希表，不是只把value加入哈希表！！
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
```

### BM101 LFU缓存结构

与LRU相似，使用哈希 + 双向链表完成

```python
class Node:
    def __init__(self, key=-1, value=-1):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
        self.freq = 1

from collections import defaultdict
class DLinkedNode:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.size += 1
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
    
    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node  
        
class Solution:
    def __init__(self):
        self.capacity = 0
        self.size = 0
        self.cache = {}
        self.freq = defaultdict(DLinkedNode)
        self.minfreq = 0
    
    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self.freq[node.freq].removeNode(node)
            if self.minfreq == node.freq and self.freq[node.freq].size == 0:
                self.minfreq += 1
            node.freq += 1
            self.freq[node.freq].addToHead(node)
            return node.value
        return -1
    
    def put(self, key, value):
        if self.capacity == 0: return
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self.freq[node.freq].removeNode(node)
            if self.minfreq == node.freq and self.freq[node.freq].size == 0:
                self.minfreq += 1
            node.freq += 1
            self.freq[node.freq].addToHead(node)
        else:
            self.size += 1
            if self.size > self.capacity:
                node = self.freq[self.minfreq].removeTail()
                self.cache.pop(node.key)
                self.size -= 1
            node = Node(key, value)
            self.cache[key] = node
            self.freq[1].addToHead(node)
            self.minfreq = 1
    
        
    def LFU(self , operators: List[List[int]], k: int) -> List[int]:
        self.capacity = k
        n = len(operators)
        res = []
        for i in range(n):
            if operators[i][0] == 1:
                self.put(operators[i][1], operators[i][2])
            else:
                res.append(self.get(operators[i][1]))
        return res
```

**LeetCode 460题**

```python
class Node:
    def __init__(self, key=-1, value=-1):
        self.key = key
        self.value = value
        self.freq = 1
        self.prev = None
        self.next = None

class DLinkedNode:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
        self.size += 1
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1
    
    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

from collections import defaultdict
class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.cache = {}
        self.freq = defaultdict(DLinkedNode)
        self.minfreq = 0

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.freq[node.freq].removeNode(node)
            if self.minfreq == node.freq and self.freq[node.freq].size == 0:
                self.minfreq += 1
            node.freq += 1
            self.freq[node.freq].addToHead(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0: return 
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self.freq[node.freq].removeNode(node)
            if self.minfreq == node.freq and self.freq[node.freq].size == 0:
                self.minfreq += 1
            node.freq += 1
            self.freq[node.freq].addToHead(node)
        else:
            self.size += 1
            if self.size > self.capacity:
                node = self.freq[self.minfreq].removeTail()
                self.cache.pop(node.key)
                self.size -= 1
            node = Node(key, value)
            self.cache[key] = node
            self.freq[1].addToHead(node)
            self.minfreq = 1
```

