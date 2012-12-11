
/// A priority queue implemented with a binary heap
use core::cmp::Ord;

pub struct PriorityQueue <T: Copy Ord>{
    priv data: ~[T],
}

impl <T: Copy Ord> PriorityQueue<T> {
    /// Returns the greatest item in the queue - fails if empty
    pure fn top(&self) -> T { self.data[0] }

    /// Returns the greatest item in the queue - None if empty
    pure fn maybe_top(&self) -> Option<T> {
        if self.is_empty() { None } else { Some(self.top()) }
    }

    /// Returns the length of the queue
    pure fn len(&self) -> uint { self.data.len() }

    /// Returns true if a queue contains no elements
    pure fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Returns true if a queue contains some elements
    pure fn is_not_empty(&self) -> bool { self.data.is_not_empty() }

    /// Returns the number of elements the queue can hold without reallocating
    pure fn capacity(&self) -> uint { vec::capacity(&self.data) }

    fn reserve(&mut self, n: uint) { vec::reserve(&mut self.data, n) }

    fn reserve_at_least(&mut self, n: uint) {
        vec::reserve_at_least(&mut self.data, n)
    }

    /// Drop all items from the queue
    fn clear(&mut self) { self.data.truncate(0) }

    /// Pop the greatest item from the queue - fails if empty
    fn pop(&mut self) -> T {
        let mut item = self.data.pop();
        if self.is_not_empty() { item <-> self.data[0]; self.siftup(0); }
        item
    }

    /// Pop the greatest item from the queue - None if empty
    fn maybe_pop(&mut self) -> Option<T> {
        if self.is_empty() { None } else { Some(self.pop()) }
    }

    /// Push an item onto the queue
    fn push(&mut self, item: T) {
        self.data.push(item);
        self.siftdown(0, self.len() - 1);
    }

    /// Optimized version of a push followed by a pop
    fn push_pop(&mut self, item: T) -> T {
        let mut item = item;
        if self.is_not_empty() && self.data[0] > item {
            item <-> self.data[0];
            self.siftup(0);
        }
        item
    }

    /// Optimized version of a pop followed by a push - fails if empty
    fn replace(&mut self, item: T) -> T {
        let mut item = item;
        item <-> self.data[0];
        self.siftup(0);
        item
    }

    /// Consume the PriorityQueue and return the underlying vector
    pure fn to_vec(self) -> ~[T] { let PriorityQueue{data: v} = self; v }

    /// Consume the PriorityQueue and return a vector in sorted (ascending) order
    pure fn to_sorted_vec(self) -> ~[T] {
        let mut q = self;
        let mut end = q.len() - 1;
        while end > 0 {
            q.data[end] <-> q.data[0];
            end -= 1;
            unsafe { q.siftup_range(0, end) } // purity-checking workaround
        }
        q.to_vec()
    }

    static pub pure fn from_vec(xs: ~[T]) -> PriorityQueue<T> {
        let mut q = PriorityQueue{data: xs,};
        let mut n = q.len() / 2;
        while n > 0 {
            n -= 1;
            unsafe { q.siftup(n) }; // purity-checking workaround
        }
        q
    }

    priv fn siftdown(&mut self, startpos: uint, pos: uint) {
        let mut pos = pos;
        let newitem = self.data[pos];

        while pos > startpos {
            let parentpos = (pos - 1) >> 1;
            let parent = self.data[parentpos];
            if newitem > parent {
                self.data[pos] = parent;
                pos = parentpos;
                loop
            }
            break
        }
        self.data[pos] = newitem;
    }

    priv fn siftup_range(&mut self, pos: uint, endpos: uint) {
        let mut pos = pos;
        let startpos = pos;
        let newitem = self.data[pos];

        let mut childpos = 2 * pos + 1;
        while childpos < endpos {
            let rightpos = childpos + 1;
            if rightpos < endpos &&
                   !(self.data[childpos] > self.data[rightpos]) {
                childpos = rightpos;
            }
            self.data[pos] = self.data[childpos];
            pos = childpos;
            childpos = 2 * pos + 1;
        }
        self.data[pos] = newitem;
        self.siftdown(startpos, pos);
    }

    priv fn siftup(&mut self, pos: uint) {
        self.siftup_range(pos, self.len());
    }
}

#[cfg(test)]
mod tests {
    use sort::merge_sort;
    use core::cmp::le;
    use PriorityQueue::from_vec;

    #[test]
    fn test_top_and_pop() {
        let data = ~[2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
        let mut sorted = merge_sort(data, le);
        let mut heap = from_vec(data);
        while heap.is_not_empty() {
            assert heap.top() == sorted.last();
            assert heap.pop() == sorted.pop();
        }
    }

    #[test]
    fn test_push() {
        let mut heap = from_vec(~[2, 4, 9]);
        assert heap.len() == 3;
        assert heap.top() == 9;
        heap.push(11);
        assert heap.len() == 4;
        assert heap.top() == 11;
        heap.push(5);
        assert heap.len() == 5;
        assert heap.top() == 11;
        heap.push(27);
        assert heap.len() == 6;
        assert heap.top() == 27;
        heap.push(3);
        assert heap.len() == 7;
        assert heap.top() == 27;
        heap.push(103);
        assert heap.len() == 8;
        assert heap.top() == 103;
    }

    #[test]
    fn test_push_pop() {
        let mut heap = from_vec(~[5, 5, 2, 1, 3]);
        assert heap.len() == 5;
        assert heap.push_pop(6) == 6;
        assert heap.len() == 5;
        assert heap.push_pop(0) == 5;
        assert heap.len() == 5;
        assert heap.push_pop(4) == 5;
        assert heap.len() == 5;
        assert heap.push_pop(1) == 4;
        assert heap.len() == 5;
    }

    #[test]
    fn test_replace() {
        let mut heap = from_vec(~[5, 5, 2, 1, 3]);
        assert heap.len() == 5;
        assert heap.replace(6) == 5;
        assert heap.len() == 5;
        assert heap.replace(0) == 6;
        assert heap.len() == 5;
        assert heap.replace(4) == 5;
        assert heap.len() == 5;
        assert heap.replace(1) == 4;
        assert heap.len() == 5;
    }

    #[test]
    fn test_to_sorted_vec() {
        let data = ~[2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
        assert from_vec(data).to_sorted_vec() == merge_sort(data, le);
    }

    #[test]
    #[should_fail]
    fn test_empty_pop() { let mut heap = from_vec::<int>(~[]); heap.pop(); }

    #[test]
    fn test_empty_maybe_pop() {
        let mut heap = from_vec::<int>(~[]);
        assert heap.maybe_pop().is_none();
    }

    #[test]
    #[should_fail]
    fn test_empty_top() { from_vec::<int>(~[]).top(); }

    #[test]
    fn test_empty_maybe_top() {
        assert from_vec::<int>(~[]).maybe_top().is_none();
    }

    #[test]
    #[should_fail]
    fn test_empty_replace() {
        let mut heap = from_vec::<int>(~[]);
        heap.replace(5);
    }

    #[test]
    fn test_to_vec() {
        let data = ~[1, 3, 5, 7, 9, 2, 4, 6, 8, 0];
        let heap = from_vec(copy data);
        assert merge_sort(heap.to_vec(), le) == merge_sort(data, le);
    }
}
