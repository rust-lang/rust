// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A priority queue implemented with a binary heap

#[allow(missing_doc)];

use std::clone::Clone;
use std::unstable::intrinsics::{move_val_init, init};
use std::util::{replace, swap};
use std::vec;
use std::iterator::FromIterator;

/// A priority queue implemented with a binary heap
#[deriving(Clone)]
pub struct PriorityQueue<T> {
    priv data: ~[T],
}

impl<T:Ord> Container for PriorityQueue<T> {
    /// Returns the length of the queue
    fn len(&self) -> uint { self.data.len() }
}

impl<T:Ord> Mutable for PriorityQueue<T> {
    /// Drop all items from the queue
    fn clear(&mut self) { self.data.truncate(0) }
}

impl<T:Ord> PriorityQueue<T> {
    /// An iterator visiting all values in underlying vector, in
    /// arbitrary order.
    pub fn iter<'a>(&'a self) -> PriorityQueueIterator<'a, T> {
        PriorityQueueIterator { iter: self.data.iter() }
    }

    /// Returns the greatest item in the queue - fails if empty
    pub fn top<'a>(&'a self) -> &'a T { &self.data[0] }

    /// Returns the greatest item in the queue - None if empty
    pub fn maybe_top<'a>(&'a self) -> Option<&'a T> {
        if self.is_empty() { None } else { Some(self.top()) }
    }

    /// Returns the number of elements the queue can hold without reallocating
    pub fn capacity(&self) -> uint { self.data.capacity() }

    pub fn reserve(&mut self, n: uint) { self.data.reserve(n) }

    pub fn reserve_at_least(&mut self, n: uint) {
        self.data.reserve_at_least(n)
    }

    /// Pop the greatest item from the queue - fails if empty
    pub fn pop(&mut self) -> T {
        let mut item = self.data.pop();
        if !self.is_empty() {
            swap(&mut item, &mut self.data[0]);
            self.siftdown(0);
        }
        item
    }

    /// Pop the greatest item from the queue - None if empty
    pub fn maybe_pop(&mut self) -> Option<T> {
        if self.is_empty() { None } else { Some(self.pop()) }
    }

    /// Push an item onto the queue
    pub fn push(&mut self, item: T) {
        self.data.push(item);
        let new_len = self.len() - 1;
        self.siftup(0, new_len);
    }

    /// Optimized version of a push followed by a pop
    pub fn push_pop(&mut self, mut item: T) -> T {
        if !self.is_empty() && self.data[0] > item {
            swap(&mut item, &mut self.data[0]);
            self.siftdown(0);
        }
        item
    }

    /// Optimized version of a pop followed by a push - fails if empty
    pub fn replace(&mut self, mut item: T) -> T {
        swap(&mut item, &mut self.data[0]);
        self.siftdown(0);
        item
    }

    /// Consume the PriorityQueue and return the underlying vector
    pub fn to_vec(self) -> ~[T] { let PriorityQueue{data: v} = self; v }

    /// Consume the PriorityQueue and return a vector in sorted
    /// (ascending) order
    pub fn to_sorted_vec(self) -> ~[T] {
        let mut q = self;
        let mut end = q.len();
        while end > 1 {
            end -= 1;
            q.data.swap(0, end);
            q.siftdown_range(0, end)
        }
        q.to_vec()
    }

    /// Create an empty PriorityQueue
    pub fn new() -> PriorityQueue<T> { PriorityQueue{data: ~[],} }

    /// Create a PriorityQueue from a vector (heapify)
    pub fn from_vec(xs: ~[T]) -> PriorityQueue<T> {
        let mut q = PriorityQueue{data: xs,};
        let mut n = q.len() / 2;
        while n > 0 {
            n -= 1;
            q.siftdown(n)
        }
        q
    }

    // The implementations of siftup and siftdown use unsafe blocks in
    // order to move an element out of the vector (leaving behind a
    // zeroed element), shift along the others and move it back into the
    // vector over the junk element.  This reduces the constant factor
    // compared to using swaps, which involves twice as many moves.
    fn siftup(&mut self, start: uint, mut pos: uint) {
        unsafe {
            let new = replace(&mut self.data[pos], init());

            while pos > start {
                let parent = (pos - 1) >> 1;
                if new > self.data[parent] {
                    let x = replace(&mut self.data[parent], init());
                    move_val_init(&mut self.data[pos], x);
                    pos = parent;
                    loop
                }
                break
            }
            move_val_init(&mut self.data[pos], new);
        }
    }

    fn siftdown_range(&mut self, mut pos: uint, end: uint) {
        unsafe {
            let start = pos;
            let new = replace(&mut self.data[pos], init());

            let mut child = 2 * pos + 1;
            while child < end {
                let right = child + 1;
                if right < end && !(self.data[child] > self.data[right]) {
                    child = right;
                }
                let x = replace(&mut self.data[child], init());
                move_val_init(&mut self.data[pos], x);
                pos = child;
                child = 2 * pos + 1;
            }

            move_val_init(&mut self.data[pos], new);
            self.siftup(start, pos);
        }
    }

    fn siftdown(&mut self, pos: uint) {
        let len = self.len();
        self.siftdown_range(pos, len);
    }
}

/// PriorityQueue iterator
pub struct PriorityQueueIterator <'self, T> {
    priv iter: vec::VecIterator<'self, T>,
}

impl<'self, T> Iterator<&'self T> for PriorityQueueIterator<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<(&'self T)> { self.iter.next() }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

impl<T: Ord, Iter: Iterator<T>> FromIterator<T, Iter> for PriorityQueue<T> {
    pub fn from_iterator(iter: &mut Iter) -> PriorityQueue<T> {
        let (lower, _) = iter.size_hint();

        let mut q = PriorityQueue::new();
        q.reserve_at_least(lower);

        for iter.advance |elem| {
            q.push(elem);
        }

        q
    }
}

#[cfg(test)]
mod tests {
    use sort::merge_sort;
    use priority_queue::PriorityQueue;

    #[test]
    fn test_iterator() {
        let data = ~[5, 9, 3];
        let iterout = ~[9, 5, 3];
        let pq = PriorityQueue::from_vec(data);
        let mut i = 0;
        for pq.iter().advance |el| {
            assert_eq!(*el, iterout[i]);
            i += 1;
        }
    }

    #[test]
    fn test_top_and_pop() {
        let data = ~[2u, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
        let mut sorted = merge_sort(data, |x, y| x.le(y));
        let mut heap = PriorityQueue::from_vec(data);
        while !heap.is_empty() {
            assert_eq!(heap.top(), sorted.last());
            assert_eq!(heap.pop(), sorted.pop());
        }
    }

    #[test]
    fn test_push() {
        let mut heap = PriorityQueue::from_vec(~[2, 4, 9]);
        assert_eq!(heap.len(), 3);
        assert!(*heap.top() == 9);
        heap.push(11);
        assert_eq!(heap.len(), 4);
        assert!(*heap.top() == 11);
        heap.push(5);
        assert_eq!(heap.len(), 5);
        assert!(*heap.top() == 11);
        heap.push(27);
        assert_eq!(heap.len(), 6);
        assert!(*heap.top() == 27);
        heap.push(3);
        assert_eq!(heap.len(), 7);
        assert!(*heap.top() == 27);
        heap.push(103);
        assert_eq!(heap.len(), 8);
        assert!(*heap.top() == 103);
    }

    #[test]
    fn test_push_unique() {
        let mut heap = PriorityQueue::from_vec(~[~2, ~4, ~9]);
        assert_eq!(heap.len(), 3);
        assert!(*heap.top() == ~9);
        heap.push(~11);
        assert_eq!(heap.len(), 4);
        assert!(*heap.top() == ~11);
        heap.push(~5);
        assert_eq!(heap.len(), 5);
        assert!(*heap.top() == ~11);
        heap.push(~27);
        assert_eq!(heap.len(), 6);
        assert!(*heap.top() == ~27);
        heap.push(~3);
        assert_eq!(heap.len(), 7);
        assert!(*heap.top() == ~27);
        heap.push(~103);
        assert_eq!(heap.len(), 8);
        assert!(*heap.top() == ~103);
    }

    #[test]
    fn test_push_pop() {
        let mut heap = PriorityQueue::from_vec(~[5, 5, 2, 1, 3]);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.push_pop(6), 6);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.push_pop(0), 5);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.push_pop(4), 5);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.push_pop(1), 4);
        assert_eq!(heap.len(), 5);
    }

    #[test]
    fn test_replace() {
        let mut heap = PriorityQueue::from_vec(~[5, 5, 2, 1, 3]);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(6), 5);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(0), 6);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(4), 5);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(1), 4);
        assert_eq!(heap.len(), 5);
    }

    fn check_to_vec(data: ~[int]) {
        let heap = PriorityQueue::from_vec(data.clone());
        assert_eq!(merge_sort(heap.clone().to_vec(), |x, y| x.le(y)),
                   merge_sort(data, |x, y| x.le(y)));
        assert_eq!(heap.to_sorted_vec(), merge_sort(data, |x, y| x.le(y)));
    }

    #[test]
    fn test_to_vec() {
        check_to_vec(~[]);
        check_to_vec(~[5]);
        check_to_vec(~[3, 2]);
        check_to_vec(~[2, 3]);
        check_to_vec(~[5, 1, 2]);
        check_to_vec(~[1, 100, 2, 3]);
        check_to_vec(~[1, 3, 5, 7, 9, 2, 4, 6, 8, 0]);
        check_to_vec(~[2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
        check_to_vec(~[9, 11, 9, 9, 9, 9, 11, 2, 3, 4, 11, 9, 0, 0, 0, 0]);
        check_to_vec(~[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        check_to_vec(~[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        check_to_vec(~[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2]);
        check_to_vec(~[5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1]);
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_empty_pop() { let mut heap = PriorityQueue::new::<int>(); heap.pop(); }

    #[test]
    fn test_empty_maybe_pop() {
        let mut heap = PriorityQueue::new::<int>();
        assert!(heap.maybe_pop().is_none());
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_empty_top() { let empty = PriorityQueue::new::<int>(); empty.top(); }

    #[test]
    fn test_empty_maybe_top() {
        let empty = PriorityQueue::new::<int>();
        assert!(empty.maybe_top().is_none());
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_empty_replace() { let mut heap = PriorityQueue::new(); heap.replace(5); }

    #[test]
    fn test_from_iter() {
        let xs = ~[9u, 8, 7, 6, 5, 4, 3, 2, 1];

        let mut q: PriorityQueue<uint> = xs.rev_iter().transform(|&x| x).collect();

        for xs.iter().advance |&x| {
            assert_eq!(q.pop(), x);
        }
    }
}
