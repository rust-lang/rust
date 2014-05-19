// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A priority queue implemented with a binary heap

#![allow(missing_doc)]

use std::clone::Clone;
use std::mem::{move_val_init, init, replace, swap};
use std::slice;

/// A priority queue implemented with a binary heap
#[deriving(Clone)]
pub struct PriorityQueue<T> {
    data: Vec<T>,
}

impl<T: TotalOrd> Container for PriorityQueue<T> {
    /// Returns the length of the queue
    fn len(&self) -> uint { self.data.len() }
}

impl<T: TotalOrd> Mutable for PriorityQueue<T> {
    /// Drop all items from the queue
    fn clear(&mut self) { self.data.truncate(0) }
}

impl<T: TotalOrd> PriorityQueue<T> {
    /// An iterator visiting all values in underlying vector, in
    /// arbitrary order.
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        Items { iter: self.data.iter() }
    }

    /// Returns the greatest item in a queue or None if it is empty
    pub fn top<'a>(&'a self) -> Option<&'a T> {
        if self.is_empty() { None } else { Some(self.data.get(0)) }
    }

    #[deprecated="renamed to `top`"]
    pub fn maybe_top<'a>(&'a self) -> Option<&'a T> { self.top() }

    /// Returns the number of elements the queue can hold without reallocating
    pub fn capacity(&self) -> uint { self.data.capacity() }

    /// Reserve capacity for exactly n elements in the PriorityQueue.
    /// Do nothing if the capacity is already sufficient.
    pub fn reserve_exact(&mut self, n: uint) { self.data.reserve_exact(n) }

    /// Reserve capacity for at least n elements in the PriorityQueue.
    /// Do nothing if the capacity is already sufficient.
    pub fn reserve(&mut self, n: uint) {
        self.data.reserve(n)
    }

    /// Remove the greatest item from a queue and return it, or `None` if it is
    /// empty.
    pub fn pop(&mut self) -> Option<T> {
        match self.data.pop() {
            None           => { None }
            Some(mut item) => {
                if !self.is_empty() {
                    swap(&mut item, self.data.get_mut(0));
                    self.siftdown(0);
                }
                Some(item)
            }
        }
    }

    #[deprecated="renamed to `pop`"]
    pub fn maybe_pop(&mut self) -> Option<T> { self.pop() }

    /// Push an item onto the queue
    pub fn push(&mut self, item: T) {
        self.data.push(item);
        let new_len = self.len() - 1;
        self.siftup(0, new_len);
    }

    /// Optimized version of a push followed by a pop
    pub fn push_pop(&mut self, mut item: T) -> T {
        if !self.is_empty() && *self.top().unwrap() > item {
            swap(&mut item, self.data.get_mut(0));
            self.siftdown(0);
        }
        item
    }

    /// Optimized version of a pop followed by a push. The push is done
    /// regardless of whether the queue is empty.
    pub fn replace(&mut self, mut item: T) -> Option<T> {
        if !self.is_empty() {
            swap(&mut item, self.data.get_mut(0));
            self.siftdown(0);
            Some(item)
        } else {
            self.push(item);
            None
        }
    }

    #[allow(dead_code)]
    #[deprecated="renamed to `into_vec`"]
    fn to_vec(self) -> Vec<T> { self.into_vec() }

    #[allow(dead_code)]
    #[deprecated="renamed to `into_sorted_vec`"]
    fn to_sorted_vec(self) -> Vec<T> { self.into_sorted_vec() }

    /// Consume the PriorityQueue and return the underlying vector
    pub fn into_vec(self) -> Vec<T> { let PriorityQueue{data: v} = self; v }

    /// Consume the PriorityQueue and return a vector in sorted
    /// (ascending) order
    pub fn into_sorted_vec(self) -> Vec<T> {
        let mut q = self;
        let mut end = q.len();
        while end > 1 {
            end -= 1;
            q.data.as_mut_slice().swap(0, end);
            q.siftdown_range(0, end)
        }
        q.into_vec()
    }

    /// Create an empty PriorityQueue
    pub fn new() -> PriorityQueue<T> { PriorityQueue{data: vec!(),} }

    /// Create an empty PriorityQueue with capacity `capacity`
    pub fn with_capacity(capacity: uint) -> PriorityQueue<T> {
        PriorityQueue { data: Vec::with_capacity(capacity) }
    }

    /// Create a PriorityQueue from a vector (heapify)
    pub fn from_vec(xs: Vec<T>) -> PriorityQueue<T> {
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
            let new = replace(self.data.get_mut(pos), init());

            while pos > start {
                let parent = (pos - 1) >> 1;
                if new > *self.data.get(parent) {
                    let x = replace(self.data.get_mut(parent), init());
                    move_val_init(self.data.get_mut(pos), x);
                    pos = parent;
                    continue
                }
                break
            }
            move_val_init(self.data.get_mut(pos), new);
        }
    }

    fn siftdown_range(&mut self, mut pos: uint, end: uint) {
        unsafe {
            let start = pos;
            let new = replace(self.data.get_mut(pos), init());

            let mut child = 2 * pos + 1;
            while child < end {
                let right = child + 1;
                if right < end && !(*self.data.get(child) > *self.data.get(right)) {
                    child = right;
                }
                let x = replace(self.data.get_mut(child), init());
                move_val_init(self.data.get_mut(pos), x);
                pos = child;
                child = 2 * pos + 1;
            }

            move_val_init(self.data.get_mut(pos), new);
            self.siftup(start, pos);
        }
    }

    fn siftdown(&mut self, pos: uint) {
        let len = self.len();
        self.siftdown_range(pos, len);
    }
}

/// PriorityQueue iterator
pub struct Items <'a, T> {
    iter: slice::Items<'a, T>,
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<(&'a T)> { self.iter.next() }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

impl<T: TotalOrd> FromIterator<T> for PriorityQueue<T> {
    fn from_iter<Iter: Iterator<T>>(iter: Iter) -> PriorityQueue<T> {
        let mut q = PriorityQueue::new();
        q.extend(iter);
        q
    }
}

impl<T: TotalOrd> Extendable<T> for PriorityQueue<T> {
    fn extend<Iter: Iterator<T>>(&mut self, mut iter: Iter) {
        let (lower, _) = iter.size_hint();

        let len = self.capacity();
        self.reserve(len + lower);

        for elem in iter {
            self.push(elem);
        }
    }
}

#[cfg(test)]
mod tests {
    use priority_queue::PriorityQueue;

    #[test]
    fn test_iterator() {
        let data = vec!(5, 9, 3);
        let iterout = [9, 5, 3];
        let pq = PriorityQueue::from_vec(data);
        let mut i = 0;
        for el in pq.iter() {
            assert_eq!(*el, iterout[i]);
            i += 1;
        }
    }

    #[test]
    fn test_top_and_pop() {
        let data = vec!(2u, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1);
        let mut sorted = data.clone();
        sorted.sort();
        let mut heap = PriorityQueue::from_vec(data);
        while !heap.is_empty() {
            assert_eq!(heap.top().unwrap(), sorted.last().unwrap());
            assert_eq!(heap.pop().unwrap(), sorted.pop().unwrap());
        }
    }

    #[test]
    fn test_push() {
        let mut heap = PriorityQueue::from_vec(vec!(2, 4, 9));
        assert_eq!(heap.len(), 3);
        assert!(*heap.top().unwrap() == 9);
        heap.push(11);
        assert_eq!(heap.len(), 4);
        assert!(*heap.top().unwrap() == 11);
        heap.push(5);
        assert_eq!(heap.len(), 5);
        assert!(*heap.top().unwrap() == 11);
        heap.push(27);
        assert_eq!(heap.len(), 6);
        assert!(*heap.top().unwrap() == 27);
        heap.push(3);
        assert_eq!(heap.len(), 7);
        assert!(*heap.top().unwrap() == 27);
        heap.push(103);
        assert_eq!(heap.len(), 8);
        assert!(*heap.top().unwrap() == 103);
    }

    #[test]
    fn test_push_unique() {
        let mut heap = PriorityQueue::from_vec(vec!(box 2, box 4, box 9));
        assert_eq!(heap.len(), 3);
        assert!(*heap.top().unwrap() == box 9);
        heap.push(box 11);
        assert_eq!(heap.len(), 4);
        assert!(*heap.top().unwrap() == box 11);
        heap.push(box 5);
        assert_eq!(heap.len(), 5);
        assert!(*heap.top().unwrap() == box 11);
        heap.push(box 27);
        assert_eq!(heap.len(), 6);
        assert!(*heap.top().unwrap() == box 27);
        heap.push(box 3);
        assert_eq!(heap.len(), 7);
        assert!(*heap.top().unwrap() == box 27);
        heap.push(box 103);
        assert_eq!(heap.len(), 8);
        assert!(*heap.top().unwrap() == box 103);
    }

    #[test]
    fn test_push_pop() {
        let mut heap = PriorityQueue::from_vec(vec!(5, 5, 2, 1, 3));
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
        let mut heap = PriorityQueue::from_vec(vec!(5, 5, 2, 1, 3));
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(6).unwrap(), 5);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(0).unwrap(), 6);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(4).unwrap(), 5);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.replace(1).unwrap(), 4);
        assert_eq!(heap.len(), 5);
    }

    fn check_to_vec(mut data: Vec<int>) {
        let heap = PriorityQueue::from_vec(data.clone());
        let mut v = heap.clone().into_vec();
        v.sort();
        data.sort();

        assert_eq!(v, data);
        assert_eq!(heap.into_sorted_vec(), data);
    }

    #[test]
    fn test_to_vec() {
        check_to_vec(vec!());
        check_to_vec(vec!(5));
        check_to_vec(vec!(3, 2));
        check_to_vec(vec!(2, 3));
        check_to_vec(vec!(5, 1, 2));
        check_to_vec(vec!(1, 100, 2, 3));
        check_to_vec(vec!(1, 3, 5, 7, 9, 2, 4, 6, 8, 0));
        check_to_vec(vec!(2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1));
        check_to_vec(vec!(9, 11, 9, 9, 9, 9, 11, 2, 3, 4, 11, 9, 0, 0, 0, 0));
        check_to_vec(vec!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        check_to_vec(vec!(10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
        check_to_vec(vec!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2));
        check_to_vec(vec!(5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1));
    }

    #[test]
    fn test_empty_pop() {
        let mut heap: PriorityQueue<int> = PriorityQueue::new();
        assert!(heap.pop().is_none());
    }

    #[test]
    fn test_empty_top() {
        let empty: PriorityQueue<int> = PriorityQueue::new();
        assert!(empty.top().is_none());
    }

    #[test]
    fn test_empty_replace() {
        let mut heap: PriorityQueue<int> = PriorityQueue::new();
        heap.replace(5).is_none();
    }

    #[test]
    fn test_from_iter() {
        let xs = vec!(9u, 8, 7, 6, 5, 4, 3, 2, 1);

        let mut q: PriorityQueue<uint> = xs.as_slice().iter().rev().map(|&x| x).collect();

        for &x in xs.iter() {
            assert_eq!(q.pop().unwrap(), x);
        }
    }
}
