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

use core::container::{Container, Mutable};
use core::cmp::Ord;
use core::iter::BaseIter;
use core::prelude::*;
use core::ptr::addr_of;
use core::vec;

#[abi = "rust-intrinsic"]
extern "rust-intrinsic" mod rusti {
    fn move_val_init<T>(dst: &mut T, +src: T);
    fn init<T>() -> T;
}

pub struct PriorityQueue<T> {
    priv data: ~[T],
}

impl<T:Ord> BaseIter<T> for PriorityQueue<T> {
    /// Visit all values in the underlying vector.
    ///
    /// The values are **not** visited in order.
    fn each(&self, f: &fn(&T) -> bool) { self.data.each(f) }
    fn size_hint(&self) -> Option<uint> { self.data.size_hint() }
}

impl<T:Ord> Container for PriorityQueue<T> {
    /// Returns the length of the queue
    fn len(&const self) -> uint { vec::uniq_len(&const self.data) }

    /// Returns true if a queue contains no elements
    fn is_empty(&const self) -> bool { self.len() == 0 }
}

impl<T:Ord> Mutable for PriorityQueue<T> {
    /// Drop all items from the queue
    fn clear(&mut self) { self.data.truncate(0) }
}

pub impl <T:Ord> PriorityQueue<T> {
    /// Returns the greatest item in the queue - fails if empty
    #[cfg(stage0)]
    fn top(&self) -> &'self T { &self.data[0] }

    /// Returns the greatest item in the queue - fails if empty
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn top<'a>(&'a self) -> &'a T { &self.data[0] }

    /// Returns the greatest item in the queue - None if empty
    #[cfg(stage0)]
    fn maybe_top(&self) -> Option<&'self T> {
        if self.is_empty() { None } else { Some(self.top()) }
    }

    /// Returns the greatest item in the queue - None if empty
    #[cfg(stage1)]
    #[cfg(stage2)]
    #[cfg(stage3)]
    fn maybe_top<'a>(&'a self) -> Option<&'a T> {
        if self.is_empty() { None } else { Some(self.top()) }
    }

    /// Returns the number of elements the queue can hold without reallocating
    fn capacity(&self) -> uint { vec::capacity(&self.data) }

    fn reserve(&mut self, n: uint) { vec::reserve(&mut self.data, n) }

    fn reserve_at_least(&mut self, n: uint) {
        vec::reserve_at_least(&mut self.data, n)
    }

    /// Pop the greatest item from the queue - fails if empty
    fn pop(&mut self) -> T {
        let mut item = self.data.pop();
        if !self.is_empty() { item <-> self.data[0]; self.siftdown(0); }
        item
    }

    /// Pop the greatest item from the queue - None if empty
    fn maybe_pop(&mut self) -> Option<T> {
        if self.is_empty() { None } else { Some(self.pop()) }
    }

    /// Push an item onto the queue
    fn push(&mut self, item: T) {
        self.data.push(item);
        let new_len = self.len() - 1;
        self.siftup(0, new_len);
    }

    /// Optimized version of a push followed by a pop
    fn push_pop(&mut self, mut item: T) -> T {
        if !self.is_empty() && self.data[0] > item {
            item <-> self.data[0];
            self.siftdown(0);
        }
        item
    }

    /// Optimized version of a pop followed by a push - fails if empty
    fn replace(&mut self, mut item: T) -> T {
        item <-> self.data[0];
        self.siftdown(0);
        item
    }

    /// Consume the PriorityQueue and return the underlying vector
    fn to_vec(self) -> ~[T] { let PriorityQueue{data: v} = self; v }

    /// Consume the PriorityQueue and return a vector in sorted
    /// (ascending) order
    fn to_sorted_vec(self) -> ~[T] {
        let mut q = self;
        let mut end = q.len();
        while end > 1 {
            end -= 1;
            q.data[end] <-> q.data[0];
            q.siftdown_range(0, end)
        }
        q.to_vec()
    }

    /// Create an empty PriorityQueue
    fn new() -> PriorityQueue<T> { PriorityQueue{data: ~[],} }

    /// Create a PriorityQueue from a vector (heapify)
    fn from_vec(xs: ~[T]) -> PriorityQueue<T> {
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
    // junk element), shift along the others and move it back into the
    // vector over the junk element.  This reduces the constant factor
    // compared to using swaps, which involves twice as many moves.

    priv fn siftup(&mut self, start: uint, mut pos: uint) {
        unsafe {
            let new = *addr_of(&self.data[pos]);

            while pos > start {
                let parent = (pos - 1) >> 1;
                if new > self.data[parent] {
                    let mut x = rusti::init();
                    x <-> self.data[parent];
                    rusti::move_val_init(&mut self.data[pos], x);
                    pos = parent;
                    loop
                }
                break
            }
            rusti::move_val_init(&mut self.data[pos], new);
        }
    }

    priv fn siftdown_range(&mut self, mut pos: uint, end: uint) {
        unsafe {
            let start = pos;
            let new = *addr_of(&self.data[pos]);

            let mut child = 2 * pos + 1;
            while child < end {
                let right = child + 1;
                if right < end && !(self.data[child] > self.data[right]) {
                    child = right;
                }
                let mut x = rusti::init();
                x <-> self.data[child];
                rusti::move_val_init(&mut self.data[pos], x);
                pos = child;
                child = 2 * pos + 1;
            }

            rusti::move_val_init(&mut self.data[pos], new);
            self.siftup(start, pos);
        }
    }

    priv fn siftdown(&mut self, pos: uint) {
        let len = self.len();
        self.siftdown_range(pos, len);
    }
}

#[cfg(test)]
mod tests {
    use sort::merge_sort;
    use core::cmp::le;
    use priority_queue::PriorityQueue::{from_vec, new};

    #[test]
    fn test_top_and_pop() {
        let data = ~[2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
        let mut sorted = merge_sort(data, le);
        let mut heap = from_vec(data);
        while !heap.is_empty() {
            assert!(heap.top() == sorted.last());
            assert!(heap.pop() == sorted.pop());
        }
    }

    #[test]
    fn test_push() {
        let mut heap = from_vec(~[2, 4, 9]);
        assert!(heap.len() == 3);
        assert!(*heap.top() == 9);
        heap.push(11);
        assert!(heap.len() == 4);
        assert!(*heap.top() == 11);
        heap.push(5);
        assert!(heap.len() == 5);
        assert!(*heap.top() == 11);
        heap.push(27);
        assert!(heap.len() == 6);
        assert!(*heap.top() == 27);
        heap.push(3);
        assert!(heap.len() == 7);
        assert!(*heap.top() == 27);
        heap.push(103);
        assert!(heap.len() == 8);
        assert!(*heap.top() == 103);
    }

    #[test]
    fn test_push_unique() {
        let mut heap = from_vec(~[~2, ~4, ~9]);
        assert!(heap.len() == 3);
        assert!(*heap.top() == ~9);
        heap.push(~11);
        assert!(heap.len() == 4);
        assert!(*heap.top() == ~11);
        heap.push(~5);
        assert!(heap.len() == 5);
        assert!(*heap.top() == ~11);
        heap.push(~27);
        assert!(heap.len() == 6);
        assert!(*heap.top() == ~27);
        heap.push(~3);
        assert!(heap.len() == 7);
        assert!(*heap.top() == ~27);
        heap.push(~103);
        assert!(heap.len() == 8);
        assert!(*heap.top() == ~103);
    }

    #[test]
    fn test_push_pop() {
        let mut heap = from_vec(~[5, 5, 2, 1, 3]);
        assert!(heap.len() == 5);
        assert!(heap.push_pop(6) == 6);
        assert!(heap.len() == 5);
        assert!(heap.push_pop(0) == 5);
        assert!(heap.len() == 5);
        assert!(heap.push_pop(4) == 5);
        assert!(heap.len() == 5);
        assert!(heap.push_pop(1) == 4);
        assert!(heap.len() == 5);
    }

    #[test]
    fn test_replace() {
        let mut heap = from_vec(~[5, 5, 2, 1, 3]);
        assert!(heap.len() == 5);
        assert!(heap.replace(6) == 5);
        assert!(heap.len() == 5);
        assert!(heap.replace(0) == 6);
        assert!(heap.len() == 5);
        assert!(heap.replace(4) == 5);
        assert!(heap.len() == 5);
        assert!(heap.replace(1) == 4);
        assert!(heap.len() == 5);
    }

    fn check_to_vec(data: ~[int]) {
        let heap = from_vec(data);
        assert!(merge_sort(heap.to_vec(), le) == merge_sort(data, le));
        assert!(heap.to_sorted_vec() == merge_sort(data, le));
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
    fn test_empty_pop() { let mut heap = new::<int>(); heap.pop(); }

    #[test]
    fn test_empty_maybe_pop() {
        let mut heap = new::<int>();
        assert!(heap.maybe_pop().is_none());
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_empty_top() { let empty = new::<int>(); empty.top(); }

    #[test]
    fn test_empty_maybe_top() {
        let empty = new::<int>();
        assert!(empty.maybe_top().is_none());
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(windows))]
    fn test_empty_replace() { let mut heap = new(); heap.replace(5); }
}
