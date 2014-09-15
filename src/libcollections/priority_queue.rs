// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A priority queue implemented with a binary heap.
//!
//! Insertions have `O(log n)` time complexity and checking or popping the largest element is
//! `O(1)`. Converting a vector to a priority queue can be done in-place, and has `O(n)`
//! complexity. A priority queue can also be converted to a sorted vector in-place, allowing it to
//! be used for an `O(n log n)` in-place heapsort.
//!
//! # Example
//!
//! This is a larger example which implements [Dijkstra's algorithm][dijkstra]
//! to solve the [shortest path problem][sssp] on a [directed graph][dir_graph].
//! It showcases how to use the `PriorityQueue` with custom types.
//!
//! [dijkstra]: http://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
//! [sssp]: http://en.wikipedia.org/wiki/Shortest_path_problem
//! [dir_graph]: http://en.wikipedia.org/wiki/Directed_graph
//!
//! ```
//! use std::collections::PriorityQueue;
//! use std::uint;
//!
//! #[deriving(Eq, PartialEq)]
//! struct State {
//!     cost: uint,
//!     position: uint
//! }
//!
//! // The priority queue depends on `Ord`.
//! // Explicitly implement the trait so the queue becomes a min-heap
//! // instead of a max-heap.
//! impl Ord for State {
//!     fn cmp(&self, other: &State) -> Ordering {
//!         // Notice that the we flip the ordering here
//!         other.cost.cmp(&self.cost)
//!     }
//! }
//!
//! // `PartialOrd` needs to be implemented as well.
//! impl PartialOrd for State {
//!     fn partial_cmp(&self, other: &State) -> Option<Ordering> {
//!         Some(self.cmp(other))
//!     }
//! }
//!
//! // Each node is represented as an `uint`, for a shorter implementation.
//! struct Edge {
//!     node: uint,
//!     cost: uint
//! }
//!
//! // Dijkstra's shortest path algorithm.
//!
//! // Start at `start` and use `dist` to track the current shortest distance
//! // to each node. This implementation isn't memory efficient as it may leave duplicate
//! // nodes in the queue. It also uses `uint::MAX` as a sentinel value,
//! // for a simpler implementation.
//! fn shortest_path(adj_list: &Vec<Vec<Edge>>, start: uint, goal: uint) -> uint {
//!     // dist[node] = current shortest distance from `start` to `node`
//!     let mut dist = Vec::from_elem(adj_list.len(), uint::MAX);
//!
//!     let mut pq = PriorityQueue::new();
//!
//!     // We're at `start`, with a zero cost
//!     *dist.get_mut(start) = 0u;
//!     pq.push(State { cost: 0u, position: start });
//!
//!     // Examine the frontier with lower cost nodes first (min-heap)
//!     loop {
//!         let State { cost, position } = match pq.pop() {
//!             None => break, // empty
//!             Some(s) => s
//!         };
//!
//!         // Alternatively we could have continued to find all shortest paths
//!         if position == goal { return cost }
//!
//!         // Important as we may have already found a better way
//!         if cost > dist[position] { continue }
//!
//!         // For each node we can reach, see if we can find a way with
//!         // a lower cost going through this node
//!         for edge in adj_list[position].iter() {
//!             let next = State { cost: cost + edge.cost, position: edge.node };
//!
//!             // If so, add it to the frontier and continue
//!             if next.cost < dist[next.position] {
//!                 pq.push(next);
//!                 // Relaxation, we have now found a better way
//!                 *dist.get_mut(next.position) = next.cost;
//!             }
//!         }
//!     }
//!
//!     // Goal not reachable
//!     uint::MAX
//! }
//!
//! fn main() {
//!     // This is the directed graph we're going to use.
//!     // The node numbers correspond to the different states,
//!     // and the edge weights symbolises the cost of moving
//!     // from one node to another.
//!     // Note that the edges are one-way.
//!     //
//!     //                  7
//!     //          +-----------------+
//!     //          |                 |
//!     //          v   1        2    |
//!     //          0 -----> 1 -----> 3 ---> 4
//!     //          |        ^        ^      ^
//!     //          |        | 1      |      |
//!     //          |        |        | 3    | 1
//!     //          +------> 2 -------+      |
//!     //           10      |               |
//!     //                   +---------------+
//!     //
//!     // The graph is represented as an adjacency list where each index,
//!     // corresponding to a node value, has a list of outgoing edges.
//!     // Chosen for it's efficiency.
//!     let graph = vec![
//!         // Node 0
//!         vec![Edge { node: 2, cost: 10 },
//!              Edge { node: 1, cost: 1 }],
//!         // Node 1
//!         vec![Edge { node: 3, cost: 2 }],
//!         // Node 2
//!         vec![Edge { node: 1, cost: 1 },
//!              Edge { node: 3, cost: 3 },
//!              Edge { node: 4, cost: 1 }],
//!         // Node 3
//!         vec![Edge { node: 0, cost: 7 },
//!              Edge { node: 4, cost: 2 }],
//!         // Node 4
//!         vec![]];
//!
//!     assert_eq!(shortest_path(&graph, 0, 1), 1);
//!     assert_eq!(shortest_path(&graph, 0, 3), 3);
//!     assert_eq!(shortest_path(&graph, 3, 0), 7);
//!     assert_eq!(shortest_path(&graph, 0, 4), 5);
//!     assert_eq!(shortest_path(&graph, 4, 0), uint::MAX);
//! }
//! ```

#![allow(missing_doc)]

use core::prelude::*;

use core::default::Default;
use core::mem::{zeroed, replace, swap};
use core::ptr;

use {Mutable, MutableSeq};
use slice;
use vec::Vec;

/// A priority queue implemented with a binary heap.
///
/// This will be a max-heap.
#[deriving(Clone)]
pub struct PriorityQueue<T> {
    data: Vec<T>,
}

impl<T: Ord> Collection for PriorityQueue<T> {
    /// Returns the length of the queue.
    fn len(&self) -> uint { self.data.len() }
}

impl<T: Ord> Mutable for PriorityQueue<T> {
    /// Drops all items from the queue.
    fn clear(&mut self) { self.data.truncate(0) }
}

impl<T: Ord> Default for PriorityQueue<T> {
    #[inline]
    fn default() -> PriorityQueue<T> { PriorityQueue::new() }
}

impl<T: Ord> PriorityQueue<T> {
    /// Creates an empty `PriorityQueue` as a max-heap.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    /// let pq: PriorityQueue<uint> = PriorityQueue::new();
    /// ```
    pub fn new() -> PriorityQueue<T> { PriorityQueue{data: vec!(),} }

    /// Creates an empty `PriorityQueue` with a specific capacity.
    /// This preallocates enough memory for `capacity` elements,
    /// so that the `PriorityQueue` does not have to be reallocated
    /// until it contains at least that many values.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    /// let pq: PriorityQueue<uint> = PriorityQueue::with_capacity(10u);
    /// ```
    pub fn with_capacity(capacity: uint) -> PriorityQueue<T> {
        PriorityQueue { data: Vec::with_capacity(capacity) }
    }

    /// Creates a `PriorityQueue` from a vector. This is sometimes called
    /// `heapifying` the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    /// let pq = PriorityQueue::from_vec(vec![9i, 1, 2, 7, 3, 2]);
    /// ```
    pub fn from_vec(xs: Vec<T>) -> PriorityQueue<T> {
        let mut q = PriorityQueue{data: xs,};
        let mut n = q.len() / 2;
        while n > 0 {
            n -= 1;
            q.siftdown(n)
        }
        q
    }

    /// An iterator visiting all values in underlying vector, in
    /// arbitrary order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    /// let pq = PriorityQueue::from_vec(vec![1i, 2, 3, 4]);
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order
    /// for x in pq.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        Items { iter: self.data.iter() }
    }

    /// Returns the greatest item in a queue, or `None` if it is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq = PriorityQueue::new();
    /// assert_eq!(pq.top(), None);
    ///
    /// pq.push(1i);
    /// pq.push(5i);
    /// pq.push(2i);
    /// assert_eq!(pq.top(), Some(&5i));
    ///
    /// ```
    pub fn top<'a>(&'a self) -> Option<&'a T> {
        if self.is_empty() { None } else { Some(&self.data[0]) }
    }

    #[deprecated="renamed to `top`"]
    pub fn maybe_top<'a>(&'a self) -> Option<&'a T> { self.top() }

    /// Returns the number of elements the queue can hold without reallocating.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let pq: PriorityQueue<uint> = PriorityQueue::with_capacity(100u);
    /// assert!(pq.capacity() >= 100u);
    /// ```
    pub fn capacity(&self) -> uint { self.data.capacity() }

    /// Reserves capacity for exactly `n` elements in the `PriorityQueue`.
    /// Do nothing if the capacity is already sufficient.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq: PriorityQueue<uint> = PriorityQueue::new();
    /// pq.reserve_exact(100u);
    /// assert!(pq.capacity() == 100u);
    /// ```
    pub fn reserve_exact(&mut self, n: uint) { self.data.reserve_exact(n) }

    /// Reserves capacity for at least `n` elements in the `PriorityQueue`.
    /// Do nothing if the capacity is already sufficient.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq: PriorityQueue<uint> = PriorityQueue::new();
    /// pq.reserve(100u);
    /// assert!(pq.capacity() >= 100u);
    /// ```
    pub fn reserve(&mut self, n: uint) {
        self.data.reserve(n)
    }

    /// Removes the greatest item from a queue and returns it, or `None` if it
    /// is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq = PriorityQueue::from_vec(vec![1i, 3]);
    ///
    /// assert_eq!(pq.pop(), Some(3i));
    /// assert_eq!(pq.pop(), Some(1i));
    /// assert_eq!(pq.pop(), None);
    /// ```
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

    /// Pushes an item onto the queue.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq = PriorityQueue::new();
    /// pq.push(3i);
    /// pq.push(5i);
    /// pq.push(1i);
    ///
    /// assert_eq!(pq.len(), 3);
    /// assert_eq!(pq.top(), Some(&5i));
    /// ```
    pub fn push(&mut self, item: T) {
        self.data.push(item);
        let new_len = self.len() - 1;
        self.siftup(0, new_len);
    }

    /// Pushes an item onto a queue then pops the greatest item off the queue in
    /// an optimized fashion.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq = PriorityQueue::new();
    /// pq.push(1i);
    /// pq.push(5i);
    ///
    /// assert_eq!(pq.push_pop(3i), 5);
    /// assert_eq!(pq.push_pop(9i), 9);
    /// assert_eq!(pq.len(), 2);
    /// assert_eq!(pq.top(), Some(&3i));
    /// ```
    pub fn push_pop(&mut self, mut item: T) -> T {
        if !self.is_empty() && *self.top().unwrap() > item {
            swap(&mut item, self.data.get_mut(0));
            self.siftdown(0);
        }
        item
    }

    /// Pops the greatest item off a queue then pushes an item onto the queue in
    /// an optimized fashion. The push is done regardless of whether the queue
    /// was empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq = PriorityQueue::new();
    ///
    /// assert_eq!(pq.replace(1i), None);
    /// assert_eq!(pq.replace(3i), Some(1i));
    /// assert_eq!(pq.len(), 1);
    /// assert_eq!(pq.top(), Some(&3i));
    /// ```
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

    /// Consumes the `PriorityQueue` and returns the underlying vector
    /// in arbitrary order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let pq = PriorityQueue::from_vec(vec![1i, 2, 3, 4, 5, 6, 7]);
    /// let vec = pq.into_vec();
    ///
    /// // Will print in some order
    /// for x in vec.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    pub fn into_vec(self) -> Vec<T> { let PriorityQueue{data: v} = self; v }

    /// Consumes the `PriorityQueue` and returns a vector in sorted
    /// (ascending) order.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::PriorityQueue;
    ///
    /// let mut pq = PriorityQueue::from_vec(vec![1i, 2, 4, 5, 7]);
    /// pq.push(6);
    /// pq.push(3);
    ///
    /// let vec = pq.into_sorted_vec();
    /// assert_eq!(vec, vec![1i, 2, 3, 4, 5, 6, 7]);
    /// ```
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

    // The implementations of siftup and siftdown use unsafe blocks in
    // order to move an element out of the vector (leaving behind a
    // zeroed element), shift along the others and move it back into the
    // vector over the junk element.  This reduces the constant factor
    // compared to using swaps, which involves twice as many moves.
    fn siftup(&mut self, start: uint, mut pos: uint) {
        unsafe {
            let new = replace(self.data.get_mut(pos), zeroed());

            while pos > start {
                let parent = (pos - 1) >> 1;
                if new > self.data[parent] {
                    let x = replace(self.data.get_mut(parent), zeroed());
                    ptr::write(self.data.get_mut(pos), x);
                    pos = parent;
                    continue
                }
                break
            }
            ptr::write(self.data.get_mut(pos), new);
        }
    }

    fn siftdown_range(&mut self, mut pos: uint, end: uint) {
        unsafe {
            let start = pos;
            let new = replace(self.data.get_mut(pos), zeroed());

            let mut child = 2 * pos + 1;
            while child < end {
                let right = child + 1;
                if right < end && !(self.data[child] > self.data[right]) {
                    child = right;
                }
                let x = replace(self.data.get_mut(child), zeroed());
                ptr::write(self.data.get_mut(pos), x);
                pos = child;
                child = 2 * pos + 1;
            }

            ptr::write(self.data.get_mut(pos), new);
            self.siftup(start, pos);
        }
    }

    fn siftdown(&mut self, pos: uint) {
        let len = self.len();
        self.siftdown_range(pos, len);
    }
}

/// `PriorityQueue` iterator.
pub struct Items <'a, T:'a> {
    iter: slice::Items<'a, T>,
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    #[inline]
    fn next(&mut self) -> Option<(&'a T)> { self.iter.next() }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) { self.iter.size_hint() }
}

impl<T: Ord> FromIterator<T> for PriorityQueue<T> {
    fn from_iter<Iter: Iterator<T>>(mut iter: Iter) -> PriorityQueue<T> {
        let vec: Vec<T> = iter.collect();
        PriorityQueue::from_vec(vec)
    }
}

impl<T: Ord> Extendable<T> for PriorityQueue<T> {
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
    use std::prelude::*;

    use priority_queue::PriorityQueue;
    use vec::Vec;
    use MutableSeq;

    #[test]
    fn test_iterator() {
        let data = vec!(5i, 9, 3);
        let iterout = [9i, 5, 3];
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
        let mut heap = PriorityQueue::from_vec(vec!(2i, 4, 9));
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
        let mut heap = PriorityQueue::from_vec(vec!(box 2i, box 4, box 9));
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
        let mut heap = PriorityQueue::from_vec(vec!(5i, 5, 2, 1, 3));
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
        let mut heap = PriorityQueue::from_vec(vec!(5i, 5, 2, 1, 3));
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

        assert_eq!(v.as_slice(), data.as_slice());
        assert_eq!(heap.into_sorted_vec().as_slice(), data.as_slice());
    }

    #[test]
    fn test_to_vec() {
        check_to_vec(vec!());
        check_to_vec(vec!(5i));
        check_to_vec(vec!(3i, 2));
        check_to_vec(vec!(2i, 3));
        check_to_vec(vec!(5i, 1, 2));
        check_to_vec(vec!(1i, 100, 2, 3));
        check_to_vec(vec!(1i, 3, 5, 7, 9, 2, 4, 6, 8, 0));
        check_to_vec(vec!(2i, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1));
        check_to_vec(vec!(9i, 11, 9, 9, 9, 9, 11, 2, 3, 4, 11, 9, 0, 0, 0, 0));
        check_to_vec(vec!(0i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        check_to_vec(vec!(10i, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
        check_to_vec(vec!(0i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2));
        check_to_vec(vec!(5i, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1));
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
