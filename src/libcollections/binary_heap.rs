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
//! Insertion and popping the largest element have `O(log n)` time complexity.
//! Checking the largest element is `O(1)`. Converting a vector to a binary heap
//! can be done in-place, and has `O(n)` complexity. A binary heap can also be
//! converted to a sorted vector in-place, allowing it to be used for an `O(n
//! log n)` in-place heapsort.
//!
//! # Examples
//!
//! This is a larger example that implements [Dijkstra's algorithm][dijkstra]
//! to solve the [shortest path problem][sssp] on a [directed graph][dir_graph].
//! It shows how to use `BinaryHeap` with custom types.
//!
//! [dijkstra]: http://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
//! [sssp]: http://en.wikipedia.org/wiki/Shortest_path_problem
//! [dir_graph]: http://en.wikipedia.org/wiki/Directed_graph
//!
//! ```
//! use std::cmp::Ordering;
//! use std::collections::BinaryHeap;
//! use std::usize;
//!
//! #[derive(Copy, Clone, Eq, PartialEq)]
//! struct State {
//!     cost: usize,
//!     position: usize,
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
//! // Each node is represented as an `usize`, for a shorter implementation.
//! struct Edge {
//!     node: usize,
//!     cost: usize,
//! }
//!
//! // Dijkstra's shortest path algorithm.
//!
//! // Start at `start` and use `dist` to track the current shortest distance
//! // to each node. This implementation isn't memory-efficient as it may leave duplicate
//! // nodes in the queue. It also uses `usize::MAX` as a sentinel value,
//! // for a simpler implementation.
//! fn shortest_path(adj_list: &Vec<Vec<Edge>>, start: usize, goal: usize) -> usize {
//!     // dist[node] = current shortest distance from `start` to `node`
//!     let mut dist: Vec<_> = (0..adj_list.len()).map(|_| usize::MAX).collect();
//!
//!     let mut heap = BinaryHeap::new();
//!
//!     // We're at `start`, with a zero cost
//!     dist[start] = 0;
//!     heap.push(State { cost: 0, position: start });
//!
//!     // Examine the frontier with lower cost nodes first (min-heap)
//!     while let Some(State { cost, position }) = heap.pop() {
//!         // Alternatively we could have continued to find all shortest paths
//!         if position == goal { return cost; }
//!
//!         // Important as we may have already found a better way
//!         if cost > dist[position] { continue; }
//!
//!         // For each node we can reach, see if we can find a way with
//!         // a lower cost going through this node
//!         for edge in &adj_list[position] {
//!             let next = State { cost: cost + edge.cost, position: edge.node };
//!
//!             // If so, add it to the frontier and continue
//!             if next.cost < dist[next.position] {
//!                 heap.push(next);
//!                 // Relaxation, we have now found a better way
//!                 dist[next.position] = next.cost;
//!             }
//!         }
//!     }
//!
//!     // Goal not reachable
//!     usize::MAX
//! }
//!
//! fn main() {
//!     // This is the directed graph we're going to use.
//!     // The node numbers correspond to the different states,
//!     // and the edge weights symbolize the cost of moving
//!     // from one node to another.
//!     // Note that the edges are one-way.
//!     //
//!     //                  7
//!     //          +-----------------+
//!     //          |                 |
//!     //          v   1        2    |  2
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
//!     // Chosen for its efficiency.
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
//!     assert_eq!(shortest_path(&graph, 4, 0), usize::MAX);
//! }
//! ```

#![allow(missing_docs)]
#![stable(feature = "rust1", since = "1.0.0")]

use core::iter::FromIterator;
use core::mem::swap;
use core::ptr;
use core::fmt;

use slice;
use vec::{self, Vec};

/// A priority queue implemented with a binary heap.
///
/// This will be a max-heap.
///
/// It is a logic error for an item to be modified in such a way that the
/// item's ordering relative to any other item, as determined by the `Ord`
/// trait, changes while it is in the heap. This is normally only possible
/// through `Cell`, `RefCell`, global state, I/O, or unsafe code.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct BinaryHeap<T> {
    data: Vec<T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for BinaryHeap<T> {
    fn clone(&self) -> Self {
        BinaryHeap { data: self.data.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Default for BinaryHeap<T> {
    #[inline]
    fn default() -> BinaryHeap<T> {
        BinaryHeap::new()
    }
}

#[stable(feature = "binaryheap_debug", since = "1.4.0")]
impl<T: fmt::Debug + Ord> fmt::Debug for BinaryHeap<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Ord> BinaryHeap<T> {
    /// Creates an empty `BinaryHeap` as a max-heap.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::new();
    /// heap.push(4);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> BinaryHeap<T> {
        BinaryHeap { data: vec![] }
    }

    /// Creates an empty `BinaryHeap` with a specific capacity.
    /// This preallocates enough memory for `capacity` elements,
    /// so that the `BinaryHeap` does not have to be reallocated
    /// until it contains at least that many values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::with_capacity(10);
    /// heap.push(4);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn with_capacity(capacity: usize) -> BinaryHeap<T> {
        BinaryHeap { data: Vec::with_capacity(capacity) }
    }

    /// Creates a `BinaryHeap` from a vector. This is sometimes called
    /// `heapifying` the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(binary_heap_extras)]
    /// # #![allow(deprecated)]
    ///
    /// use std::collections::BinaryHeap;
    /// let heap = BinaryHeap::from_vec(vec![9, 1, 2, 7, 3, 2]);
    /// ```
    #[unstable(feature = "binary_heap_extras",
               reason = "needs to be audited",
               issue = "28147")]
    #[deprecated(since = "1.5.0", reason = "use BinaryHeap::from instead")]
    pub fn from_vec(vec: Vec<T>) -> BinaryHeap<T> {
        BinaryHeap::from(vec)
    }

    /// Returns an iterator visiting all values in the underlying vector, in
    /// arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let heap = BinaryHeap::from(vec![1, 2, 3, 4]);
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order
    /// for x in heap.iter() {
    ///     println!("{}", x);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter { iter: self.data.iter() }
    }

    /// Returns the greatest item in the binary heap, or `None` if it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::new();
    /// assert_eq!(heap.peek(), None);
    ///
    /// heap.push(1);
    /// heap.push(5);
    /// heap.push(2);
    /// assert_eq!(heap.peek(), Some(&5));
    ///
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn peek(&self) -> Option<&T> {
        self.data.get(0)
    }

    /// Returns the number of elements the binary heap can hold without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::with_capacity(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to be inserted in the
    /// given `BinaryHeap`. Does nothing if the capacity is already sufficient.
    ///
    /// Note that the allocator may give the collection more space than it requests. Therefore
    /// capacity can not be relied upon to be precisely minimal. Prefer `reserve` if future
    /// insertions are expected.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::new();
    /// heap.reserve_exact(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.data.reserve_exact(additional);
    }

    /// Reserves capacity for at least `additional` more elements to be inserted in the
    /// `BinaryHeap`. The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::new();
    /// heap.reserve(100);
    /// assert!(heap.capacity() >= 100);
    /// heap.push(4);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Discards as much additional capacity as possible.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Removes the greatest item from the binary heap and returns it, or `None` if it
    /// is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::from(vec![1, 3]);
    ///
    /// assert_eq!(heap.pop(), Some(3));
    /// assert_eq!(heap.pop(), Some(1));
    /// assert_eq!(heap.pop(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop().map(|mut item| {
            if !self.is_empty() {
                swap(&mut item, &mut self.data[0]);
                self.sift_down(0);
            }
            item
        })
    }

    /// Pushes an item onto the binary heap.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::new();
    /// heap.push(3);
    /// heap.push(5);
    /// heap.push(1);
    ///
    /// assert_eq!(heap.len(), 3);
    /// assert_eq!(heap.peek(), Some(&5));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push(&mut self, item: T) {
        let old_len = self.len();
        self.data.push(item);
        self.sift_up(0, old_len);
    }

    /// Pushes an item onto the binary heap, then pops the greatest item off the queue in
    /// an optimized fashion.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(binary_heap_extras)]
    ///
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::new();
    /// heap.push(1);
    /// heap.push(5);
    ///
    /// assert_eq!(heap.push_pop(3), 5);
    /// assert_eq!(heap.push_pop(9), 9);
    /// assert_eq!(heap.len(), 2);
    /// assert_eq!(heap.peek(), Some(&3));
    /// ```
    #[unstable(feature = "binary_heap_extras",
               reason = "needs to be audited",
               issue = "28147")]
    pub fn push_pop(&mut self, mut item: T) -> T {
        match self.data.get_mut(0) {
            None => return item,
            Some(top) => {
                if *top > item {
                    swap(&mut item, top);
                } else {
                    return item;
                }
            }
        }

        self.sift_down(0);
        item
    }

    /// Pops the greatest item off the binary heap, then pushes an item onto the queue in
    /// an optimized fashion. The push is done regardless of whether the binary heap
    /// was empty.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(binary_heap_extras)]
    ///
    /// use std::collections::BinaryHeap;
    /// let mut heap = BinaryHeap::new();
    ///
    /// assert_eq!(heap.replace(1), None);
    /// assert_eq!(heap.replace(3), Some(1));
    /// assert_eq!(heap.len(), 1);
    /// assert_eq!(heap.peek(), Some(&3));
    /// ```
    #[unstable(feature = "binary_heap_extras",
               reason = "needs to be audited",
               issue = "28147")]
    pub fn replace(&mut self, mut item: T) -> Option<T> {
        if !self.is_empty() {
            swap(&mut item, &mut self.data[0]);
            self.sift_down(0);
            Some(item)
        } else {
            self.push(item);
            None
        }
    }

    /// Consumes the `BinaryHeap` and returns the underlying vector
    /// in arbitrary order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let heap = BinaryHeap::from(vec![1, 2, 3, 4, 5, 6, 7]);
    /// let vec = heap.into_vec();
    ///
    /// // Will print in some order
    /// for x in vec {
    ///     println!("{}", x);
    /// }
    /// ```
    #[stable(feature = "binary_heap_extras_15", since = "1.5.0")]
    pub fn into_vec(self) -> Vec<T> {
        self.into()
    }

    /// Consumes the `BinaryHeap` and returns a vector in sorted
    /// (ascending) order.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    ///
    /// let mut heap = BinaryHeap::from(vec![1, 2, 4, 5, 7]);
    /// heap.push(6);
    /// heap.push(3);
    ///
    /// let vec = heap.into_sorted_vec();
    /// assert_eq!(vec, [1, 2, 3, 4, 5, 6, 7]);
    /// ```
    #[stable(feature = "binary_heap_extras_15", since = "1.5.0")]
    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut end = self.len();
        while end > 1 {
            end -= 1;
            self.data.swap(0, end);
            self.sift_down_range(0, end);
        }
        self.into_vec()
    }

    // The implementations of sift_up and sift_down use unsafe blocks in
    // order to move an element out of the vector (leaving behind a
    // hole), shift along the others and move the removed element back into the
    // vector at the final location of the hole.
    // The `Hole` type is used to represent this, and make sure
    // the hole is filled back at the end of its scope, even on panic.
    // Using a hole reduces the constant factor compared to using swaps,
    // which involves twice as many moves.
    fn sift_up(&mut self, start: usize, pos: usize) {
        unsafe {
            // Take out the value at `pos` and create a hole.
            let mut hole = Hole::new(&mut self.data, pos);

            while hole.pos() > start {
                let parent = (hole.pos() - 1) / 2;
                if hole.element() <= hole.get(parent) {
                    break;
                }
                hole.move_to(parent);
            }
        }
    }

    /// Take an element at `pos` and move it down the heap,
    /// while its children are larger.
    fn sift_down_range(&mut self, pos: usize, end: usize) {
        unsafe {
            let mut hole = Hole::new(&mut self.data, pos);
            let mut child = 2 * pos + 1;
            while child < end {
                let right = child + 1;
                // compare with the greater of the two children
                if right < end && !(hole.get(child) > hole.get(right)) {
                    child = right;
                }
                // if we are already in order, stop.
                if hole.element() >= hole.get(child) {
                    break;
                }
                hole.move_to(child);
                child = 2 * hole.pos() + 1;
            }
        }
    }

    fn sift_down(&mut self, pos: usize) {
        let len = self.len();
        self.sift_down_range(pos, len);
    }

    /// Returns the length of the binary heap.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks if the binary heap is empty.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the binary heap, returning an iterator over the removed elements.
    ///
    /// The elements are removed in arbitrary order.
    #[inline]
    #[unstable(feature = "drain",
               reason = "matches collection reform specification, \
                         waiting for dust to settle",
               issue = "27711")]
    pub fn drain(&mut self) -> Drain<T> {
        Drain { iter: self.data.drain(..) }
    }

    /// Drops all items from the binary heap.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        self.drain();
    }
}

/// Hole represents a hole in a slice i.e. an index without valid value
/// (because it was moved from or duplicated).
/// In drop, `Hole` will restore the slice by filling the hole
/// position with the value that was originally removed.
struct Hole<'a, T: 'a> {
    data: &'a mut [T],
    /// `elt` is always `Some` from new until drop.
    elt: Option<T>,
    pos: usize,
}

impl<'a, T> Hole<'a, T> {
    /// Create a new Hole at index `pos`.
    fn new(data: &'a mut [T], pos: usize) -> Self {
        unsafe {
            let elt = ptr::read(&data[pos]);
            Hole {
                data: data,
                elt: Some(elt),
                pos: pos,
            }
        }
    }

    #[inline(always)]
    fn pos(&self) -> usize {
        self.pos
    }

    /// Return a reference to the element removed
    #[inline(always)]
    fn element(&self) -> &T {
        self.elt.as_ref().unwrap()
    }

    /// Return a reference to the element at `index`.
    ///
    /// Panics if the index is out of bounds.
    ///
    /// Unsafe because index must not equal pos.
    #[inline(always)]
    unsafe fn get(&self, index: usize) -> &T {
        debug_assert!(index != self.pos);
        &self.data[index]
    }

    /// Move hole to new location
    ///
    /// Unsafe because index must not equal pos.
    #[inline(always)]
    unsafe fn move_to(&mut self, index: usize) {
        debug_assert!(index != self.pos);
        let index_ptr: *const _ = &self.data[index];
        let hole_ptr = &mut self.data[self.pos];
        ptr::copy_nonoverlapping(index_ptr, hole_ptr, 1);
        self.pos = index;
    }
}

impl<'a, T> Drop for Hole<'a, T> {
    fn drop(&mut self) {
        // fill the hole again
        unsafe {
            let pos = self.pos;
            ptr::write(&mut self.data[pos], self.elt.take().unwrap());
        }
    }
}

/// `BinaryHeap` iterator.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    iter: slice::Iter<'a, T>,
}

// FIXME(#19839) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> {
        Iter { iter: self.iter.clone() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        self.iter.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

/// An iterator that moves out of a `BinaryHeap`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    iter: vec::IntoIter<T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {}

/// An iterator that drains a `BinaryHeap`.
#[unstable(feature = "drain", reason = "recent addition", issue = "27711")]
pub struct Drain<'a, T: 'a> {
    iter: vec::Drain<'a, T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: 'a> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: 'a> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: 'a> ExactSizeIterator for Drain<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> From<Vec<T>> for BinaryHeap<T> {
    fn from(vec: Vec<T>) -> BinaryHeap<T> {
        let mut heap = BinaryHeap { data: vec };
        let mut n = heap.len() / 2;
        while n > 0 {
            n -= 1;
            heap.sift_down(n);
        }
        heap
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> From<BinaryHeap<T>> for Vec<T> {
    fn from(heap: BinaryHeap<T>) -> Vec<T> {
        heap.data
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> FromIterator<T> for BinaryHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> BinaryHeap<T> {
        BinaryHeap::from(iter.into_iter().collect::<Vec<_>>())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> IntoIterator for BinaryHeap<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Creates a consuming iterator, that is, one that moves each value out of
    /// the binary heap in arbitrary order. The binary heap cannot be used
    /// after calling this.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::BinaryHeap;
    /// let heap = BinaryHeap::from(vec![1, 2, 3, 4]);
    ///
    /// // Print 1, 2, 3, 4 in arbitrary order
    /// for x in heap.into_iter() {
    ///     // x has type i32, not &i32
    ///     println!("{}", x);
    /// }
    /// ```
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { iter: self.data.into_iter() }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a BinaryHeap<T> where T: Ord {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Extend<T> for BinaryHeap<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iterable: I) {
        let iter = iterable.into_iter();
        let (lower, _) = iter.size_hint();

        self.reserve(lower);

        for elem in iter {
            self.push(elem);
        }
    }
}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Ord + Copy> Extend<&'a T> for BinaryHeap<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}
