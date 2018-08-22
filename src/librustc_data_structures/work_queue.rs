// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use indexed_set::IdxSet;
use indexed_vec::Idx;
use std::collections::VecDeque;

/// A work queue is a handy data structure for tracking work left to
/// do. (For example, basic blocks left to process.) It is basically a
/// de-duplicating queue; so attempting to insert X if X is already
/// enqueued has no effect. This implementation assumes that the
/// elements are dense indices, so it can allocate the queue to size
/// and also use a bit set to track occupancy.
pub struct WorkQueue<T: Idx> {
    deque: VecDeque<T>,
    set: IdxSet<T>,
}

impl<T: Idx> WorkQueue<T> {
    /// Create a new work queue with all the elements from (0..len).
    #[inline]
    pub fn with_all(len: usize) -> Self {
        WorkQueue {
            deque: (0..len).map(T::new).collect(),
            set: IdxSet::new_filled(len),
        }
    }

    /// Create a new work queue that starts empty, where elements range from (0..len).
    #[inline]
    pub fn with_none(len: usize) -> Self {
        WorkQueue {
            deque: VecDeque::with_capacity(len),
            set: IdxSet::new_empty(len),
        }
    }

    /// Attempt to enqueue `element` in the work queue. Returns false if it was already present.
    #[inline]
    pub fn insert(&mut self, element: T) -> bool {
        if self.set.add(&element) {
            self.deque.push_back(element);
            true
        } else {
            false
        }
    }

    /// Attempt to enqueue `element` in the work queue. Returns false if it was already present.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if let Some(element) = self.deque.pop_front() {
            self.set.remove(&element);
            Some(element)
        } else {
            None
        }
    }

    /// True if nothing is enqueued.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }
}
