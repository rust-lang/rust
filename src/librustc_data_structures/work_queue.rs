use crate::bit_set::BitSet;
use crate::indexed_vec::Idx;
use std::collections::VecDeque;

/// A work queue is a handy data structure for tracking work left to
/// do. (For example, basic blocks left to process.) It is basically a
/// de-duplicating queue; so attempting to insert X if X is already
/// enqueued has no effect. This implementation assumes that the
/// elements are dense indices, so it can allocate the queue to size
/// and also use a bit set to track occupancy.
pub struct WorkQueue<T: Idx> {
    deque: VecDeque<T>,
    set: BitSet<T>,
}

impl<T: Idx> WorkQueue<T> {
    /// Creates a new work queue with all the elements from (0..len).
    #[inline]
    pub fn with_all(len: usize) -> Self {
        WorkQueue {
            deque: (0..len).map(T::new).collect(),
            set: BitSet::new_filled(len),
        }
    }

    /// Creates a new work queue that starts empty, where elements range from (0..len).
    #[inline]
    pub fn with_none(len: usize) -> Self {
        WorkQueue {
            deque: VecDeque::with_capacity(len),
            set: BitSet::new_empty(len),
        }
    }

    /// Attempt to enqueue `element` in the work queue. Returns false if it was already present.
    #[inline]
    pub fn insert(&mut self, element: T) -> bool {
        if self.set.insert(element) {
            self.deque.push_back(element);
            true
        } else {
            false
        }
    }

    /// Attempt to pop an element from the work queue.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if let Some(element) = self.deque.pop_front() {
            self.set.remove(element);
            Some(element)
        } else {
            None
        }
    }

    /// Returns `true` if nothing is enqueued.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }
}
