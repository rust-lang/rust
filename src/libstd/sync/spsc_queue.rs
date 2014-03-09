/* Copyright (c) 2010-2011 Dmitry Vyukov. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *    1. Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY DMITRY VYUKOV "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL DMITRY VYUKOV OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of Dmitry Vyukov.
 */

// http://www.1024cores.net/home/lock-free-algorithms/queues/unbounded-spsc-queue

//! A single-producer single-consumer concurrent queue
//!
//! This module contains the implementation of an SPSC queue which can be used
//! concurrently between two tasks. This data structure is safe to use and
//! enforces the semantics that there is one pusher and one popper.

use cast;
use kinds::Send;
use ops::Drop;
use option::{Some, None, Option};
use ptr::RawPtr;
use sync::atomics::{AtomicPtr, Relaxed, AtomicUint, Acquire, Release};

// Node within the linked list queue of messages to send
struct Node<T> {
    // FIXME: this could be an uninitialized T if we're careful enough, and
    //      that would reduce memory usage (and be a bit faster).
    //      is it worth it?
    value: Option<T>,           // nullable for re-use of nodes
    next: AtomicPtr<Node<T>>,   // next node in the queue
}

/// The single-producer single-consumer queue. This structure is not cloneable,
/// but it can be safely shared in an UnsafeArc if it is guaranteed that there
/// is only one popper and one pusher touching the queue at any one point in
/// time.
pub struct Queue<T> {
    // consumer fields
    priv tail: *mut Node<T>, // where to pop from
    priv tail_prev: AtomicPtr<Node<T>>, // where to pop from

    // producer fields
    priv head: *mut Node<T>,      // where to push to
    priv first: *mut Node<T>,     // where to get new nodes from
    priv tail_copy: *mut Node<T>, // between first/tail

    // Cache maintenance fields. Additions and subtractions are stored
    // separately in order to allow them to use nonatomic addition/subtraction.
    priv cache_bound: uint,
    priv cache_additions: AtomicUint,
    priv cache_subtractions: AtomicUint,
}

impl<T: Send> Node<T> {
    fn new() -> *mut Node<T> {
        unsafe {
            cast::transmute(~Node {
                value: None,
                next: AtomicPtr::new(0 as *mut Node<T>),
            })
        }
    }
}

impl<T: Send> Queue<T> {
    /// Creates a new queue. The producer returned is connected to the consumer
    /// to push all data to the consumer.
    ///
    /// # Arguments
    ///
    ///   * `bound` - This queue implementation is implemented with a linked
    ///               list, and this means that a push is always a malloc. In
    ///               order to amortize this cost, an internal cache of nodes is
    ///               maintained to prevent a malloc from always being
    ///               necessary. This bound is the limit on the size of the
    ///               cache (if desired). If the value is 0, then the cache has
    ///               no bound. Otherwise, the cache will never grow larger than
    ///               `bound` (although the queue itself could be much larger.
    pub fn new(bound: uint) -> Queue<T> {
        let n1 = Node::new();
        let n2 = Node::new();
        unsafe { (*n1).next.store(n2, Relaxed) }
        Queue {
            tail: n2,
            tail_prev: AtomicPtr::new(n1),
            head: n2,
            first: n1,
            tail_copy: n1,
            cache_bound: bound,
            cache_additions: AtomicUint::new(0),
            cache_subtractions: AtomicUint::new(0),
        }
    }

    /// Pushes a new value onto this queue. Note that to use this function
    /// safely, it must be externally guaranteed that there is only one pusher.
    pub fn push(&mut self, t: T) {
        unsafe {
            // Acquire a node (which either uses a cached one or allocates a new
            // one), and then append this to the 'head' node.
            let n = self.alloc();
            assert!((*n).value.is_none());
            (*n).value = Some(t);
            (*n).next.store(0 as *mut Node<T>, Relaxed);
            (*self.head).next.store(n, Release);
            self.head = n;
        }
    }

    unsafe fn alloc(&mut self) -> *mut Node<T> {
        // First try to see if we can consume the 'first' node for our uses.
        // We try to avoid as many atomic instructions as possible here, so
        // the addition to cache_subtractions is not atomic (plus we're the
        // only one subtracting from the cache).
        if self.first != self.tail_copy {
            if self.cache_bound > 0 {
                let b = self.cache_subtractions.load(Relaxed);
                self.cache_subtractions.store(b + 1, Relaxed);
            }
            let ret = self.first;
            self.first = (*ret).next.load(Relaxed);
            return ret;
        }
        // If the above fails, then update our copy of the tail and try
        // again.
        self.tail_copy = self.tail_prev.load(Acquire);
        if self.first != self.tail_copy {
            if self.cache_bound > 0 {
                let b = self.cache_subtractions.load(Relaxed);
                self.cache_subtractions.store(b + 1, Relaxed);
            }
            let ret = self.first;
            self.first = (*ret).next.load(Relaxed);
            return ret;
        }
        // If all of that fails, then we have to allocate a new node
        // (there's nothing in the node cache).
        Node::new()
    }

    /// Attempts to pop a value from this queue. Remember that to use this type
    /// safely you must ensure that there is only one popper at a time.
    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            // The `tail` node is not actually a used node, but rather a
            // sentinel from where we should start popping from. Hence, look at
            // tail's next field and see if we can use it. If we do a pop, then
            // the current tail node is a candidate for going into the cache.
            let tail = self.tail;
            let next = (*tail).next.load(Acquire);
            if next.is_null() { return None }
            assert!((*next).value.is_some());
            let ret = (*next).value.take();

            self.tail = next;
            if self.cache_bound == 0 {
                self.tail_prev.store(tail, Release);
            } else {
                // FIXME: this is dubious with overflow.
                let additions = self.cache_additions.load(Relaxed);
                let subtractions = self.cache_subtractions.load(Relaxed);
                let size = additions - subtractions;

                if size < self.cache_bound {
                    self.tail_prev.store(tail, Release);
                    self.cache_additions.store(additions + 1, Relaxed);
                } else {
                    (*self.tail_prev.load(Relaxed)).next.store(next, Relaxed);
                    // We have successfully erased all references to 'tail', so
                    // now we can safely drop it.
                    let _: ~Node<T> = cast::transmute(tail);
                }
            }
            return ret;
        }
    }

    /// Attempts to peek at the head of the queue, returning `None` if the queue
    /// has no data currently
    pub fn peek<'a>(&'a mut self) -> Option<&'a mut T> {
        // This is essentially the same as above with all the popping bits
        // stripped out.
        unsafe {
            let tail = self.tail;
            let next = (*tail).next.load(Acquire);
            if next.is_null() { return None }
            return (*next).value.as_mut();
        }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Queue<T> {
    fn drop(&mut self) {
        unsafe {
            let mut cur = self.first;
            while !cur.is_null() {
                let next = (*cur).next.load(Relaxed);
                let _n: ~Node<T> = cast::transmute(cur);
                cur = next;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use native;
    use super::Queue;
    use sync::arc::UnsafeArc;

    #[test]
    fn smoke() {
        let mut q = Queue::new(0);
        q.push(1);
        q.push(2);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), None);
        q.push(3);
        q.push(4);
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn drop_full() {
        let mut q = Queue::new(0);
        q.push(~1);
        q.push(~2);
    }

    #[test]
    fn smoke_bound() {
        let mut q = Queue::new(1);
        q.push(1);
        q.push(2);
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), None);
        q.push(3);
        q.push(4);
        assert_eq!(q.pop(), Some(3));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn stress() {
        stress_bound(0);
        stress_bound(1);

        fn stress_bound(bound: uint) {
            let (a, b) = UnsafeArc::new2(Queue::new(bound));
            let (tx, rx) = channel();
            native::task::spawn(proc() {
                for _ in range(0, 100000) {
                    loop {
                        match unsafe { (*b.get()).pop() } {
                            Some(1) => break,
                            Some(_) => fail!(),
                            None => {}
                        }
                    }
                }
                tx.send(());
            });
            for _ in range(0, 100000) {
                unsafe { (*a.get()).push(1); }
            }
            rx.recv();
        }
    }
}
