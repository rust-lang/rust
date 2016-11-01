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
//! concurrently between two threads. This data structure is safe to use and
//! enforces the semantics that there is one pusher and one popper.

use alloc::boxed::Box;
use core::ptr;
use core::cell::UnsafeCell;

use sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

// Node within the linked list queue of messages to send
struct Node<T> {
    // FIXME: this could be an uninitialized T if we're careful enough, and
    //      that would reduce memory usage (and be a bit faster).
    //      is it worth it?
    value: Option<T>,           // nullable for re-use of nodes
    next: AtomicPtr<Node<T>>,   // next node in the queue
}

/// The single-producer single-consumer queue. This structure is not cloneable,
/// but it can be safely shared in an Arc if it is guaranteed that there
/// is only one popper and one pusher touching the queue at any one point in
/// time.
pub struct Queue<T> {
    // consumer fields
    tail: UnsafeCell<*mut Node<T>>, // where to pop from
    tail_prev: AtomicPtr<Node<T>>, // where to pop from

    // producer fields
    head: UnsafeCell<*mut Node<T>>,      // where to push to
    first: UnsafeCell<*mut Node<T>>,     // where to get new nodes from
    tail_copy: UnsafeCell<*mut Node<T>>, // between first/tail

    // Cache maintenance fields. Additions and subtractions are stored
    // separately in order to allow them to use nonatomic addition/subtraction.
    cache_bound: usize,
    cache_additions: AtomicUsize,
    cache_subtractions: AtomicUsize,
}

unsafe impl<T: Send> Send for Queue<T> { }

unsafe impl<T: Send> Sync for Queue<T> { }

impl<T> Node<T> {
    fn new() -> *mut Node<T> {
        Box::into_raw(box Node {
            value: None,
            next: AtomicPtr::new(ptr::null_mut::<Node<T>>()),
        })
    }
}

impl<T> Queue<T> {
    /// Creates a new queue.
    ///
    /// This is unsafe as the type system doesn't enforce a single
    /// consumer-producer relationship. It also allows the consumer to `pop`
    /// items while there is a `peek` active due to all methods having a
    /// non-mutable receiver.
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
    pub unsafe fn new(bound: usize) -> Queue<T> {
        let n1 = Node::new();
        let n2 = Node::new();
        (*n1).next.store(n2, Ordering::Relaxed);
        Queue {
            tail: UnsafeCell::new(n2),
            tail_prev: AtomicPtr::new(n1),
            head: UnsafeCell::new(n2),
            first: UnsafeCell::new(n1),
            tail_copy: UnsafeCell::new(n1),
            cache_bound: bound,
            cache_additions: AtomicUsize::new(0),
            cache_subtractions: AtomicUsize::new(0),
        }
    }

    /// Pushes a new value onto this queue. Note that to use this function
    /// safely, it must be externally guaranteed that there is only one pusher.
    pub fn push(&self, t: T) {
        unsafe {
            // Acquire a node (which either uses a cached one or allocates a new
            // one), and then append this to the 'head' node.
            let n = self.alloc();
            assert!((*n).value.is_none());
            (*n).value = Some(t);
            (*n).next.store(ptr::null_mut(), Ordering::Relaxed);
            (**self.head.get()).next.store(n, Ordering::Release);
            *self.head.get() = n;
        }
    }

    unsafe fn alloc(&self) -> *mut Node<T> {
        // First try to see if we can consume the 'first' node for our uses.
        // We try to avoid as many atomic instructions as possible here, so
        // the addition to cache_subtractions is not atomic (plus we're the
        // only one subtracting from the cache).
        if *self.first.get() != *self.tail_copy.get() {
            if self.cache_bound > 0 {
                let b = self.cache_subtractions.load(Ordering::Relaxed);
                self.cache_subtractions.store(b + 1, Ordering::Relaxed);
            }
            let ret = *self.first.get();
            *self.first.get() = (*ret).next.load(Ordering::Relaxed);
            return ret;
        }
        // If the above fails, then update our copy of the tail and try
        // again.
        *self.tail_copy.get() = self.tail_prev.load(Ordering::Acquire);
        if *self.first.get() != *self.tail_copy.get() {
            if self.cache_bound > 0 {
                let b = self.cache_subtractions.load(Ordering::Relaxed);
                self.cache_subtractions.store(b + 1, Ordering::Relaxed);
            }
            let ret = *self.first.get();
            *self.first.get() = (*ret).next.load(Ordering::Relaxed);
            return ret;
        }
        // If all of that fails, then we have to allocate a new node
        // (there's nothing in the node cache).
        Node::new()
    }

    /// Attempts to pop a value from this queue. Remember that to use this type
    /// safely you must ensure that there is only one popper at a time.
    pub fn pop(&self) -> Option<T> {
        unsafe {
            // The `tail` node is not actually a used node, but rather a
            // sentinel from where we should start popping from. Hence, look at
            // tail's next field and see if we can use it. If we do a pop, then
            // the current tail node is a candidate for going into the cache.
            let tail = *self.tail.get();
            let next = (*tail).next.load(Ordering::Acquire);
            if next.is_null() { return None }
            assert!((*next).value.is_some());
            let ret = (*next).value.take();

            *self.tail.get() = next;
            if self.cache_bound == 0 {
                self.tail_prev.store(tail, Ordering::Release);
            } else {
                // FIXME: this is dubious with overflow.
                let additions = self.cache_additions.load(Ordering::Relaxed);
                let subtractions = self.cache_subtractions.load(Ordering::Relaxed);
                let size = additions - subtractions;

                if size < self.cache_bound {
                    self.tail_prev.store(tail, Ordering::Release);
                    self.cache_additions.store(additions + 1, Ordering::Relaxed);
                } else {
                    (*self.tail_prev.load(Ordering::Relaxed))
                          .next.store(next, Ordering::Relaxed);
                    // We have successfully erased all references to 'tail', so
                    // now we can safely drop it.
                    let _: Box<Node<T>> = Box::from_raw(tail);
                }
            }
            ret
        }
    }

    /// Attempts to peek at the head of the queue, returning `None` if the queue
    /// has no data currently
    ///
    /// # Warning
    /// The reference returned is invalid if it is not used before the consumer
    /// pops the value off the queue. If the producer then pushes another value
    /// onto the queue, it will overwrite the value pointed to by the reference.
    pub fn peek(&self) -> Option<&mut T> {
        // This is essentially the same as above with all the popping bits
        // stripped out.
        unsafe {
            let tail = *self.tail.get();
            let next = (*tail).next.load(Ordering::Acquire);
            if next.is_null() { None } else { (*next).value.as_mut() }
        }
    }
}

impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        unsafe {
            let mut cur = *self.first.get();
            while !cur.is_null() {
                let next = (*cur).next.load(Ordering::Relaxed);
                let _n: Box<Node<T>> = Box::from_raw(cur);
                cur = next;
            }
        }
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use sync::Arc;
    use super::Queue;
    use thread;
    use sync::mpsc::channel;

    #[test]
    fn smoke() {
        unsafe {
            let queue = Queue::new(0);
            queue.push(1);
            queue.push(2);
            assert_eq!(queue.pop(), Some(1));
            assert_eq!(queue.pop(), Some(2));
            assert_eq!(queue.pop(), None);
            queue.push(3);
            queue.push(4);
            assert_eq!(queue.pop(), Some(3));
            assert_eq!(queue.pop(), Some(4));
            assert_eq!(queue.pop(), None);
        }
    }

    #[test]
    fn peek() {
        unsafe {
            let queue = Queue::new(0);
            queue.push(vec![1]);

            // Ensure the borrowchecker works
            match queue.peek() {
                Some(vec) => {
                    assert_eq!(&*vec, &[1]);
                },
                None => unreachable!()
            }

            match queue.pop() {
                Some(vec) => {
                    assert_eq!(&*vec, &[1]);
                },
                None => unreachable!()
            }
        }
    }

    #[test]
    fn drop_full() {
        unsafe {
            let q: Queue<Box<_>> = Queue::new(0);
            q.push(box 1);
            q.push(box 2);
        }
    }

    #[test]
    fn smoke_bound() {
        unsafe {
            let q = Queue::new(0);
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
    }

    #[test]
    fn stress() {
        unsafe {
            stress_bound(0);
            stress_bound(1);
        }

        unsafe fn stress_bound(bound: usize) {
            let q = Arc::new(Queue::new(bound));

            let (tx, rx) = channel();
            let q2 = q.clone();
            let _t = thread::spawn(move|| {
                for _ in 0..100000 {
                    loop {
                        match q2.pop() {
                            Some(1) => break,
                            Some(_) => panic!(),
                            None => {}
                        }
                    }
                }
                tx.send(()).unwrap();
            });
            for _ in 0..100000 {
                q.push(1);
            }
            rx.recv().unwrap();
        }
    }
}
