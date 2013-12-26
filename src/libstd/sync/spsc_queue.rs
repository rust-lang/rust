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
use sync::arc::UnsafeArc;
use sync::atomics::{AtomicPtr, Relaxed, AtomicUint, Acquire, Release};

// Node within the linked list queue of messages to send
struct Node<T> {
    // XXX: this could be an uninitialized T if we're careful enough, and
    //      that would reduce memory usage (and be a bit faster).
    //      is it worth it?
    value: Option<T>,           // nullable for re-use of nodes
    next: AtomicPtr<Node<T>>,   // next node in the queue
}

// The producer/consumer halves both need access to the `tail` field, and if
// they both have access to that we may as well just give them both access
// to this whole structure.
struct State<T, P> {
    // consumer fields
    tail: *mut Node<T>, // where to pop from
    tail_prev: AtomicPtr<Node<T>>, // where to pop from

    // producer fields
    head: *mut Node<T>,      // where to push to
    first: *mut Node<T>,     // where to get new nodes from
    tail_copy: *mut Node<T>, // between first/tail

    // Cache maintenance fields. Additions and subtractions are stored
    // separately in order to allow them to use nonatomic addition/subtraction.
    cache_bound: uint,
    cache_additions: AtomicUint,
    cache_subtractions: AtomicUint,

    packet: P,
}

/// Producer half of this queue. This handle is used to push data to the
/// consumer.
pub struct Producer<T, P> {
    priv state: UnsafeArc<State<T, P>>,
}

/// Consumer half of this queue. This handle is used to receive data from the
/// producer.
pub struct Consumer<T, P> {
    priv state: UnsafeArc<State<T, P>>,
}

/// Creates a new queue. The producer returned is connected to the consumer to
/// push all data to the consumer.
///
/// # Arguments
///
///   * `bound` - This queue implementation is implemented with a linked list,
///               and this means that a push is always a malloc. In order to
///               amortize this cost, an internal cache of nodes is maintained
///               to prevent a malloc from always being necessary. This bound is
///               the limit on the size of the cache (if desired). If the value
///               is 0, then the cache has no bound. Otherwise, the cache will
///               never grow larger than `bound` (although the queue itself
///               could be much larger.
///
///   * `p` - This is the user-defined packet of data which will also be shared
///           between the producer and consumer.
pub fn queue<T: Send, P: Send>(bound: uint,
                               p: P) -> (Consumer<T, P>, Producer<T, P>)
{
    let n1 = Node::new();
    let n2 = Node::new();
    unsafe { (*n1).next.store(n2, Relaxed) }
    let state = State {
        tail: n2,
        tail_prev: AtomicPtr::new(n1),
        head: n2,
        first: n1,
        tail_copy: n1,
        cache_bound: bound,
        cache_additions: AtomicUint::new(0),
        cache_subtractions: AtomicUint::new(0),
        packet: p,
    };
    let (arc1, arc2) = UnsafeArc::new2(state);
    (Consumer { state: arc1 }, Producer { state: arc2 })
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

impl<T: Send, P: Send> Producer<T, P> {
    /// Pushes data onto the queue
    pub fn push(&mut self, t: T) {
        unsafe { (*self.state.get()).push(t) }
    }
    /// Tests whether the queue is empty. Note that if this function returns
    /// `false`, the return value is significant, but if the return value is
    /// `true` then almost no meaning can be attached to the return value.
    pub fn is_empty(&self) -> bool {
        unsafe { (*self.state.get()).is_empty() }
    }
    /// Acquires an unsafe pointer to the underlying user-defined packet. Note
    /// that care must be taken to ensure that the queue outlives the usage of
    /// the packet (because it is an unsafe pointer).
    pub unsafe fn packet(&self) -> *mut P {
        &mut (*self.state.get()).packet as *mut P
    }
}

impl<T: Send, P: Send> Consumer<T, P> {
    /// Pops some data from this queue, returning `None` when the queue is
    /// empty.
    pub fn pop(&mut self) -> Option<T> {
        unsafe { (*self.state.get()).pop() }
    }
    /// Same function as the producer's `packet` method.
    pub unsafe fn packet(&self) -> *mut P {
        &mut (*self.state.get()).packet as *mut P
    }
}

impl<T: Send, P: Send> State<T, P> {
    // remember that there is only one thread executing `push` (and only one
    // thread executing `pop`)
    unsafe fn push(&mut self, t: T) {
        // Acquire a node (which either uses a cached one or allocates a new
        // one), and then append this to the 'head' node.
        let n = self.alloc();
        assert!((*n).value.is_none());
        (*n).value = Some(t);
        (*n).next.store(0 as *mut Node<T>, Relaxed);
        (*self.head).next.store(n, Release);
        self.head = n;
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

    // remember that there is only one thread executing `pop` (and only one
    // thread executing `push`)
    unsafe fn pop(&mut self) -> Option<T> {
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
            // XXX: this is dubious with overflow.
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

    unsafe fn is_empty(&self) -> bool {
        let tail = self.tail;
        let next = (*tail).next.load(Acquire);
        return next.is_null();
    }
}

#[unsafe_destructor]
impl<T: Send, P: Send> Drop for State<T, P> {
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
    use super::queue;
    use native;

    #[test]
    fn smoke() {
        let (mut c, mut p) = queue(0, ());
        p.push(1);
        p.push(2);
        assert_eq!(c.pop(), Some(1));
        assert_eq!(c.pop(), Some(2));
        assert_eq!(c.pop(), None);
        p.push(3);
        p.push(4);
        assert_eq!(c.pop(), Some(3));
        assert_eq!(c.pop(), Some(4));
        assert_eq!(c.pop(), None);
    }

    #[test]
    fn drop_full() {
        let (_, mut p) = queue(0, ());
        p.push(~1);
        p.push(~2);
    }

    #[test]
    fn smoke_bound() {
        let (mut c, mut p) = queue(1, ());
        p.push(1);
        p.push(2);
        assert_eq!(c.pop(), Some(1));
        assert_eq!(c.pop(), Some(2));
        assert_eq!(c.pop(), None);
        p.push(3);
        p.push(4);
        assert_eq!(c.pop(), Some(3));
        assert_eq!(c.pop(), Some(4));
        assert_eq!(c.pop(), None);
    }

    #[test]
    fn stress() {
        stress_bound(0);
        stress_bound(1);

        fn stress_bound(bound: uint) {
            let (c, mut p) = queue(bound, ());
            let (port, chan) = Chan::new();
            do native::task::spawn {
                let mut c = c;
                for _ in range(0, 100000) {
                    loop {
                        match c.pop() {
                            Some(1) => break,
                            Some(_) => fail!(),
                            None => {}
                        }
                    }
                }
                chan.send(());
            }
            for _ in range(0, 100000) {
                p.push(1);
            }
            port.recv();
        }
    }
}
