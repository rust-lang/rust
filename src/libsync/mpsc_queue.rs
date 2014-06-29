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

//! A mostly lock-free multi-producer, single consumer queue.
//!
//! This module contains an implementation of a concurrent MPSC queue. This
//! queue can be used to share data between tasks, and is also used as the
//! building block of channels in rust.
//!
//! Note that the current implementation of this queue has a caveat of the `pop`
//! method, and see the method for more information about it. Due to this
//! caveat, this queue may not be appropriate for all use-cases.

#![experimental]

// http://www.1024cores.net/home/lock-free-algorithms
//                         /queues/non-intrusive-mpsc-node-based-queue

use core::prelude::*;

use alloc::owned::Box;
use core::mem;
use core::ty::Unsafe;

use atomics::{AtomicPtr, Release, Acquire, AcqRel, Relaxed};

/// A result of the `pop` function.
pub enum PopResult<T> {
    /// Some data has been popped
    Data(T),
    /// The queue is empty
    Empty,
    /// The queue is in an inconsistent state. Popping data should succeed, but
    /// some pushers have yet to make enough progress in order allow a pop to
    /// succeed. It is recommended that a pop() occur "in the near future" in
    /// order to see if the sender has made progress or not
    Inconsistent,
}

struct Node<T> {
    next: AtomicPtr<Node<T>>,
    value: Option<T>,
}

/// The multi-producer single-consumer structure. This is not cloneable, but it
/// may be safely shared so long as it is guaranteed that there is only one
/// popper at a time (many pushers are allowed).
pub struct Queue<T> {
    head: AtomicPtr<Node<T>>,
    tail: Unsafe<*mut Node<T>>,
}

impl<T> Node<T> {
    unsafe fn new(v: Option<T>) -> *mut Node<T> {
        mem::transmute(box Node {
            next: AtomicPtr::new(0 as *mut Node<T>),
            value: v,
        })
    }
}

impl<T: Send> Queue<T> {
    /// Creates a new queue that is safe to share among multiple producers and
    /// one consumer.
    pub fn new() -> Queue<T> {
        let stub = unsafe { Node::new(None) };
        Queue {
            head: AtomicPtr::new(stub),
            tail: Unsafe::new(stub),
        }
    }

    /// Pushes a new value onto this queue.
    pub fn push(&self, t: T) {
        unsafe {
            let n = Node::new(Some(t));
            let prev = self.head.swap(n, AcqRel);
            (*prev).next.store(n, Release);
        }
    }

    /// Pops some data from this queue.
    ///
    /// Note that the current implementation means that this function cannot
    /// return `Option<T>`. It is possible for this queue to be in an
    /// inconsistent state where many pushes have succeeded and completely
    /// finished, but pops cannot return `Some(t)`. This inconsistent state
    /// happens when a pusher is pre-empted at an inopportune moment.
    ///
    /// This inconsistent state means that this queue does indeed have data, but
    /// it does not currently have access to it at this time.
    pub fn pop(&self) -> PopResult<T> {
        unsafe {
            let tail = *self.tail.get();
            let next = (*tail).next.load(Acquire);

            if !next.is_null() {
                *self.tail.get() = next;
                assert!((*tail).value.is_none());
                assert!((*next).value.is_some());
                let ret = (*next).value.take_unwrap();
                let _: Box<Node<T>> = mem::transmute(tail);
                return Data(ret);
            }

            if self.head.load(Acquire) == tail {Empty} else {Inconsistent}
        }
    }

    /// Attempts to pop data from this queue, but doesn't attempt too hard. This
    /// will canonicalize inconsistent states to a `None` value.
    pub fn casual_pop(&self) -> Option<T> {
        match self.pop() {
            Data(t) => Some(t),
            Empty | Inconsistent => None,
        }
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for Queue<T> {
    fn drop(&mut self) {
        unsafe {
            let mut cur = *self.tail.get();
            while !cur.is_null() {
                let next = (*cur).next.load(Relaxed);
                let _: Box<Node<T>> = mem::transmute(cur);
                cur = next;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::*;

    use alloc::arc::Arc;

    use native;
    use super::{Queue, Data, Empty, Inconsistent};

    #[test]
    fn test_full() {
        let q = Queue::new();
        q.push(box 1i);
        q.push(box 2i);
    }

    #[test]
    fn test() {
        let nthreads = 8u;
        let nmsgs = 1000u;
        let q = Queue::new();
        match q.pop() {
            Empty => {}
            Inconsistent | Data(..) => fail!()
        }
        let (tx, rx) = channel();
        let q = Arc::new(q);

        for _ in range(0, nthreads) {
            let tx = tx.clone();
            let q = q.clone();
            native::task::spawn(proc() {
                for i in range(0, nmsgs) {
                    q.push(i);
                }
                tx.send(());
            });
        }

        let mut i = 0u;
        while i < nthreads * nmsgs {
            match q.pop() {
                Empty | Inconsistent => {},
                Data(_) => { i += 1 }
            }
        }
        drop(tx);
        for _ in range(0, nthreads) {
            rx.recv();
        }
    }
}
