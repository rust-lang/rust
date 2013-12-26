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

// http://www.1024cores.net/home/lock-free-algorithms
//                         /queues/non-intrusive-mpsc-node-based-queue

use cast;
use clone::Clone;
use kinds::Send;
use ops::Drop;
use option::{Option, None, Some};
use ptr::RawPtr;
use sync::arc::UnsafeArc;
use sync::atomics::{AtomicPtr, Release, Acquire, AcqRel, Relaxed};

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

struct State<T, P> {
    head: AtomicPtr<Node<T>>,
    tail: *mut Node<T>,
    packet: P,
}

/// The consumer half of this concurrent queue. This half is used to receive
/// data from the producers.
pub struct Consumer<T, P> {
    priv state: UnsafeArc<State<T, P>>,
}

/// The production half of the concurrent queue. This handle may be cloned in
/// order to make handles for new producers.
pub struct Producer<T, P> {
    priv state: UnsafeArc<State<T, P>>,
}

impl<T: Send, P: Send> Clone for Producer<T, P> {
    fn clone(&self) -> Producer<T, P> {
        Producer { state: self.state.clone() }
    }
}

/// Creates a new MPSC queue. The given argument `p` is a user-defined "packet"
/// of information which will be shared by the consumer and the producer which
/// can be re-acquired via the `packet` function. This is helpful when extra
/// state is shared between the producer and consumer, but note that there is no
/// synchronization performed of this data.
pub fn queue<T: Send, P: Send>(p: P) -> (Consumer<T, P>, Producer<T, P>) {
    unsafe {
        let (a, b) = UnsafeArc::new2(State::new(p));
        (Consumer { state: a }, Producer { state: b })
    }
}

impl<T> Node<T> {
    unsafe fn new(v: Option<T>) -> *mut Node<T> {
        cast::transmute(~Node {
            next: AtomicPtr::new(0 as *mut Node<T>),
            value: v,
        })
    }
}

impl<T: Send, P: Send> State<T, P> {
    unsafe fn new(p: P) -> State<T, P> {
        let stub = Node::new(None);
        State {
            head: AtomicPtr::new(stub),
            tail: stub,
            packet: p,
        }
    }

    unsafe fn push(&mut self, t: T) {
        let n = Node::new(Some(t));
        let prev = self.head.swap(n, AcqRel);
        (*prev).next.store(n, Release);
    }

    unsafe fn pop(&mut self) -> PopResult<T> {
        let tail = self.tail;
        let next = (*tail).next.load(Acquire);

        if !next.is_null() {
            self.tail = next;
            assert!((*tail).value.is_none());
            assert!((*next).value.is_some());
            let ret = (*next).value.take_unwrap();
            let _: ~Node<T> = cast::transmute(tail);
            return Data(ret);
        }

        if self.head.load(Acquire) == tail {Empty} else {Inconsistent}
    }
}

#[unsafe_destructor]
impl<T: Send, P: Send> Drop for State<T, P> {
    fn drop(&mut self) {
        unsafe {
            let mut cur = self.tail;
            while !cur.is_null() {
                let next = (*cur).next.load(Relaxed);
                let _: ~Node<T> = cast::transmute(cur);
                cur = next;
            }
        }
    }
}

impl<T: Send, P: Send> Producer<T, P> {
    /// Pushes a new value onto this queue.
    pub fn push(&mut self, value: T) {
        unsafe { (*self.state.get()).push(value) }
    }
    /// Gets an unsafe pointer to the user-defined packet shared by the
    /// producers and the consumer. Note that care must be taken to ensure that
    /// the lifetime of the queue outlives the usage of the returned pointer.
    pub unsafe fn packet(&self) -> *mut P {
        &mut (*self.state.get()).packet as *mut P
    }
}

impl<T: Send, P: Send> Consumer<T, P> {
    /// Pops some data from this queue.
    ///
    /// Note that the current implementation means that this function cannot
    /// return `Option<T>`. It is possible for this queue to be in an
    /// inconsistent state where many pushes have suceeded and completely
    /// finished, but pops cannot return `Some(t)`. This inconsistent state
    /// happens when a pusher is pre-empted at an inopportune moment.
    ///
    /// This inconsistent state means that this queue does indeed have data, but
    /// it does not currently have access to it at this time.
    pub fn pop(&mut self) -> PopResult<T> {
        unsafe { (*self.state.get()).pop() }
    }
    /// Attempts to pop data from this queue, but doesn't attempt too hard. This
    /// will canonicalize inconsistent states to a `None` value.
    pub fn casual_pop(&mut self) -> Option<T> {
        match self.pop() {
            Data(t) => Some(t),
            Empty | Inconsistent => None,
        }
    }
    /// Gets an unsafe pointer to the underlying user-defined packet. See
    /// `Producer.packet` for more information.
    pub unsafe fn packet(&self) -> *mut P {
        &mut (*self.state.get()).packet as *mut P
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    use super::{queue, Data, Empty, Inconsistent};
    use native;

    #[test]
    fn test_full() {
        let (_, mut p) = queue(());
        p.push(~1);
        p.push(~2);
    }

    #[test]
    fn test() {
        let nthreads = 8u;
        let nmsgs = 1000u;
        let (mut c, p) = queue(());
        match c.pop() {
            Empty => {}
            Inconsistent | Data(..) => fail!()
        }
        let (port, chan) = SharedChan::new();

        for _ in range(0, nthreads) {
            let q = p.clone();
            let chan = chan.clone();
            do native::task::spawn {
                let mut q = q;
                for i in range(0, nmsgs) {
                    q.push(i);
                }
                chan.send(());
            }
        }

        let mut i = 0u;
        while i < nthreads * nmsgs {
            match c.pop() {
                Empty | Inconsistent => {},
                Data(_) => { i += 1 }
            }
        }
        for _ in range(0, nthreads) {
            port.recv();
        }
    }
}

