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

// http://www.1024cores.net/home/lock-free-algorithms
//                         /queues/non-intrusive-mpsc-node-based-queue

use cast;
use clone::Clone;
use kinds::Send;
use ops::Drop;
use option::{Option, None, Some};
use unstable::atomics::{AtomicPtr, Release, Acquire, AcqRel, Relaxed};
use unstable::sync::UnsafeArc;

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

pub struct Consumer<T, P> {
    priv state: UnsafeArc<State<T, P>>,
}

pub struct Producer<T, P> {
    priv state: UnsafeArc<State<T, P>>,
}

impl<T: Send, P: Send> Clone for Producer<T, P> {
    fn clone(&self) -> Producer<T, P> {
        Producer { state: self.state.clone() }
    }
}

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
    pub unsafe fn new(p: P) -> State<T, P> {
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

    unsafe fn is_empty(&mut self) -> bool {
        return (*self.tail).next.load(Acquire).is_null();
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
    pub fn push(&mut self, value: T) {
        unsafe { (*self.state.get()).push(value) }
    }
    pub fn is_empty(&self) -> bool {
        unsafe{ (*self.state.get()).is_empty() }
    }
    pub unsafe fn packet(&self) -> *mut P {
        &mut (*self.state.get()).packet as *mut P
    }
}

impl<T: Send, P: Send> Consumer<T, P> {
    pub fn pop(&mut self) -> PopResult<T> {
        unsafe { (*self.state.get()).pop() }
    }
    pub fn casual_pop(&mut self) -> Option<T> {
        match self.pop() {
            Data(t) => Some(t),
            Empty | Inconsistent => None,
        }
    }
    pub unsafe fn packet(&self) -> *mut P {
        &mut (*self.state.get()).packet as *mut P
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;

    use task;
    use super::{queue, Data, Empty, Inconsistent};

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

        for _ in range(0, nthreads) {
            let q = p.clone();
            do task::spawn_sched(task::SingleThreaded) {
                let mut q = q;
                for i in range(0, nmsgs) {
                    q.push(i);
                }
            }
        }

        let mut i = 0u;
        while i < nthreads * nmsgs {
            match c.pop() {
                Empty | Inconsistent => {},
                Data(_) => { i += 1 }
            }
        }
    }
}

