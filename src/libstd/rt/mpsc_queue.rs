/* Multi-producer/single-consumer queue
 * Copyright (c) 2010-2011 Dmitry Vyukov. All rights reserved.
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

use unstable::sync::UnsafeArc;
use unstable::atomics::{AtomicPtr,Relaxed,Release,Acquire};
use ptr::{mut_null, to_mut_unsafe_ptr};
use cast;
use option::*;
use clone::Clone;
use default::Default;
use kinds::Send;
use fmt;

struct Node<T> {
    next: AtomicPtr<Node<T>>,
    value: Option<T>,
}

impl<T> Node<T> {
    fn new(value: T) -> Node<T> {
        Node{next: AtomicPtr::new(mut_null()), value: Some(value)}
    }
}

impl<T> Default for Node<T> {
    fn default() -> Node<T> {
        Node{next: AtomicPtr::new(mut_null()), value: None}
    }
}

struct State<T> {
    stub: Node<T>,
    head: AtomicPtr<Node<T>>,
    tail: *mut Node<T>,
}

struct Queue<T> {
    priv state: UnsafeArc<State<T>>,
}

impl<T: Send> Clone for Queue<T> {
    fn clone(&self) -> Queue<T> {
        Queue {
            state: self.state.clone()
        }
    }
}

impl<T: Send> fmt::Default for Queue<T> {
    fn fmt(value: &Queue<T>, f: &mut fmt::Formatter) {
        write!(f.buf, "Queue({})", value.state.get());
    }
}

impl<T: Send> Queue<T> {
    pub fn new() -> Queue<T> {
        let mut q = Queue{state: UnsafeArc::new(State {
            stub: Default::default(),
            head: AtomicPtr::new(mut_null()),
            tail: mut_null(),
        })};
        let stub = q.get_stub_unsafe();
        q.get_head().store(stub, Relaxed);
        q.set_tail(stub);
        q
    }

    pub fn push(&mut self, value: T) {
        unsafe {
            let node = cast::transmute(~Node::new(value));
            self.push_node(node);
        }
    }

    fn push_node(&mut self, node: *mut Node<T>) {
        unsafe {
            (*node).next.store(mut_null(), Release);
            let prev = (*self.state.get()).head.swap(node, Relaxed);
            (*prev).next.store(node, Release);
        }
    }

    fn get_stub_unsafe(&mut self) -> *mut Node<T> {
        unsafe { to_mut_unsafe_ptr(&mut (*self.state.get()).stub) }
    }

    fn get_head(&mut self) -> &mut AtomicPtr<Node<T>> {
        unsafe { &mut (*self.state.get()).head }
    }

    fn get_tail(&mut self) -> *mut Node<T> {
        unsafe { (*self.state.get()).tail }
    }

    fn set_tail(&mut self, tail: *mut Node<T>) {
        unsafe { (*self.state.get()).tail = tail }
    }

    pub fn casual_pop(&mut self) -> Option<T> {
        self.pop()
    }

    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            let mut tail = self.get_tail();
            let mut next = (*tail).next.load(Acquire);
            let stub = self.get_stub_unsafe();
            if tail == stub {
                if mut_null() == next {
                    return None
                }
                self.set_tail(next);
                tail = next;
                next = (*next).next.load(Acquire);
            }
            if next != mut_null() {
                let tail: ~Node<T> = cast::transmute(tail);
                self.set_tail(next);
                return tail.value
            }
            let head = self.get_head().load(Relaxed);
            if tail != head {
                return None
            }
            self.push_node(stub);
            next = (*tail).next.load(Acquire);
            if next != mut_null() {
                let tail: ~Node<T> = cast::transmute(tail);
                self.set_tail(next);
                return tail.value
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use option::*;
    use task;
    use comm;
    use fmt;
    use super::Queue;

    #[test]
    fn test() {
        let nthreads = 8u;
        let nmsgs = 1000u;
        let mut q = Queue::new();
        assert_eq!(None, q.pop());

        for _ in range(0, nthreads) {
            let (port, chan)  = comm::stream();
            chan.send(q.clone());
            do task::spawn_sched(task::SingleThreaded) {
                let mut q = port.recv();
                for i in range(0, nmsgs) {
                    q.push(i);
                }
            }
        }

        let mut i = 0u;
        loop {
            match q.pop() {
                None => {},
                Some(_) => {
                    i += 1;
                    if i == nthreads*nmsgs { break }
                }
            }
        }
    }
}

