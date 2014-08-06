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
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL DMITRY VYUKOV OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of Dmitry Vyukov.
 */

//! A mostly lock-free multi-producer, single consumer queue.
//!
//! This module implements an intrusive MPSC queue. This queue is incredibly
//! unsafe (due to use of unsafe pointers for nodes), and hence is not public.

#![experimental]

// http://www.1024cores.net/home/lock-free-algorithms
//                         /queues/intrusive-mpsc-node-based-queue

use core::prelude::*;

use core::atomic;
use core::mem;
use core::cell::UnsafeCell;

// NB: all links are done as AtomicUint instead of AtomicPtr to allow for static
// initialization.

pub struct Node<T> {
    pub next: atomic::AtomicUint,
    pub data: T,
}

pub struct DummyNode {
    pub next: atomic::AtomicUint,
}

pub struct Queue<T> {
    pub head: atomic::AtomicUint,
    pub tail: UnsafeCell<*mut Node<T>>,
    pub stub: DummyNode,
}

impl<T: Send> Queue<T> {
    pub fn new() -> Queue<T> {
        Queue {
            head: atomic::AtomicUint::new(0),
            tail: UnsafeCell::new(0 as *mut Node<T>),
            stub: DummyNode {
                next: atomic::AtomicUint::new(0),
            },
        }
    }

    pub unsafe fn push(&self, node: *mut Node<T>) {
        (*node).next.store(0, atomic::Release);
        let prev = self.head.swap(node as uint, atomic::AcqRel);

        // Note that this code is slightly modified to allow static
        // initialization of these queues with rust's flavor of static
        // initialization.
        if prev == 0 {
            self.stub.next.store(node as uint, atomic::Release);
        } else {
            let prev = prev as *mut Node<T>;
            (*prev).next.store(node as uint, atomic::Release);
        }
    }

    /// You'll note that the other MPSC queue in std::sync is non-intrusive and
    /// returns a `PopResult` here to indicate when the queue is inconsistent.
    /// An "inconsistent state" in the other queue means that a pusher has
    /// pushed, but it hasn't finished linking the rest of the chain.
    ///
    /// This queue also suffers from this problem, but I currently haven't been
    /// able to detangle when this actually happens. This code is translated
    /// verbatim from the website above, and is more complicated than the
    /// non-intrusive version.
    ///
    /// Right now consumers of this queue must be ready for this fact. Just
    /// because `pop` returns `None` does not mean that there is not data
    /// on the queue.
    pub unsafe fn pop(&self) -> Option<*mut Node<T>> {
        let tail = *self.tail.get();
        let mut tail = if !tail.is_null() {tail} else {
            mem::transmute(&self.stub)
        };
        let mut next = (*tail).next(atomic::Relaxed);
        if tail as uint == &self.stub as *const DummyNode as uint {
            if next.is_null() {
                return None;
            }
            *self.tail.get() = next;
            tail = next;
            next = (*next).next(atomic::Relaxed);
        }
        if !next.is_null() {
            *self.tail.get() = next;
            return Some(tail);
        }
        let head = self.head.load(atomic::Acquire) as *mut Node<T>;
        if tail != head {
            return None;
        }
        let stub = mem::transmute(&self.stub);
        self.push(stub);
        next = (*tail).next(atomic::Relaxed);
        if !next.is_null() {
            *self.tail.get() = next;
            return Some(tail);
        }
        return None
    }
}

impl<T: Send> Node<T> {
    pub fn new(t: T) -> Node<T> {
        Node {
            data: t,
            next: atomic::AtomicUint::new(0),
        }
    }
    pub unsafe fn next(&self, ord: atomic::Ordering) -> *mut Node<T> {
        mem::transmute::<uint, *mut Node<T>>(self.next.load(ord))
    }
}
