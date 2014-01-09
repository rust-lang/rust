// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A lock-free multi-producer, single consumer queue.
//!
//! This module implements an intrusive MPSC queue. This queue is incredibly
//! unsafe (due to use of unsafe pointers for nodes), and hence is not public.

use cast;
use kinds::Send;
use option::{Option, Some, None};
use sync::atomics;

// NB: all links are done as AtomicUint instead of AtomicPtr to allow for static
// initialization.

pub struct Node<T> {
    next: *mut Node<T>,
    data: T,
}

pub struct Queue<T> {
    producer: atomics::AtomicUint,
    consumer: *mut Node<T>,
}

impl<T: Send> Queue<T> {
    pub fn new() -> Queue<T> {
        Queue {
            producer: atomics::AtomicUint::new(0),
            consumer: 0 as *mut Node<T>,
        }
    }

    pub unsafe fn push(&self, node: *mut Node<T>) {
        // prepend the node to the producer queue
        let mut a = 0;
        loop {
            (*node).next = cast::transmute(a);
            let v = self.producer.compare_and_swap(a, node as uint, atomics::Release);
            if a == v {
                return;
            }
            a = v;
        }
    }

    /// This has worst case O(n) because it needs to reverse the queue
    /// However it is of course amortized O(1)
    pub unsafe fn pop(&self) -> Option<*mut Node<T>> {
        // self.consumer is only used by the single consumer, so let's get an &mut to it
        let Queue {producer: ref ref_producer, consumer: ref ref_consumer} = *self;
        let mut_consumer: &mut *mut Node<T> = cast::transmute(ref_consumer);

        let node = *mut_consumer;
        if node != 0 as *mut Node<T> {
            // pop from the consumer queue if non-empty
            *mut_consumer = (*node).next;
            Some(node)
        } else {
            // otherwise steal the producer queue, reverse it, take the last element
            // and store the rest as the consumer queue
            let mut node: *mut Node<T> = cast::transmute(ref_producer.swap(0, atomics::Acquire));
            if node != 0 as *mut Node<T> {
                let mut prev = 0 as *mut Node<T>;

                loop {
                    let next = (*node).next;
                    if next == 0 as *mut Node<T> {break};
                    (*node).next = prev;
                    prev = node;
                    node = next;
                }
                *mut_consumer = prev;
                Some(node)
            } else {
                None
            }
        }
    }
}

impl<T: Send> Node<T> {
    pub fn new(t: T) -> Node<T> {
        Node {
            data: t,
            next: 0 as *mut Node<T>,
        }
    }
}

