// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A concurrent queue that supports multiple producers and a
//! single consumer.

use kinds::Send;
use vec::OwnedVector;
use option::Option;
use clone::Clone;
use rt::mpsc_queue::Queue;

pub struct MessageQueue<T> {
    priv queue: Queue<T>
}

impl<T: Send> MessageQueue<T> {
    pub fn new() -> MessageQueue<T> {
        MessageQueue {
            queue: Queue::new()
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.queue.push(value)
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.queue.pop()
    }

    /// A pop that may sometimes miss enqueued elements, but is much faster
    /// to give up without doing any synchronization
    #[inline]
    pub fn casual_pop(&mut self) -> Option<T> {
        self.queue.pop()
    }
}

impl<T: Send> Clone for MessageQueue<T> {
    fn clone(&self) -> MessageQueue<T> {
        MessageQueue {
            queue: self.queue.clone()
        }
    }
}
