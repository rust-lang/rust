// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use container::Container;
use kinds::Owned;
use vec::OwnedVector;
use cell::Cell;
use option::*;
use unstable::sync::{Exclusive, exclusive};
use clone::Clone;

pub struct MessageQueue<T> {
    // XXX: Another mystery bug fixed by boxing this lock
    priv queue: ~Exclusive<~[T]>
}

impl<T: Owned> MessageQueue<T> {
    pub fn new() -> MessageQueue<T> {
        MessageQueue {
            queue: ~exclusive(~[])
        }
    }

    pub fn push(&mut self, value: T) {
        unsafe {
            let value = Cell(value);
            self.queue.with(|q| q.push(value.take()) );
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            do self.queue.with |q| {
                if !q.is_empty() {
                    Some(q.shift())
                } else {
                    None
                }
            }
        }
    }
}

impl<T> Clone for MessageQueue<T> {
    fn clone(&self) -> MessageQueue<T> {
        MessageQueue {
            queue: self.queue.clone()
        }
    }
}
