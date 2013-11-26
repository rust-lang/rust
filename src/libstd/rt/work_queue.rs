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
use option::*;
use vec::OwnedVector;
use unstable::sync::Exclusive;
use cell::Cell;
use kinds::Send;
use clone::Clone;

pub struct WorkQueue<T> {
    // XXX: Another mystery bug fixed by boxing this lock
    priv queue: ~Exclusive<~[T]>
}

impl<T: Send> WorkQueue<T> {
    pub fn new() -> WorkQueue<T> {
        WorkQueue {
            queue: ~Exclusive::new(~[])
        }
    }

    pub fn push(&mut self, value: T) {
        unsafe {
            let value = Cell::new(value);
            self.queue.with(|q| q.unshift(value.take()) );
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            self.queue.with(|q| {
                if !q.is_empty() {
                    Some(q.shift())
                } else {
                    None
                }
            })
        }
    }

    pub fn steal(&mut self) -> Option<T> {
        unsafe {
            self.queue.with(|q| {
                if !q.is_empty() {
                    Some(q.pop())
                } else {
                    None
                }
            })
        }
    }

    pub fn is_empty(&self) -> bool {
        unsafe {
            self.queue.with_imm(|q| q.is_empty() )
        }
    }
}

impl<T> Clone for WorkQueue<T> {
    fn clone(&self) -> WorkQueue<T> {
        WorkQueue {
            queue: self.queue.clone()
        }
    }
}
