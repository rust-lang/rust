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
use unstable::sync::{Exclusive, exclusive};
use cell::Cell;
use kinds::Owned;
use clone::Clone;

pub struct WorkQueue<T> {
    // XXX: Another mystery bug fixed by boxing this lock
    priv queue: ~Exclusive<~[T]>
}

pub impl<T: Owned> WorkQueue<T> {
    fn new() -> WorkQueue<T> {
        WorkQueue {
            queue: ~exclusive(~[])
        }
    }

    fn push(&mut self, value: T) {
        unsafe {
            let value = Cell(value);
            self.queue.with(|q| q.unshift(value.take()) );
        }
    }

    fn pop(&mut self) -> Option<T> {
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

    fn steal(&mut self) -> Option<T> {
        unsafe {
            do self.queue.with |q| {
                if !q.is_empty() {
                    Some(q.pop())
                } else {
                    None
                }
            }
        }
    }

    fn is_empty(&self) -> bool {
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
