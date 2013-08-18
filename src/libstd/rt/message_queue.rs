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

use container::Container;
use kinds::Send;
use vec::OwnedVector;
use cell::Cell;
use option::*;
use unstable::sync::{UnsafeAtomicRcBox, LittleLock};
use clone::Clone;

pub struct MessageQueue<T> {
    priv state: UnsafeAtomicRcBox<State<T>>
}

struct State<T> {
    count: uint,
    queue: ~[T],
    lock: LittleLock
}

impl<T: Send> MessageQueue<T> {
    pub fn new() -> MessageQueue<T> {
        MessageQueue {
            state: UnsafeAtomicRcBox::new(State {
                count: 0,
                queue: ~[],
                lock: LittleLock::new()
            })
        }
    }

    pub fn push(&mut self, value: T) {
        unsafe {
            let value = Cell::new(value);
            let state = self.state.get();
            do (*state).lock.lock {
                (*state).count += 1;
                (*state).queue.push(value.take());
            }
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        unsafe {
            let state = self.state.get();
            do (*state).lock.lock {
                if !(*state).queue.is_empty() {
                    (*state).count += 1;
                    Some((*state).queue.shift())
                } else {
                    None
                }
            }
        }
    }

    /// A pop that may sometimes miss enqueued elements, but is much faster
    /// to give up without doing any synchronization
    pub fn casual_pop(&mut self) -> Option<T> {
        unsafe {
            let state = self.state.get();
            // NB: Unsynchronized check
            if (*state).count == 0 { return None; }
            do (*state).lock.lock {
                if !(*state).queue.is_empty() {
                    (*state).count += 1;
                    Some((*state).queue.shift())
                } else {
                    None
                }
            }
        }
    }
}

impl<T: Send> Clone for MessageQueue<T> {
    fn clone(&self) -> MessageQueue<T> {
        MessageQueue {
            state: self.state.clone()
        }
    }
}
