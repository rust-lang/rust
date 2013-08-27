// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Maintains a shared list of sleeping schedulers. Schedulers
//! use this to wake each other up.

use container::Container;
use vec::OwnedVector;
use option::{Option, Some, None};
use cell::Cell;
use unstable::sync::{UnsafeArc, LittleLock};
use rt::sched::SchedHandle;
use clone::Clone;

pub struct SleeperList {
    priv state: UnsafeArc<State>
}

struct State {
    count: uint,
    stack: ~[SchedHandle],
    lock: LittleLock
}

impl SleeperList {
    pub fn new() -> SleeperList {
        SleeperList {
            state: UnsafeArc::new(State {
                count: 0,
                stack: ~[],
                lock: LittleLock::new()
            })
        }
    }

    pub fn push(&mut self, handle: SchedHandle) {
        let handle = Cell::new(handle);
        unsafe {
            let state = self.state.get();
            do (*state).lock.lock {
                (*state).count += 1;
                (*state).stack.push(handle.take());
            }
        }
    }

    pub fn pop(&mut self) -> Option<SchedHandle> {
        unsafe {
            let state = self.state.get();
            do (*state).lock.lock {
                if !(*state).stack.is_empty() {
                    (*state).count -= 1;
                    Some((*state).stack.pop())
                } else {
                    None
                }
            }
        }
    }

    /// A pop that may sometimes miss enqueued elements, but is much faster
    /// to give up without doing any synchronization
    pub fn casual_pop(&mut self) -> Option<SchedHandle> {
        unsafe {
            let state = self.state.get();
            // NB: Unsynchronized check
            if (*state).count == 0 { return None; }
            do (*state).lock.lock {
                if !(*state).stack.is_empty() {
                    // NB: count is also protected by the lock
                    (*state).count -= 1;
                    Some((*state).stack.pop())
                } else {
                    None
                }
            }
        }
    }
}

impl Clone for SleeperList {
    fn clone(&self) -> SleeperList {
        SleeperList {
            state: self.state.clone()
        }
    }
}
