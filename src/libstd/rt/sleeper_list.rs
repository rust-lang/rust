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
use unstable::sync::Exclusive;
use rt::sched::SchedHandle;
use clone::Clone;

pub struct SleeperList {
    priv stack: ~Exclusive<~[SchedHandle]>
}

impl SleeperList {
    pub fn new() -> SleeperList {
        SleeperList {
            stack: ~Exclusive::new(~[])
        }
    }

    pub fn push(&mut self, handle: SchedHandle) {
        let handle = Cell::new(handle);
        unsafe {
            self.stack.with(|s| s.push(handle.take()));
        }
    }

    pub fn pop(&mut self) -> Option<SchedHandle> {
        unsafe {
            do self.stack.with |s| {
                if !s.is_empty() {
                    Some(s.pop())
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
            stack: self.stack.clone()
        }
    }
}