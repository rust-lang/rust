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

use std::sync::mpmc_bounded_queue::Queue;

use sched::SchedHandle;

pub struct SleeperList {
    q: Queue<SchedHandle>,
}

impl SleeperList {
    pub fn new() -> SleeperList {
        SleeperList{q: Queue::with_capacity(8*1024)}
    }

    pub fn push(&mut self, value: SchedHandle) {
        assert!(self.q.push(value))
    }

    pub fn pop(&mut self) -> Option<SchedHandle> {
        self.q.pop()
    }

    pub fn casual_pop(&mut self) -> Option<SchedHandle> {
        self.q.pop()
    }
}

impl Clone for SleeperList {
    fn clone(&self) -> SleeperList {
        SleeperList {
            q: self.q.clone()
        }
    }
}
