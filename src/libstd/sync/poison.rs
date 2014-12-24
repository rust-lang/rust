// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use thread::Thread;

pub struct Flag { pub failed: bool }

impl Flag {
    pub fn borrow(&mut self) -> Guard {
        Guard { flag: &mut self.failed, panicking: Thread::panicking() }
    }
}

pub struct Guard<'a> {
    flag: &'a mut bool,
    panicking: bool,
}

impl<'a> Guard<'a> {
    pub fn check(&self, name: &str) {
        if *self.flag {
            panic!("poisoned {} - another task failed inside", name);
        }
    }

    pub fn done(&mut self) {
        if !self.panicking && Thread::panicking() {
            *self.flag = true;
        }
    }
}
