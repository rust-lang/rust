// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::mutex::Mutex;
use time::Duration;

pub struct Condvar { }

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn notify_one(&self) {
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
    }

    pub unsafe fn wait(&self, _mutex: &Mutex) {
        panic!("can't block with web assembly")
    }

    pub unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool {
        panic!("can't block with web assembly");
    }

    #[inline]
    pub unsafe fn destroy(&self) {
    }
}
