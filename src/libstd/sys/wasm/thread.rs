// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use boxed::FnBox;
use ffi::CStr;
use io;
use sys::{unsupported, Void};
use time::Duration;

pub struct Thread(Void);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

impl Thread {
    pub unsafe fn new<'a>(_stack: usize, _p: Box<dyn FnBox() + 'a>)
        -> io::Result<Thread>
    {
        unsupported()
    }

    pub fn yield_now() {
        // do nothing
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    #[cfg(not(target_feature = "atomics"))]
    pub fn sleep(_dur: Duration) {
        panic!("can't sleep");
    }

    #[cfg(target_feature = "atomics")]
    pub fn sleep(dur: Duration) {
        use arch::wasm32::atomic;
        use cmp;

        // Use an atomic wait to block the current thread artificially with a
        // timeout listed. Note that we should never be notified (return value
        // of 0) or our comparison should never fail (return value of 1) so we
        // should always only resume execution through a timeout (return value
        // 2).
        let mut nanos = dur.as_nanos();
        while nanos > 0 {
            let amt = cmp::min(i64::max_value() as u128, nanos);
            let mut x = 0;
            let val = unsafe { atomic::wait_i32(&mut x, 0, amt as i64) };
            debug_assert_eq!(val, 2);
            nanos -= amt;
        }
    }

    pub fn join(self) {
        match self.0 {}
    }
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
    pub unsafe fn deinit() {}
}
