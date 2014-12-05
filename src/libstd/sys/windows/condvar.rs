// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::UnsafeCell;
use libc::{mod, DWORD};
use libc;
use os;
use sys::mutex::{mod, Mutex};
use sys::sync as ffi;
use time::Duration;

pub struct Condvar { inner: UnsafeCell<ffi::CONDITION_VARIABLE> }

pub const CONDVAR_INIT: Condvar = Condvar {
    inner: UnsafeCell { value: ffi::CONDITION_VARIABLE_INIT }
};

impl Condvar {
    #[inline]
    pub unsafe fn new() -> Condvar { CONDVAR_INIT }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        let r = ffi::SleepConditionVariableCS(self.inner.get(),
                                              mutex::raw(mutex),
                                              libc::INFINITE);
        debug_assert!(r != 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let r = ffi::SleepConditionVariableCS(self.inner.get(),
                                              mutex::raw(mutex),
                                              dur.num_milliseconds() as DWORD);
        if r == 0 {
            const ERROR_TIMEOUT: DWORD = 0x5B4;
            debug_assert_eq!(os::errno() as uint, ERROR_TIMEOUT as uint);
            false
        } else {
            true
        }
    }

    #[inline]
    pub unsafe fn notify_one(&self) {
        ffi::WakeConditionVariable(self.inner.get())
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        ffi::WakeAllConditionVariable(self.inner.get())
    }

    pub unsafe fn destroy(&self) {
        // ...
    }
}
