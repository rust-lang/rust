// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use marker::Sync;
use cell::UnsafeCell;
use sys::sync as ffi;

pub struct Mutex { inner: UnsafeCell<ffi::SRWLOCK> }

pub const MUTEX_INIT: Mutex = Mutex {
    inner: UnsafeCell { value: ffi::SRWLOCK_INIT }
};

unsafe impl Sync for Mutex {}

#[inline]
pub unsafe fn raw(m: &Mutex) -> ffi::PSRWLOCK {
    m.inner.get()
}

// So you might be asking why we're using SRWLock instead of CriticalSection?
//
// 1. SRWLock is several times faster than CriticalSection according to benchmarks performed on both
// Windows 8 and Windows 7.
//
// 2. CriticalSection allows recursive locking while SRWLock deadlocks. The Unix implementation
// deadlocks so consistency is preferred. See #19962 for more details.
//
// 3. While CriticalSection is fair and SRWLock is not, the current Rust policy is there there are
// no guarantees of fairness.

impl Mutex {
    #[inline]
    pub unsafe fn lock(&self) {
        ffi::AcquireSRWLockExclusive(self.inner.get())
    }
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        ffi::TryAcquireSRWLockExclusive(self.inner.get()) != 0
    }
    #[inline]
    pub unsafe fn unlock(&self) {
        ffi::ReleaseSRWLockExclusive(self.inner.get())
    }
    #[inline]
    pub unsafe fn destroy(&self) {
        // ...
    }
}
