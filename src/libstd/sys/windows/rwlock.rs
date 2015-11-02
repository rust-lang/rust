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
use sys::windows::c;

pub struct RwLock { inner: UnsafeCell<c::SRWLOCK> }

unsafe impl Send for RwLock {}
unsafe impl Sync for RwLock {}

impl RwLock {
    pub const fn new() -> RwLock {
        RwLock { inner: UnsafeCell::new(c::SRWLOCK_INIT) }
    }
    #[inline]
    pub unsafe fn read(&self) {
        c::AcquireSRWLockShared(self.inner.get())
    }
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        c::TryAcquireSRWLockShared(self.inner.get()) != 0
    }
    #[inline]
    pub unsafe fn write(&self) {
        c::AcquireSRWLockExclusive(self.inner.get())
    }
    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        c::TryAcquireSRWLockExclusive(self.inner.get()) != 0
    }
    #[inline]
    pub unsafe fn read_unlock(&self) {
        c::ReleaseSRWLockShared(self.inner.get())
    }
    #[inline]
    pub unsafe fn write_unlock(&self) {
        c::ReleaseSRWLockExclusive(self.inner.get())
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        // ...
    }
}
