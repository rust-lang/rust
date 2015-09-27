// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! System Mutexes
//!
//! The Windows implementation of mutexes is a little odd and it may not be
//! immediately obvious what's going on. The primary oddness is that SRWLock is
//! used instead of CriticalSection, and this is done because:
//!
//! 1. SRWLock is several times faster than CriticalSection according to
//!    benchmarks performed on both Windows 8 and Windows 7.
//!
//! 2. CriticalSection allows recursive locking while SRWLock deadlocks. The
//!    Unix implementation deadlocks so consistency is preferred. See #19962 for
//!    more details.
//!
//! 3. While CriticalSection is fair and SRWLock is not, the current Rust policy
//!    is there there are no guarantees of fairness.

use prelude::v1::*;

use cell::UnsafeCell;
use mem;
use sys::c;

pub struct Mutex { inner: UnsafeCell<c::SRWLOCK> }

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

#[inline]
pub unsafe fn raw(m: &Mutex) -> c::PSRWLOCK {
    m.inner.get()
}

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { inner: UnsafeCell::new(c::SRWLOCK_INIT) }
    }
    #[inline]
    pub unsafe fn lock(&self) {
        c::AcquireSRWLockExclusive(self.inner.get())
    }
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        c::TryAcquireSRWLockExclusive(self.inner.get()) != 0
    }
    #[inline]
    pub unsafe fn unlock(&self) {
        c::ReleaseSRWLockExclusive(self.inner.get())
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        // ...
    }
}

pub struct ReentrantMutex { inner: UnsafeCell<c::CRITICAL_SECTION> }

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub unsafe fn uninitialized() -> ReentrantMutex {
        mem::uninitialized()
    }

    pub unsafe fn init(&mut self) {
        c::InitializeCriticalSection(self.inner.get());
    }

    pub unsafe fn lock(&self) {
        c::EnterCriticalSection(self.inner.get());
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        c::TryEnterCriticalSection(self.inner.get()) != 0
    }

    pub unsafe fn unlock(&self) {
        c::LeaveCriticalSection(self.inner.get());
    }

    pub unsafe fn destroy(&self) {
        c::DeleteCriticalSection(self.inner.get());
    }
}
