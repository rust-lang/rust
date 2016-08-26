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
//!    is that there are no guarantees of fairness.
//!
//! The downside of this approach, however, is that SRWLock is not available on
//! Windows XP, so we continue to have a fallback implementation where
//! CriticalSection is used and we keep track of who's holding the mutex to
//! detect recursive locks.

use cell::UnsafeCell;
use mem;
use sync::atomic::{AtomicUsize, Ordering};
use sys::c;
use sys::compat;

pub struct Mutex {
    lock: AtomicUsize,
    held: UnsafeCell<bool>,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

#[derive(Clone, Copy)]
enum Kind {
    SRWLock = 1,
    CriticalSection = 2,
}

#[inline]
pub unsafe fn raw(m: &Mutex) -> c::PSRWLOCK {
    debug_assert!(mem::size_of::<c::SRWLOCK>() <= mem::size_of_val(&m.lock));
    &m.lock as *const _ as *mut _
}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex {
            lock: AtomicUsize::new(0),
            held: UnsafeCell::new(false),
        }
    }
    #[inline]
    pub unsafe fn init(&mut self) {}
    pub unsafe fn lock(&self) {
        match kind() {
            Kind::SRWLock => c::AcquireSRWLockExclusive(raw(self)),
            Kind::CriticalSection => {
                let re = self.remutex();
                (*re).lock();
                if !self.flag_locked() {
                    (*re).unlock();
                    panic!("cannot recursively lock a mutex");
                }
            }
        }
    }
    pub unsafe fn try_lock(&self) -> bool {
        match kind() {
            Kind::SRWLock => c::TryAcquireSRWLockExclusive(raw(self)) != 0,
            Kind::CriticalSection => {
                let re = self.remutex();
                if !(*re).try_lock() {
                    false
                } else if self.flag_locked() {
                    true
                } else {
                    (*re).unlock();
                    false
                }
            }
        }
    }
    pub unsafe fn unlock(&self) {
        *self.held.get() = false;
        match kind() {
            Kind::SRWLock => c::ReleaseSRWLockExclusive(raw(self)),
            Kind::CriticalSection => (*self.remutex()).unlock(),
        }
    }
    pub unsafe fn destroy(&self) {
        match kind() {
            Kind::SRWLock => {}
            Kind::CriticalSection => {
                match self.lock.load(Ordering::SeqCst) {
                    0 => {}
                    n => { Box::from_raw(n as *mut ReentrantMutex).destroy(); }
                }
            }
        }
    }

    unsafe fn remutex(&self) -> *mut ReentrantMutex {
        match self.lock.load(Ordering::SeqCst) {
            0 => {}
            n => return n as *mut _,
        }
        let mut re = Box::new(ReentrantMutex::uninitialized());
        re.init();
        let re = Box::into_raw(re);
        match self.lock.compare_and_swap(0, re as usize, Ordering::SeqCst) {
            0 => re,
            n => { Box::from_raw(re).destroy(); n as *mut _ }
        }
    }

    unsafe fn flag_locked(&self) -> bool {
        if *self.held.get() {
            false
        } else {
            *self.held.get() = true;
            true
        }

    }
}

fn kind() -> Kind {
    static KIND: AtomicUsize = AtomicUsize::new(0);

    let val = KIND.load(Ordering::SeqCst);
    if val == Kind::SRWLock as usize {
        return Kind::SRWLock
    } else if val == Kind::CriticalSection as usize {
        return Kind::CriticalSection
    }

    let ret = match compat::lookup("kernel32", "AcquireSRWLockExclusive") {
        None => Kind::CriticalSection,
        Some(..) => Kind::SRWLock,
    };
    KIND.store(ret as usize, Ordering::SeqCst);
    return ret;
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
