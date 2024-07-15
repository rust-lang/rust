//! System Mutexes
//!
//! The Windows implementation of mutexes is a little odd and it might not be
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

use crate::cell::UnsafeCell;
use crate::sys::c;

pub struct Mutex {
    srwlock: UnsafeCell<c::SRWLOCK>,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

#[inline]
pub unsafe fn raw(m: &Mutex) -> *mut c::SRWLOCK {
    m.srwlock.get()
}

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { srwlock: UnsafeCell::new(c::SRWLOCK_INIT) }
    }

    #[inline]
    pub fn lock(&self) {
        unsafe {
            c::AcquireSRWLockExclusive(raw(self));
        }
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        unsafe { c::TryAcquireSRWLockExclusive(raw(self)) != 0 }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        c::ReleaseSRWLockExclusive(raw(self));
    }
}
