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
use crate::mem::MaybeUninit;
use crate::sys::c;

pub struct Mutex {
    srwlock: UnsafeCell<c::SRWLOCK>,
}

// Windows SRW Locks are movable (while not borrowed).
pub type MovableMutex = Mutex;

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

#[inline]
pub unsafe fn raw(m: &Mutex) -> c::PSRWLOCK {
    m.srwlock.get()
}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { srwlock: UnsafeCell::new(c::SRWLOCK_INIT) }
    }
    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn lock(&self) {
        c::AcquireSRWLockExclusive(raw(self));
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        c::TryAcquireSRWLockExclusive(raw(self)) != 0
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        c::ReleaseSRWLockExclusive(raw(self));
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        // SRWLock does not need to be destroyed.
    }
}

pub struct ReentrantMutex {
    inner: MaybeUninit<UnsafeCell<c::CRITICAL_SECTION>>,
}

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub const fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { inner: MaybeUninit::uninit() }
    }

    pub unsafe fn init(&self) {
        c::InitializeCriticalSection(UnsafeCell::raw_get(self.inner.as_ptr()));
    }

    pub unsafe fn lock(&self) {
        c::EnterCriticalSection(UnsafeCell::raw_get(self.inner.as_ptr()));
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        c::TryEnterCriticalSection(UnsafeCell::raw_get(self.inner.as_ptr())) != 0
    }

    pub unsafe fn unlock(&self) {
        c::LeaveCriticalSection(UnsafeCell::raw_get(self.inner.as_ptr()));
    }

    pub unsafe fn destroy(&self) {
        c::DeleteCriticalSection(UnsafeCell::raw_get(self.inner.as_ptr()));
    }
}
