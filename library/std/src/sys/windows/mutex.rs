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

use crate::cell::{Cell, UnsafeCell};
use crate::mem::{self, MaybeUninit};
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sys::c;

pub struct Mutex {
    // This is either directly an SRWLOCK (if supported), or a Box<Inner> otherwise.
    lock: AtomicUsize,
}

// Windows SRW Locks are movable (while not borrowed).
// ReentrantMutexes (in Inner) are not, but those are stored indirectly through
// a Box, so do not move when the Mutex it self is moved.
pub type MovableMutex = Mutex;

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

struct Inner {
    remutex: ReentrantMutex,
    held: Cell<bool>,
}

#[derive(Clone, Copy)]
enum Kind {
    SRWLock,
    CriticalSection,
}

#[inline]
pub unsafe fn raw(m: &Mutex) -> c::PSRWLOCK {
    debug_assert!(mem::size_of::<c::SRWLOCK>() <= mem::size_of_val(&m.lock));
    &m.lock as *const _ as *mut _
}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex {
            // This works because SRWLOCK_INIT is 0 (wrapped in a struct), so we are also properly
            // initializing an SRWLOCK here.
            lock: AtomicUsize::new(0),
        }
    }
    #[inline]
    pub unsafe fn init(&mut self) {}
    pub unsafe fn lock(&self) {
        match kind() {
            Kind::SRWLock => c::AcquireSRWLockExclusive(raw(self)),
            Kind::CriticalSection => {
                let inner = &*self.inner();
                inner.remutex.lock();
                if inner.held.replace(true) {
                    // It was already locked, so we got a recursive lock which we do not want.
                    inner.remutex.unlock();
                    panic!("cannot recursively lock a mutex");
                }
            }
        }
    }
    pub unsafe fn try_lock(&self) -> bool {
        match kind() {
            Kind::SRWLock => c::TryAcquireSRWLockExclusive(raw(self)) != 0,
            Kind::CriticalSection => {
                let inner = &*self.inner();
                if !inner.remutex.try_lock() {
                    false
                } else if inner.held.replace(true) {
                    // It was already locked, so we got a recursive lock which we do not want.
                    inner.remutex.unlock();
                    false
                } else {
                    true
                }
            }
        }
    }
    pub unsafe fn unlock(&self) {
        match kind() {
            Kind::SRWLock => c::ReleaseSRWLockExclusive(raw(self)),
            Kind::CriticalSection => {
                let inner = &*(self.lock.load(Ordering::SeqCst) as *const Inner);
                inner.held.set(false);
                inner.remutex.unlock();
            }
        }
    }
    pub unsafe fn destroy(&self) {
        match kind() {
            Kind::SRWLock => {}
            Kind::CriticalSection => match self.lock.load(Ordering::SeqCst) {
                0 => {}
                n => Box::from_raw(n as *mut Inner).remutex.destroy(),
            },
        }
    }

    unsafe fn inner(&self) -> *const Inner {
        match self.lock.load(Ordering::SeqCst) {
            0 => {}
            n => return n as *const _,
        }
        let inner = box Inner { remutex: ReentrantMutex::uninitialized(), held: Cell::new(false) };
        inner.remutex.init();
        let inner = Box::into_raw(inner);
        match self.lock.compare_exchange(0, inner as usize, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => inner,
            Err(n) => {
                Box::from_raw(inner).remutex.destroy();
                n as *const _
            }
        }
    }
}

fn kind() -> Kind {
    if c::AcquireSRWLockExclusive::is_available() { Kind::SRWLock } else { Kind::CriticalSection }
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
