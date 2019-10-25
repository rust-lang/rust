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

use crate::cell::UnsafeCell;
use crate::mem::{self, MaybeUninit};
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sys::c;
use crate::sys::compat;

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
            // This works because SRWLOCK_INIT is 0 (wrapped in a struct), so we are also properly
            // initializing an SRWLOCK here.
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
        let mut re = box ReentrantMutex::uninitialized();
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

pub struct ReentrantMutex { inner: UnsafeCell<MaybeUninit<c::CRITICAL_SECTION>> }

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { inner: UnsafeCell::new(MaybeUninit::uninit()) }
    }

    pub unsafe fn init(&mut self) {
        c::InitializeCriticalSection((&mut *self.inner.get()).as_mut_ptr());
    }

    pub unsafe fn lock(&self) {
        c::EnterCriticalSection((&mut *self.inner.get()).as_mut_ptr());
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        c::TryEnterCriticalSection((&mut *self.inner.get()).as_mut_ptr()) != 0
    }

    pub unsafe fn unlock(&self) {
        c::LeaveCriticalSection((&mut *self.inner.get()).as_mut_ptr());
    }

    pub unsafe fn destroy(&self) {
        c::DeleteCriticalSection((&mut *self.inner.get()).as_mut_ptr());
    }
}
