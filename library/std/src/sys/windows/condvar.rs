#![deny(unsafe_op_in_unsafe_fn)]

use crate::cell::UnsafeCell;
use crate::sys::c;
use crate::sys::mutex::{self, Mutex};
use crate::sys::os;
use crate::time::Duration;

pub struct Condvar {
    inner: UnsafeCell<c::CONDITION_VARIABLE>,
}

pub type MovableCondvar = Condvar;

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { inner: UnsafeCell::new(c::CONDITION_VARIABLE_INIT) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        // SAFETY: The caller must ensure that the condvar is not moved or copied
        let r = unsafe {
            c::SleepConditionVariableSRW(self.inner.get(), mutex::raw(mutex), c::INFINITE, 0)
        };
        debug_assert!(r != 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        // SAFETY: The caller must ensure that the condvar is not moved or copied
        let r = unsafe {
            c::SleepConditionVariableSRW(
                self.inner.get(),
                mutex::raw(mutex),
                super::dur2timeout(dur),
                0,
            )
        };
        if r == 0 {
            debug_assert_eq!(os::errno() as usize, c::ERROR_TIMEOUT as usize);
            false
        } else {
            true
        }
    }

    #[inline]
    pub unsafe fn notify_one(&self) {
        // SAFETY: The caller must ensure that the condvar is not moved or copied
        unsafe { c::WakeConditionVariable(self.inner.get()) }
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        // SAFETY: The caller must ensure that the condvar is not moved or copied
        unsafe { c::WakeAllConditionVariable(self.inner.get()) }
    }

    pub unsafe fn destroy(&self) {
        // ...
    }
}
