use crate::cell::UnsafeCell;
use crate::sys::sync::{Mutex, mutex};
use crate::sys::{c, os};
use crate::time::Duration;

pub struct Condvar {
    inner: UnsafeCell<c::CONDITION_VARIABLE>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar { inner: UnsafeCell::new(c::CONDITION_VARIABLE_INIT) }
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        let r = c::SleepConditionVariableSRW(self.inner.get(), mutex::raw(mutex), c::INFINITE, 0);
        debug_assert!(r != 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let r = c::SleepConditionVariableSRW(
            self.inner.get(),
            mutex::raw(mutex),
            crate::sys::pal::dur2timeout(dur),
            0,
        );
        if r == 0 {
            debug_assert_eq!(os::errno() as usize, c::ERROR_TIMEOUT as usize);
            false
        } else {
            true
        }
    }

    #[inline]
    pub fn notify_one(&self) {
        unsafe { c::WakeConditionVariable(self.inner.get()) }
    }

    #[inline]
    pub fn notify_all(&self) {
        unsafe { c::WakeAllConditionVariable(self.inner.get()) }
    }
}
