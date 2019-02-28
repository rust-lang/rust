use crate::cell::UnsafeCell;
use crate::sys::c;
use crate::sys::mutex::{self, Mutex};
use crate::sys::os;
use crate::time::Duration;

pub struct Condvar { inner: UnsafeCell<c::CONDITION_VARIABLE> }

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
        let r = c::SleepConditionVariableSRW(self.inner.get(),
                                             mutex::raw(mutex),
                                             c::INFINITE,
                                             0);
        debug_assert!(r != 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let r = c::SleepConditionVariableSRW(self.inner.get(),
                                             mutex::raw(mutex),
                                             super::dur2timeout(dur),
                                             0);
        if r == 0 {
            debug_assert_eq!(os::errno() as usize, c::ERROR_TIMEOUT as usize);
            false
        } else {
            true
        }
    }

    #[inline]
    pub unsafe fn notify_one(&self) {
        c::WakeConditionVariable(self.inner.get())
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        c::WakeAllConditionVariable(self.inner.get())
    }

    pub unsafe fn destroy(&self) {
        // ...
    }
}
