use crate::sys::locks::Mutex;
use crate::time::Duration;

use super::waitqueue::{SpinMutex, WaitQueue, WaitVariable};

pub struct Condvar {
    inner: SpinMutex<WaitVariable<()>>,
}

pub type MovableCondvar = Box<Condvar>;

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { inner: SpinMutex::new(WaitVariable::new(())) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn notify_one(&self) {
        let _ = WaitQueue::notify_one(self.inner.lock());
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        let _ = WaitQueue::notify_all(self.inner.lock());
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let guard = self.inner.lock();
        WaitQueue::wait(guard, || unsafe { mutex.unlock() });
        unsafe { mutex.lock() }
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let success = WaitQueue::wait_timeout(&self.inner, dur, || unsafe { mutex.unlock() });
        unsafe { mutex.lock() };
        success
    }
}
