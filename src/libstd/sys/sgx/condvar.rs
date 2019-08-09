use crate::sys::mutex::Mutex;
use crate::time::Duration;

use super::waitqueue::{WaitVariable, WaitQueue, SpinMutex};

pub struct Condvar {
    inner: SpinMutex<WaitVariable<()>>,
}

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
        mutex.unlock();
        WaitQueue::wait(guard);
        mutex.lock()
    }

    pub unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool {
        rtabort!("timeout not supported in SGX");
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}
