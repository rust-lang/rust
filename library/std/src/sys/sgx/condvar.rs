use crate::sys::locks::Mutex;
use crate::sys_common::lazy_box::{LazyBox, LazyInit};
use crate::time::Duration;

use super::waitqueue::{SpinMutex, WaitQueue, WaitVariable};

/// FIXME: `UnsafeList` is not movable.
struct AllocatedCondvar(SpinMutex<WaitVariable<()>>);

pub struct Condvar {
    inner: LazyBox<AllocatedCondvar>,
}

impl LazyInit for AllocatedCondvar {
    fn init() -> Box<Self> {
        Box::new(AllocatedCondvar(SpinMutex::new(WaitVariable::new(()))))
    }
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { inner: LazyBox::new() }
    }

    #[inline]
    pub fn notify_one(&self) {
        let _ = WaitQueue::notify_one(self.inner.0.lock());
    }

    #[inline]
    pub fn notify_all(&self) {
        let _ = WaitQueue::notify_all(self.inner.0.lock());
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let guard = self.inner.0.lock();
        WaitQueue::wait(guard, || unsafe { mutex.unlock() });
        mutex.lock()
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let success = WaitQueue::wait_timeout(&self.inner.0, dur, || unsafe { mutex.unlock() });
        mutex.lock();
        success
    }
}
