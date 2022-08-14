use crate::sys::locks::Mutex;
use crate::sys_common::lazy_box::{LazyBox, LazyInit};
use crate::time::Duration;

use super::waitqueue::{SpinMutex, WaitQueue, WaitVariable};

pub struct Condvar(LazyBox<StaticCondvar>);

struct StaticCondvar(SpinMutex<WaitVariable<()>>);

impl LazyInit for StaticCondvar {
    fn init() -> Box<StaticCondvar> {
        Box::new(StaticCondvar(SpinMutex::new(WaitVariable::new(()))))
    }
}

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar(LazyBox::new())
    }

    #[inline]
    pub fn notify_one(&self) {
        let _ = WaitQueue::notify_one(self.0.0.lock());
    }

    #[inline]
    pub fn notify_all(&self) {
        let _ = WaitQueue::notify_all(self.0.0.lock());
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let guard = self.0.0.lock();
        WaitQueue::wait(guard, || unsafe { mutex.unlock() });
        mutex.lock()
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let success = WaitQueue::wait_timeout(&self.0.0, dur, || unsafe { mutex.unlock() });
        mutex.lock();
        success
    }
}
