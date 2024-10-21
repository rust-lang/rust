use crate::sys::pal::waitqueue::{SpinMutex, WaitQueue, WaitVariable};
use crate::sys::sync::{Mutex, OnceBox};
use crate::time::Duration;

pub struct Condvar {
    // FIXME: `UnsafeList` is not movable.
    inner: OnceBox<SpinMutex<WaitVariable<()>>>,
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { inner: OnceBox::new() }
    }

    fn get(&self) -> &SpinMutex<WaitVariable<()>> {
        self.inner.get_or_init(|| Box::new(SpinMutex::new(WaitVariable::new(()))))
    }

    #[inline]
    pub fn notify_one(&self) {
        let _ = WaitQueue::notify_one(self.get().lock());
    }

    #[inline]
    pub fn notify_all(&self) {
        let _ = WaitQueue::notify_all(self.get().lock());
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let guard = self.get().lock();
        WaitQueue::wait(guard, || unsafe { mutex.unlock() });
        mutex.lock()
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let success = WaitQueue::wait_timeout(self.get(), dur, || unsafe { mutex.unlock() });
        mutex.lock();
        success
    }
}
