use super::waitqueue::{try_lock_or_false, SpinMutex, WaitQueue, WaitVariable};
use crate::sys_common::lazy_box::{LazyBox, LazyInit};

/// FIXME: `UnsafeList` is not movable.
struct AllocatedMutex(SpinMutex<WaitVariable<bool>>);

pub struct Mutex {
    inner: LazyBox<AllocatedMutex>,
}

impl LazyInit for AllocatedMutex {
    fn init() -> Box<Self> {
        Box::new(AllocatedMutex(SpinMutex::new(WaitVariable::new(false))))
    }
}

// Implementation according to “Operating Systems: Three Easy Pieces”, chapter 28
impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { inner: LazyBox::new() }
    }

    #[inline]
    pub fn lock(&self) {
        let mut guard = self.inner.0.lock();
        if *guard.lock_var() {
            // Another thread has the lock, wait
            WaitQueue::wait(guard, || {})
        // Another thread has passed the lock to us
        } else {
            // We are just now obtaining the lock
            *guard.lock_var_mut() = true;
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let guard = self.inner.0.lock();
        if let Err(mut guard) = WaitQueue::notify_one(guard) {
            // No other waiters, unlock
            *guard.lock_var_mut() = false;
        } else {
            // There was a thread waiting, just pass the lock
        }
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        let mut guard = try_lock_or_false!(self.inner.0);
        if *guard.lock_var() {
            // Another thread has the lock
            false
        } else {
            // We are just now obtaining the lock
            *guard.lock_var_mut() = true;
            true
        }
    }
}
