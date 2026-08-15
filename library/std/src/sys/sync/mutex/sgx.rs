use crate::sys::pal::waitqueue::{SpinMutex, WaitQueue, WaitVariable, try_lock_or_false};
use crate::sys::sync::OnceBox;

pub struct Mutex {
    // FIXME: `UnsafeList` is not movable.
    inner: OnceBox<SpinMutex<WaitVariable<bool>>>,
}

// Implementation according to “Operating Systems: Three Easy Pieces”, chapter 28
impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { inner: OnceBox::new() }
    }

    fn get(&self) -> &SpinMutex<WaitVariable<bool>> {
        self.inner.get_or_init(|| Box::pin(SpinMutex::new(WaitVariable::new(false)))).get_ref()
    }

    #[inline]
    pub fn lock(&self) {
        let mut guard = self.get().lock();
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
        // SAFETY: the mutex was locked by the current thread, so it has been
        // initialized already.
        let guard = unsafe { self.inner.get_unchecked().get_ref().lock() };
        if let Err(mut guard) = WaitQueue::notify_one(guard) {
            // No other waiters, unlock
            *guard.lock_var_mut() = false;
        } else {
            // There was a thread waiting, just pass the lock
        }
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        let mut guard = try_lock_or_false!(self.get());
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
