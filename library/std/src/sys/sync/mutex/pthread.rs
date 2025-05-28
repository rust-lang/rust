#![forbid(unsafe_op_in_unsafe_fn)]

use crate::mem::forget;
use crate::pin::Pin;
use crate::sys::pal::sync as pal;
use crate::sys::sync::OnceBox;

pub struct Mutex {
    pub(in crate::sys::sync) pal: OnceBox<pal::Mutex>,
}

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { pal: OnceBox::new() }
    }

    #[inline]
    fn get(&self) -> Pin<&pal::Mutex> {
        // If the initialization race is lost, the new mutex is destroyed.
        // This is sound however, as it cannot have been locked.
        self.pal.get_or_init(|| {
            let mut pal = Box::pin(pal::Mutex::new());
            // SAFETY: we only call `init` once per `pal::Mutex`, namely here.
            unsafe { pal.as_mut().init() };
            pal
        })
    }

    #[inline]
    // Make this a diagnostic item for Miri's concurrency model checker.
    #[cfg_attr(not(test), rustc_diagnostic_item = "sys_mutex_lock")]
    pub fn lock(&self) {
        // SAFETY: we call `init` above, therefore reentrant locking is safe.
        // In `drop` we ensure that the mutex is not destroyed while locked.
        unsafe { self.get().lock() }
    }

    #[inline]
    // Make this a diagnostic item for Miri's concurrency model checker.
    #[cfg_attr(not(test), rustc_diagnostic_item = "sys_mutex_unlock")]
    pub unsafe fn unlock(&self) {
        // SAFETY: the mutex can only be locked if it is already initialized
        // and we observed this initialization since we observed the locking.
        unsafe { self.pal.get_unchecked().unlock() }
    }

    #[inline]
    // Make this a diagnostic item for Miri's concurrency model checker.
    #[cfg_attr(not(test), rustc_diagnostic_item = "sys_mutex_try_lock")]
    pub fn try_lock(&self) -> bool {
        // SAFETY: we call `init` above, therefore reentrant locking is safe.
        // In `drop` we ensure that the mutex is not destroyed while locked.
        unsafe { self.get().try_lock() }
    }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        let Some(pal) = self.pal.take() else { return };
        // We're not allowed to pthread_mutex_destroy a locked mutex,
        // so check first if it's unlocked.
        if unsafe { pal.as_ref().try_lock() } {
            unsafe { pal.as_ref().unlock() };
            drop(pal)
        } else {
            // The mutex is locked. This happens if a MutexGuard is leaked.
            // In this case, we just leak the Mutex too.
            forget(pal)
        }
    }
}
