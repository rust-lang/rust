#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

use crate::marker::PhantomPinned;
use crate::ops::Deref;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::pin::Pin;
use crate::sys::mutex as sys;

/// A re-entrant mutual exclusion
///
/// This mutex will block *other* threads waiting for the lock to become
/// available. The thread which has already locked the mutex can lock it
/// multiple times without blocking, preventing a common source of deadlocks.
pub struct ReentrantMutex<T> {
    inner: sys::ReentrantMutex,
    data: T,
    _pinned: PhantomPinned,
}

unsafe impl<T: Send> Send for ReentrantMutex<T> {}
unsafe impl<T: Send> Sync for ReentrantMutex<T> {}

impl<T> UnwindSafe for ReentrantMutex<T> {}
impl<T> RefUnwindSafe for ReentrantMutex<T> {}

/// An RAII implementation of a "scoped lock" of a mutex. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the mutex can be accessed through this guard via its
/// Deref implementation.
///
/// # Mutability
///
/// Unlike `MutexGuard`, `ReentrantMutexGuard` does not implement `DerefMut`,
/// because implementation of the trait would violate Rust’s reference aliasing
/// rules. Use interior mutability (usually `RefCell`) in order to mutate the
/// guarded data.
#[must_use = "if unused the ReentrantMutex will immediately unlock"]
pub struct ReentrantMutexGuard<'a, T: 'a> {
    lock: Pin<&'a ReentrantMutex<T>>,
}

impl<T> !Send for ReentrantMutexGuard<'_, T> {}

impl<T> ReentrantMutex<T> {
    /// Creates a new reentrant mutex in an unlocked state.
    ///
    /// # Unsafety
    ///
    /// This function is unsafe because it is required that `init` is called
    /// once this mutex is in its final resting place, and only then are the
    /// lock/unlock methods safe.
    pub const unsafe fn new(t: T) -> ReentrantMutex<T> {
        ReentrantMutex {
            inner: sys::ReentrantMutex::uninitialized(),
            data: t,
            _pinned: PhantomPinned,
        }
    }

    /// Initializes this mutex so it's ready for use.
    ///
    /// # Unsafety
    ///
    /// Unsafe to call more than once, and must be called after this will no
    /// longer move in memory.
    pub unsafe fn init(self: Pin<&mut Self>) {
        self.get_unchecked_mut().inner.init()
    }

    /// Acquires a mutex, blocking the current thread until it is able to do so.
    ///
    /// This function will block the caller until it is available to acquire the mutex.
    /// Upon returning, the thread is the only thread with the mutex held. When the thread
    /// calling this method already holds the lock, the call shall succeed without
    /// blocking.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return failure if the mutex would otherwise be
    /// acquired.
    pub fn lock(self: Pin<&Self>) -> ReentrantMutexGuard<'_, T> {
        unsafe { self.inner.lock() }
        ReentrantMutexGuard { lock: self }
    }

    /// Attempts to acquire this lock.
    ///
    /// If the lock could not be acquired at this time, then `Err` is returned.
    /// Otherwise, an RAII guard is returned.
    ///
    /// This function does not block.
    ///
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return failure if the mutex would otherwise be
    /// acquired.
    pub fn try_lock(self: Pin<&Self>) -> Option<ReentrantMutexGuard<'_, T>> {
        if unsafe { self.inner.try_lock() } {
            Some(ReentrantMutexGuard { lock: self })
        } else {
            None
        }
    }
}

impl<T> Drop for ReentrantMutex<T> {
    fn drop(&mut self) {
        // This is actually safe b/c we know that there is no further usage of
        // this mutex (it's up to the user to arrange for a mutex to get
        // dropped, that's not our job)
        unsafe { self.inner.destroy() }
    }
}

impl<T> Deref for ReentrantMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.lock.data
    }
}

impl<T> Drop for ReentrantMutexGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.lock.inner.unlock();
        }
    }
}
