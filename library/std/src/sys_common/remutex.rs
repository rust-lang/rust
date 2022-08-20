#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

use super::mutex as sys;
use crate::cell::UnsafeCell;
use crate::ops::Deref;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// A re-entrant mutual exclusion
///
/// This mutex will block *other* threads waiting for the lock to become
/// available. The thread which has already locked the mutex can lock it
/// multiple times without blocking, preventing a common source of deadlocks.
///
/// This is used by stdout().lock() and friends.
///
/// ## Implementation details
///
/// The 'owner' field tracks which thread has locked the mutex.
///
/// We use current_thread_unique_ptr() as the thread identifier,
/// which is just the address of a thread local variable.
///
/// If `owner` is set to the identifier of the current thread,
/// we assume the mutex is already locked and instead of locking it again,
/// we increment `lock_count`.
///
/// When unlocking, we decrement `lock_count`, and only unlock the mutex when
/// it reaches zero.
///
/// `lock_count` is protected by the mutex and only accessed by the thread that has
/// locked the mutex, so needs no synchronization.
///
/// `owner` can be checked by other threads that want to see if they already
/// hold the lock, so needs to be atomic. If it compares equal, we're on the
/// same thread that holds the mutex and memory access can use relaxed ordering
/// since we're not dealing with multiple threads. If it compares unequal,
/// synchronization is left to the mutex, making relaxed memory ordering for
/// the `owner` field fine in all cases.
pub struct ReentrantMutex<T> {
    mutex: sys::MovableMutex,
    owner: AtomicUsize,
    lock_count: UnsafeCell<u32>,
    data: T,
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
    lock: &'a ReentrantMutex<T>,
}

impl<T> !Send for ReentrantMutexGuard<'_, T> {}

impl<T> ReentrantMutex<T> {
    /// Creates a new reentrant mutex in an unlocked state.
    pub const fn new(t: T) -> ReentrantMutex<T> {
        ReentrantMutex {
            mutex: sys::MovableMutex::new(),
            owner: AtomicUsize::new(0),
            lock_count: UnsafeCell::new(0),
            data: t,
        }
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
    pub fn lock(&self) -> ReentrantMutexGuard<'_, T> {
        let this_thread = current_thread_unique_ptr();
        // Safety: We only touch lock_count when we own the lock.
        unsafe {
            if self.owner.load(Relaxed) == this_thread {
                self.increment_lock_count();
            } else {
                self.mutex.raw_lock();
                self.owner.store(this_thread, Relaxed);
                debug_assert_eq!(*self.lock_count.get(), 0);
                *self.lock_count.get() = 1;
            }
        }
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
    pub fn try_lock(&self) -> Option<ReentrantMutexGuard<'_, T>> {
        let this_thread = current_thread_unique_ptr();
        // Safety: We only touch lock_count when we own the lock.
        unsafe {
            if self.owner.load(Relaxed) == this_thread {
                self.increment_lock_count();
                Some(ReentrantMutexGuard { lock: self })
            } else if self.mutex.try_lock() {
                self.owner.store(this_thread, Relaxed);
                debug_assert_eq!(*self.lock_count.get(), 0);
                *self.lock_count.get() = 1;
                Some(ReentrantMutexGuard { lock: self })
            } else {
                None
            }
        }
    }

    unsafe fn increment_lock_count(&self) {
        *self.lock_count.get() = (*self.lock_count.get())
            .checked_add(1)
            .expect("lock count overflow in reentrant mutex");
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
        // Safety: We own the lock.
        unsafe {
            *self.lock.lock_count.get() -= 1;
            if *self.lock.lock_count.get() == 0 {
                self.lock.owner.store(0, Relaxed);
                self.lock.mutex.raw_unlock();
            }
        }
    }
}

/// Get an address that is unique per running thread.
///
/// This can be used as a non-null usize-sized ID.
pub fn current_thread_unique_ptr() -> usize {
    // Use a non-drop type to make sure it's still available during thread destruction.
    thread_local! { static X: u8 = const { 0 } }
    X.with(|x| <*const _>::addr(x))
}
