use crate::fmt;
use crate::marker;
use crate::ops::Deref;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sys::mutex as sys;

/// A re-entrant mutual exclusion
///
/// This mutex will block *other* threads waiting for the lock to become
/// available. The thread which has already locked the mutex can lock it
/// multiple times without blocking, preventing a common source of deadlocks.
pub struct ReentrantMutex<T> {
    inner: sys::ReentrantMutex,
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
    // funny underscores due to how Deref currently works (it disregards field
    // privacy).
    __lock: &'a ReentrantMutex<T>,
}

impl<T> !marker::Send for ReentrantMutexGuard<'_, T> {}

impl<T> ReentrantMutex<T> {
    /// Creates a new reentrant mutex in an unlocked state.
    ///
    /// # Unsafety
    ///
    /// This function is unsafe because it is required that `init` is called
    /// once this mutex is in its final resting place, and only then are the
    /// lock/unlock methods safe.
    pub const unsafe fn new(t: T) -> ReentrantMutex<T> {
        ReentrantMutex { inner: sys::ReentrantMutex::uninitialized(), data: t }
    }

    /// Initializes this mutex so it's ready for use.
    ///
    /// # Unsafety
    ///
    /// Unsafe to call more than once, and must be called after this will no
    /// longer move in memory.
    pub unsafe fn init(&self) {
        self.inner.init();
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
        unsafe { self.inner.lock() }
        ReentrantMutexGuard::new(&self)
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
        if unsafe { self.inner.try_lock() } { Some(ReentrantMutexGuard::new(&self)) } else { None }
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

impl<T: fmt::Debug + 'static> fmt::Debug for ReentrantMutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.try_lock() {
            Some(guard) => f.debug_struct("ReentrantMutex").field("data", &*guard).finish(),
            None => {
                struct LockedPlaceholder;
                impl fmt::Debug for LockedPlaceholder {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str("<locked>")
                    }
                }

                f.debug_struct("ReentrantMutex").field("data", &LockedPlaceholder).finish()
            }
        }
    }
}

impl<'mutex, T> ReentrantMutexGuard<'mutex, T> {
    fn new(lock: &'mutex ReentrantMutex<T>) -> ReentrantMutexGuard<'mutex, T> {
        ReentrantMutexGuard { __lock: lock }
    }
}

impl<T> Deref for ReentrantMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.__lock.data
    }
}

impl<T> Drop for ReentrantMutexGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.__lock.inner.unlock();
        }
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use crate::cell::RefCell;
    use crate::sync::Arc;
    use crate::sys_common::remutex::{ReentrantMutex, ReentrantMutexGuard};
    use crate::thread;

    #[test]
    fn smoke() {
        let m = unsafe {
            let m = ReentrantMutex::new(());
            m.init();
            m
        };
        {
            let a = m.lock();
            {
                let b = m.lock();
                {
                    let c = m.lock();
                    assert_eq!(*c, ());
                }
                assert_eq!(*b, ());
            }
            assert_eq!(*a, ());
        }
    }

    #[test]
    fn is_mutex() {
        let m = unsafe {
            let m = Arc::new(ReentrantMutex::new(RefCell::new(0)));
            m.init();
            m
        };
        let m2 = m.clone();
        let lock = m.lock();
        let child = thread::spawn(move || {
            let lock = m2.lock();
            assert_eq!(*lock.borrow(), 4950);
        });
        for i in 0..100 {
            let lock = m.lock();
            *lock.borrow_mut() += i;
        }
        drop(lock);
        child.join().unwrap();
    }

    #[test]
    fn trylock_works() {
        let m = unsafe {
            let m = Arc::new(ReentrantMutex::new(()));
            m.init();
            m
        };
        let m2 = m.clone();
        let _lock = m.try_lock();
        let _lock2 = m.try_lock();
        thread::spawn(move || {
            let lock = m2.try_lock();
            assert!(lock.is_none());
        })
        .join()
        .unwrap();
        let _lock3 = m.try_lock();
    }

    pub struct Answer<'a>(pub ReentrantMutexGuard<'a, RefCell<u32>>);
    impl Drop for Answer<'_> {
        fn drop(&mut self) {
            *self.0.borrow_mut() = 42;
        }
    }
}
