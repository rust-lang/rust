use crate::sys::mutex as imp;

/// An OS-based mutual exclusion lock, meant for use in static variables.
///
/// This mutex has a const constructor ([`StaticMutex::new`]), does not
/// implement `Drop` to cleanup resources, and causes UB when moved or used
/// reentrantly.
///
/// This mutex does not implement poisoning.
///
/// This is a wrapper around `imp::Mutex` that does *not* call `init()` and
/// `destroy()`.
pub struct StaticMutex(imp::Mutex);

unsafe impl Sync for StaticMutex {}

impl StaticMutex {
    /// Creates a new mutex for use.
    ///
    /// Behavior is undefined if the mutex is moved after it is
    /// first used with any of the functions below.
    /// Also, the behavior is undefined if this mutex is ever used reentrantly,
    /// i.e., `lock` is called by the thread currently holding the lock.
    #[rustc_const_stable(feature = "const_sys_mutex_new", since = "1.0.0")]
    pub const fn new() -> Self {
        Self(imp::Mutex::new())
    }

    /// Calls raw_lock() and then returns an RAII guard to guarantee the mutex
    /// will be unlocked.
    ///
    /// It is undefined behaviour to call this function while locked, or if the
    /// mutex has been moved since the last time this was called.
    #[inline]
    pub unsafe fn lock(&self) -> StaticMutexGuard<'_> {
        self.0.lock();
        StaticMutexGuard(&self.0)
    }
}

#[must_use]
pub struct StaticMutexGuard<'a>(&'a imp::Mutex);

impl Drop for StaticMutexGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.0.unlock();
        }
    }
}

/// An OS-based mutual exclusion lock.
///
/// This mutex does *not* have a const constructor, cleans up its resources in
/// its `Drop` implementation, may safely be moved (when not borrowed), and
/// does not cause UB when used reentrantly.
///
/// This mutex does not implement poisoning.
///
/// This is a wrapper around `Box<imp::Mutex>`, to allow the object to be moved
/// without moving the raw mutex.
pub struct MovableMutex(Box<imp::Mutex>);

unsafe impl Sync for MovableMutex {}

impl MovableMutex {
    /// Creates a new mutex.
    pub fn new() -> Self {
        let mut mutex = box imp::Mutex::new();
        unsafe { mutex.init() };
        Self(mutex)
    }

    pub(crate) fn raw(&self) -> &imp::Mutex {
        &self.0
    }

    /// Locks the mutex blocking the current thread until it is available.
    #[inline]
    pub fn raw_lock(&self) {
        unsafe { self.0.lock() }
    }

    /// Attempts to lock the mutex without blocking, returning whether it was
    /// successfully acquired or not.
    #[inline]
    pub fn try_lock(&self) -> bool {
        unsafe { self.0.try_lock() }
    }

    /// Unlocks the mutex.
    ///
    /// Behavior is undefined if the current thread does not actually hold the
    /// mutex.
    #[inline]
    pub unsafe fn raw_unlock(&self) {
        self.0.unlock()
    }
}

impl Drop for MovableMutex {
    fn drop(&mut self) {
        unsafe { self.0.destroy() };
    }
}
