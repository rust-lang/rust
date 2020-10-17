use crate::sys::mutex as imp;

/// An OS-based mutual exclusion lock, meant for use in static variables.
///
/// This mutex has a const constructor ([`StaticMutex::new`]), does not
/// implement `Drop` to cleanup resources, and causes UB when used reentrantly.
///
/// This mutex does not implement poisoning.
///
/// This is a wrapper around `imp::Mutex` that does *not* call `init()` and
/// `destroy()`.
pub struct StaticMutex(imp::Mutex);

unsafe impl Sync for StaticMutex {}

impl StaticMutex {
    /// Creates a new mutex for use.
    pub const fn new() -> Self {
        Self(imp::Mutex::new())
    }

    /// Calls raw_lock() and then returns an RAII guard to guarantee the mutex
    /// will be unlocked.
    ///
    /// It is undefined behaviour to call this function while locked by the
    /// same thread.
    #[inline]
    pub unsafe fn lock(&'static self) -> StaticMutexGuard {
        self.0.lock();
        StaticMutexGuard(&self.0)
    }
}

#[must_use]
pub struct StaticMutexGuard(&'static imp::Mutex);

impl Drop for StaticMutexGuard {
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
/// This is either a wrapper around `Box<imp::Mutex>` or `imp::Mutex`,
/// depending on the platform. It is boxed on platforms where `imp::Mutex` may
/// not be moved.
pub struct MovableMutex(imp::MovableMutex);

unsafe impl Sync for MovableMutex {}

impl MovableMutex {
    /// Creates a new mutex.
    pub fn new() -> Self {
        let mut mutex = imp::MovableMutex::from(imp::Mutex::new());
        unsafe { mutex.init() };
        Self(mutex)
    }

    pub(super) fn raw(&self) -> &imp::Mutex {
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
