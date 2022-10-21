use crate::sys::locks as imp;

/// An OS-based mutual exclusion lock.
///
/// This mutex cleans up its resources in its `Drop` implementation, may safely
/// be moved (when not borrowed), and does not cause UB when used reentrantly.
///
/// This mutex does not implement poisoning.
///
/// This is either a wrapper around `LazyBox<imp::Mutex>` or `imp::Mutex`,
/// depending on the platform. It is boxed on platforms where `imp::Mutex` may
/// not be moved.
pub struct MovableMutex(imp::MovableMutex);

unsafe impl Sync for MovableMutex {}

impl MovableMutex {
    /// Creates a new mutex.
    #[inline]
    #[rustc_const_stable(feature = "const_locks", since = "1.63.0")]
    pub const fn new() -> Self {
        Self(imp::MovableMutex::new())
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
