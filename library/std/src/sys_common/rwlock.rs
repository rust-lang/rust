use crate::sys::locks as imp;

/// An OS-based reader-writer lock.
///
/// This rwlock cleans up its resources in its `Drop` implementation and may
/// safely be moved (when not borrowed).
///
/// This rwlock does not implement poisoning.
///
/// This is either a wrapper around `LazyBox<imp::RwLock>` or `imp::RwLock`,
/// depending on the platform. It is boxed on platforms where `imp::RwLock` may
/// not be moved.
pub struct MovableRwLock(imp::MovableRwLock);

impl MovableRwLock {
    /// Creates a new reader-writer lock for use.
    #[inline]
    #[rustc_const_stable(feature = "const_locks", since = "1.63.0")]
    pub const fn new() -> Self {
        Self(imp::MovableRwLock::new())
    }

    /// Acquires shared access to the underlying lock, blocking the current
    /// thread to do so.
    #[inline]
    pub fn read(&self) {
        unsafe { self.0.read() }
    }

    /// Attempts to acquire shared access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    #[inline]
    pub fn try_read(&self) -> bool {
        unsafe { self.0.try_read() }
    }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    #[inline]
    pub fn write(&self) {
        unsafe { self.0.write() }
    }

    /// Attempts to acquire exclusive access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    #[inline]
    pub fn try_write(&self) -> bool {
        unsafe { self.0.try_write() }
    }

    /// Unlocks previously acquired shared access to this lock.
    ///
    /// Behavior is undefined if the current thread does not have shared access.
    #[inline]
    pub unsafe fn read_unlock(&self) {
        self.0.read_unlock()
    }

    /// Unlocks previously acquired exclusive access to this lock.
    ///
    /// Behavior is undefined if the current thread does not currently have
    /// exclusive access.
    #[inline]
    pub unsafe fn write_unlock(&self) {
        self.0.write_unlock()
    }
}
