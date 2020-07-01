use crate::sys::rwlock as imp;

/// An OS-based reader-writer lock.
///
/// This structure is entirely unsafe and serves as the lowest layer of a
/// cross-platform binding of system rwlocks. It is recommended to use the
/// safer types at the top level of this crate instead of this type.
pub struct RWLock(imp::RWLock);

impl RWLock {
    /// Creates a new reader-writer lock for use.
    ///
    /// Behavior is undefined if the reader-writer lock is moved after it is
    /// first used with any of the functions below.
    pub const fn new() -> RWLock {
        RWLock(imp::RWLock::new())
    }

    /// Acquires shared access to the underlying lock, blocking the current
    /// thread to do so.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn read(&self) {
        // SAFETY: the caller must uphold the safety contract for `read`.
        unsafe { self.0.read() }
    }

    /// Attempts to acquire shared access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        // SAFETY: the caller must uphold the safety contract for `try_read`.
        unsafe { self.0.try_read() }
    }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn write(&self) {
        // SAFETY: the caller must uphold the safety contract for `write`.
        unsafe { self.0.write() }
    }

    /// Attempts to acquire exclusive access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        // SAFETY: the caller must uphold the safety contract for `try_write`.
        unsafe { self.0.try_write() }
    }

    /// Unlocks previously acquired shared access to this lock.
    ///
    /// Behavior is undefined if the current thread does not have shared access.
    #[inline]
    pub unsafe fn read_unlock(&self) {
        // SAFETY: the caller must uphold the safety contract for `read_unlock`.
        unsafe { self.0.read_unlock() }
    }

    /// Unlocks previously acquired exclusive access to this lock.
    ///
    /// Behavior is undefined if the current thread does not currently have
    /// exclusive access.
    #[inline]
    pub unsafe fn write_unlock(&self) {
        // SAFETY: the caller must uphold the safety contract for `write_unlock`.
        unsafe { self.0.write_unlock() }
    }

    /// Destroys OS-related resources with this RWLock.
    ///
    /// Behavior is undefined if there are any currently active users of this
    /// lock.
    #[inline]
    pub unsafe fn destroy(&self) {
        // SAFETY: the caller must uphold the safety contract for `destroy`.
        unsafe { self.0.destroy() }
    }
}
