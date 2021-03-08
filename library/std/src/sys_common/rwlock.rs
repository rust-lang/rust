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
        self.0.read()
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
        self.0.try_read()
    }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn write(&self) {
        self.0.write()
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
        self.0.try_write()
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

    /// Destroys OS-related resources with this RWLock.
    ///
    /// Behavior is undefined if there are any currently active users of this
    /// lock.
    #[inline]
    pub unsafe fn destroy(&self) {
        self.0.destroy()
    }
}

// the cfg annotations only exist due to dead code warnings. the code itself is portable
#[cfg(unix)]
pub struct StaticRWLock(RWLock);

#[cfg(unix)]
impl StaticRWLock {
    pub const fn new() -> StaticRWLock {
        StaticRWLock(RWLock::new())
    }

    /// Acquires shared access to the underlying lock, blocking the current
    /// thread to do so.
    ///
    /// The lock is automatically unlocked when the returned guard is dropped.
    #[inline]
    pub fn read_with_guard(&'static self) -> RWLockReadGuard {
        // SAFETY: All methods require static references, therefore self
        // cannot be moved between invocations.
        unsafe {
            self.0.read();
        }
        RWLockReadGuard(&self.0)
    }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    ///
    /// The lock is automatically unlocked when the returned guard is dropped.
    #[inline]
    pub fn write_with_guard(&'static self) -> RWLockWriteGuard {
        // SAFETY: All methods require static references, therefore self
        // cannot be moved between invocations.
        unsafe {
            self.0.write();
        }
        RWLockWriteGuard(&self.0)
    }
}

#[cfg(unix)]
pub struct RWLockReadGuard(&'static RWLock);

#[cfg(unix)]
impl Drop for RWLockReadGuard {
    fn drop(&mut self) {
        unsafe { self.0.read_unlock() }
    }
}

#[cfg(unix)]
pub struct RWLockWriteGuard(&'static RWLock);

#[cfg(unix)]
impl Drop for RWLockWriteGuard {
    fn drop(&mut self) {
        unsafe { self.0.write_unlock() }
    }
}
