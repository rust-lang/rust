use crate::sys::rwlock as imp;

/// An OS-based reader-writer lock, meant for use in static variables.
///
/// This rwlock does not implement poisoning.
///
/// This rwlock has a const constructor ([`StaticRWLock::new`]), does not
/// implement `Drop` to cleanup resources.
pub struct StaticRWLock(imp::RWLock);

impl StaticRWLock {
    /// Creates a new rwlock for use.
    pub const fn new() -> Self {
        Self(imp::RWLock::new())
    }

    /// Acquires shared access to the underlying lock, blocking the current
    /// thread to do so.
    ///
    /// The lock is automatically unlocked when the returned guard is dropped.
    #[inline]
    pub fn read(&'static self) -> StaticRWLockReadGuard {
        unsafe { self.0.read() };
        StaticRWLockReadGuard(&self.0)
    }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    ///
    /// The lock is automatically unlocked when the returned guard is dropped.
    #[inline]
    pub fn write(&'static self) -> StaticRWLockWriteGuard {
        unsafe { self.0.write() };
        StaticRWLockWriteGuard(&self.0)
    }
}

#[must_use]
pub struct StaticRWLockReadGuard(&'static imp::RWLock);

impl Drop for StaticRWLockReadGuard {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.0.read_unlock();
        }
    }
}

#[must_use]
pub struct StaticRWLockWriteGuard(&'static imp::RWLock);

impl Drop for StaticRWLockWriteGuard {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.0.write_unlock();
        }
    }
}

/// An OS-based reader-writer lock.
///
/// This rwlock does *not* have a const constructor, cleans up its resources in
/// its `Drop` implementation and may safely be moved (when not borrowed).
///
/// This rwlock does not implement poisoning.
///
/// This is either a wrapper around `Box<imp::RWLock>` or `imp::RWLock`,
/// depending on the platform. It is boxed on platforms where `imp::RWLock` may
/// not be moved.
pub struct MovableRWLock(imp::MovableRWLock);

impl MovableRWLock {
    /// Creates a new reader-writer lock for use.
    pub fn new() -> Self {
        Self(imp::MovableRWLock::from(imp::RWLock::new()))
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

impl Drop for MovableRWLock {
    fn drop(&mut self) {
        unsafe { self.0.destroy() };
    }
}
