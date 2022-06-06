use crate::sys::locks as imp;

/// An OS-based reader-writer lock, meant for use in static variables.
///
/// This rwlock does not implement poisoning.
///
/// This rwlock has a const constructor ([`StaticRwLock::new`]), does not
/// implement `Drop` to cleanup resources.
pub struct StaticRwLock(imp::RwLock);

impl StaticRwLock {
    /// Creates a new rwlock for use.
    #[inline]
    pub const fn new() -> Self {
        Self(imp::RwLock::new())
    }

    /// Acquires shared access to the underlying lock, blocking the current
    /// thread to do so.
    ///
    /// The lock is automatically unlocked when the returned guard is dropped.
    #[inline]
    pub fn read(&'static self) -> StaticRwLockReadGuard {
        unsafe { self.0.read() };
        StaticRwLockReadGuard(&self.0)
    }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    ///
    /// The lock is automatically unlocked when the returned guard is dropped.
    #[inline]
    pub fn write(&'static self) -> StaticRwLockWriteGuard {
        unsafe { self.0.write() };
        StaticRwLockWriteGuard(&self.0)
    }
}

#[must_use]
pub struct StaticRwLockReadGuard(&'static imp::RwLock);

impl Drop for StaticRwLockReadGuard {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.0.read_unlock();
        }
    }
}

#[must_use]
pub struct StaticRwLockWriteGuard(&'static imp::RwLock);

impl Drop for StaticRwLockWriteGuard {
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
/// This is either a wrapper around `Box<imp::RwLock>` or `imp::RwLock`,
/// depending on the platform. It is boxed on platforms where `imp::RwLock` may
/// not be moved.
pub struct MovableRwLock(imp::MovableRwLock);

impl MovableRwLock {
    /// Creates a new reader-writer lock for use.
    #[inline]
    pub fn new() -> Self {
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
