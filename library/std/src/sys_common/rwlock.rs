use crate::marker::PhantomPinned;
use crate::pin::Pin;
use crate::sys::rwlock as imp;

/// An OS-based reader-writer lock.
///
/// This structure is entirely unsafe and serves as the lowest layer of a
/// cross-platform binding of system rwlocks. It is recommended to use the
/// safer types at the top level of this crate instead of this type.
pub struct RWLock {
    inner: imp::RWLock,
    _pinned: PhantomPinned,
}

impl RWLock {
    /// Creates a new reader-writer lock for use.
    ///
    /// Behavior is undefined if the reader-writer lock is moved after it is
    /// first used with any of the functions below.
    pub const fn new() -> RWLock {
        RWLock { inner: imp::RWLock::new(), _pinned: PhantomPinned }
    }

    /// Acquires shared access to the underlying lock, blocking the current
    /// thread to do so.
    #[inline]
    pub fn read(self: Pin<&Self>) {
        unsafe { self.inner.read() }
    }

    /// Attempts to acquire shared access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    #[inline]
    pub fn try_read(self: Pin<&Self>) -> bool {
        unsafe { self.inner.try_read() }
    }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    #[inline]
    pub fn write(self: Pin<&Self>) {
        unsafe { self.inner.write() }
    }

    /// Attempts to acquire exclusive access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    #[inline]
    pub fn try_write(self: Pin<&Self>) -> bool {
        unsafe { self.inner.try_write() }
    }

    /// Unlocks previously acquired shared access to this lock.
    ///
    /// Behavior is undefined if the current thread does not have shared access.
    #[inline]
    pub unsafe fn read_unlock(self: Pin<&Self>) {
        self.inner.read_unlock()
    }

    /// Unlocks previously acquired exclusive access to this lock.
    ///
    /// Behavior is undefined if the current thread does not currently have
    /// exclusive access.
    #[inline]
    pub unsafe fn write_unlock(self: Pin<&Self>) {
        self.inner.write_unlock()
    }
}

impl Drop for RWLock {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: The rwlock wasn't moved since using any of its
        // functions, because they all require a Pin.
        unsafe { self.inner.destroy() }
    }
}
