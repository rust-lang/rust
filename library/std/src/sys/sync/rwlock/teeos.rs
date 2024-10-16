use crate::sys::sync::mutex::Mutex;

/// we do not supported rwlock, so use mutex to simulate rwlock.
/// it's useful because so many code in std will use rwlock.
pub struct RwLock {
    inner: Mutex,
}

impl RwLock {
    #[inline]
    pub const fn new() -> RwLock {
        RwLock { inner: Mutex::new() }
    }

    #[inline]
    pub fn read(&self) {
        self.inner.lock()
    }

    #[inline]
    pub fn try_read(&self) -> bool {
        self.inner.try_lock()
    }

    #[inline]
    pub fn write(&self) {
        self.inner.lock()
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        self.inner.try_lock()
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        unsafe { self.inner.unlock() };
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        unsafe { self.inner.unlock() };
    }

    #[inline]
    pub unsafe fn downgrade(&self) {
        // Since there is no difference between read-locked and write-locked on this platform, this
        // function is simply a no-op as only 1 reader can read: the original writer.
    }
}
