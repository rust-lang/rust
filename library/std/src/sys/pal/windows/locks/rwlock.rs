use crate::cell::UnsafeCell;
use crate::sys::c;

pub struct RwLock {
    inner: UnsafeCell<c::SRWLOCK>,
}

unsafe impl Send for RwLock {}
unsafe impl Sync for RwLock {}

impl RwLock {
    #[inline]
    pub const fn new() -> RwLock {
        RwLock { inner: UnsafeCell::new(c::SRWLOCK_INIT) }
    }
    #[inline]
    pub fn read(&self) {
        unsafe { c::AcquireSRWLockShared(self.inner.get()) }
    }
    #[inline]
    pub fn try_read(&self) -> bool {
        unsafe { c::TryAcquireSRWLockShared(self.inner.get()) != 0 }
    }
    #[inline]
    pub fn write(&self) {
        unsafe { c::AcquireSRWLockExclusive(self.inner.get()) }
    }
    #[inline]
    pub fn try_write(&self) -> bool {
        unsafe { c::TryAcquireSRWLockExclusive(self.inner.get()) != 0 }
    }
    #[inline]
    pub unsafe fn read_unlock(&self) {
        c::ReleaseSRWLockShared(self.inner.get())
    }
    #[inline]
    pub unsafe fn write_unlock(&self) {
        c::ReleaseSRWLockExclusive(self.inner.get())
    }
}
