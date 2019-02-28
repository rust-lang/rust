use crate::cell::UnsafeCell;
use crate::sys::c;

pub struct RWLock { inner: UnsafeCell<c::SRWLOCK> }

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock { inner: UnsafeCell::new(c::SRWLOCK_INIT) }
    }
    #[inline]
    pub unsafe fn read(&self) {
        c::AcquireSRWLockShared(self.inner.get())
    }
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        c::TryAcquireSRWLockShared(self.inner.get()) != 0
    }
    #[inline]
    pub unsafe fn write(&self) {
        c::AcquireSRWLockExclusive(self.inner.get())
    }
    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        c::TryAcquireSRWLockExclusive(self.inner.get()) != 0
    }
    #[inline]
    pub unsafe fn read_unlock(&self) {
        c::ReleaseSRWLockShared(self.inner.get())
    }
    #[inline]
    pub unsafe fn write_unlock(&self) {
        c::ReleaseSRWLockExclusive(self.inner.get())
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        // ...
    }
}
