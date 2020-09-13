#![deny(unsafe_op_in_unsafe_fn)]

use crate::cell::UnsafeCell;
use crate::sys::c;

pub struct RWLock {
    inner: UnsafeCell<c::SRWLOCK>,
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock { inner: UnsafeCell::new(c::SRWLOCK_INIT) }
    }
    #[inline]
    pub unsafe fn read(&self) {
        unsafe { c::AcquireSRWLockShared(self.inner.get()) }
    }
    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        unsafe { c::TryAcquireSRWLockShared(self.inner.get()) != 0 }
    }
    #[inline]
    pub unsafe fn write(&self) {
        unsafe { c::AcquireSRWLockExclusive(self.inner.get()) }
    }
    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        unsafe { c::TryAcquireSRWLockExclusive(self.inner.get()) != 0 }
    }
    #[inline]
    pub unsafe fn read_unlock(&self) {
        unsafe { c::ReleaseSRWLockShared(self.inner.get()) }
    }
    #[inline]
    pub unsafe fn write_unlock(&self) {
        unsafe { c::ReleaseSRWLockExclusive(self.inner.get()) }
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        // ...
    }
}
