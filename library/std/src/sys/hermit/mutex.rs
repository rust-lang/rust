#![deny(unsafe_op_in_unsafe_fn)]

use crate::ffi::c_void;
use crate::ptr;
use crate::sys::hermit::abi;

pub struct Mutex {
    inner: *const c_void,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { inner: ptr::null() }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        unsafe {
            let _ = abi::sem_init(&mut self.inner as *mut *const c_void, 1);
        }
    }

    #[inline]
    pub unsafe fn lock(&self) {
        unsafe {
            let _ = abi::sem_timedwait(self.inner, 0);
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        unsafe {
            let _ = abi::sem_post(self.inner);
        }
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let result = unsafe { abi::sem_trywait(self.inner) };
        result == 0
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = unsafe { abi::sem_destroy(self.inner) };
    }
}

pub struct ReentrantMutex {
    inner: *const c_void,
}

impl ReentrantMutex {
    pub const unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { inner: ptr::null() }
    }

    #[inline]
    pub unsafe fn init(&self) {
        let _ = unsafe { abi::recmutex_init(&self.inner as *const *const c_void as *mut _) };
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let _ = unsafe { abi::recmutex_lock(self.inner) };
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        true
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let _ = unsafe { abi::recmutex_unlock(self.inner) };
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = unsafe { abi::recmutex_destroy(self.inner) };
    }
}
