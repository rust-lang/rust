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
        let _ = abi::sem_init(&mut self.inner as *mut *const c_void, 1);
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let _ = abi::sem_timedwait(self.inner, 0);
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let _ = abi::sem_post(self.inner);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let result = abi::sem_trywait(self.inner);
        result == 0
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = abi::sem_destroy(self.inner);
    }
}

pub struct ReentrantMutex {
    inner: *const c_void,
}

impl ReentrantMutex {
    pub unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { inner: ptr::null() }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        let _ = abi::recmutex_init(&mut self.inner as *mut *const c_void);
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let _ = abi::recmutex_lock(self.inner);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        true
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let _ = abi::recmutex_unlock(self.inner);
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = abi::recmutex_destroy(self.inner);
    }
}
