use crate::ptr;
use crate::ffi::c_void;

extern "C" {
    fn sys_sem_init(sem: *mut *const c_void, value: u32) -> i32;
    fn sys_sem_destroy(sem: *const c_void) -> i32;
    fn sys_sem_post(sem: *const c_void) -> i32;
    fn sys_sem_trywait(sem: *const c_void) -> i32;
    fn sys_sem_timedwait(sem: *const c_void, ms: u32) -> i32;
    fn sys_recmutex_init(recmutex: *mut *const c_void) -> i32;
    fn sys_recmutex_destroy(recmutex: *const c_void) -> i32;
    fn sys_recmutex_lock(recmutex: *const c_void) -> i32;
    fn sys_recmutex_unlock(recmutex: *const c_void) -> i32;
}

pub struct Mutex {
    inner: *const c_void
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { inner: ptr::null() }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        let _ = sys_sem_init(&mut self.inner as *mut *const c_void, 1);
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let _ = sys_sem_timedwait(self.inner, 0);
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let _ = sys_sem_post(self.inner);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let result = sys_sem_trywait(self.inner);
        result == 0
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = sys_sem_destroy(self.inner);
    }
}

pub struct ReentrantMutex {
    inner: *const c_void
}

impl ReentrantMutex {
    pub unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { inner: ptr::null() }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        let _ = sys_recmutex_init(&mut self.inner as *mut *const c_void);
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let _ = sys_recmutex_lock(self.inner);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        true
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let _ = sys_recmutex_unlock(self.inner);
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = sys_recmutex_destroy(self.inner);
    }
}
