use super::pthread_mutex::PthreadMutexAttr;
use crate::cell::UnsafeCell;
use crate::mem::MaybeUninit;
use crate::sys::cvt_nz;

pub struct ReentrantMutex {
    inner: UnsafeCell<libc::pthread_mutex_t>,
}

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    pub const unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { inner: UnsafeCell::new(libc::PTHREAD_MUTEX_INITIALIZER) }
    }

    pub unsafe fn init(&self) {
        let mut attr = MaybeUninit::<libc::pthread_mutexattr_t>::uninit();
        cvt_nz(libc::pthread_mutexattr_init(attr.as_mut_ptr())).unwrap();
        let attr = PthreadMutexAttr(&mut attr);
        cvt_nz(libc::pthread_mutexattr_settype(attr.0.as_mut_ptr(), libc::PTHREAD_MUTEX_RECURSIVE))
            .unwrap();
        cvt_nz(libc::pthread_mutex_init(self.inner.get(), attr.0.as_ptr())).unwrap();
    }

    pub unsafe fn lock(&self) {
        let result = libc::pthread_mutex_lock(self.inner.get());
        debug_assert_eq!(result, 0);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        libc::pthread_mutex_trylock(self.inner.get()) == 0
    }

    pub unsafe fn unlock(&self) {
        let result = libc::pthread_mutex_unlock(self.inner.get());
        debug_assert_eq!(result, 0);
    }

    pub unsafe fn destroy(&self) {
        let result = libc::pthread_mutex_destroy(self.inner.get());
        debug_assert_eq!(result, 0);
    }
}
