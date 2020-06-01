use crate::cell::UnsafeCell;

pub struct Mutex {
    locked: UnsafeCell<bool>,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {} // no threads on this platform

impl Mutex {
    #[rustc_const_stable(feature = "const_sys_mutex_new", since = "1.0.0")]
    pub const fn new() -> Mutex {
        Mutex { locked: UnsafeCell::new(false) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn lock(&self) {
        let locked = self.locked.get();
        assert!(!*locked, "cannot recursively acquire mutex");
        *locked = true;
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        *self.locked.get() = false;
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let locked = self.locked.get();
        if *locked {
            false
        } else {
            *locked = true;
            true
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}

// All empty stubs because this platform does not yet support threads, so lock
// acquisition always succeeds.
pub struct ReentrantMutex {}

impl ReentrantMutex {
    pub const unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex {}
    }

    pub unsafe fn init(&self) {}

    pub unsafe fn lock(&self) {}

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        true
    }

    pub unsafe fn unlock(&self) {}

    pub unsafe fn destroy(&self) {}
}
