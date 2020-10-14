use crate::cell::Cell;

pub struct Mutex {
    // This platform has no threads, so we can use a Cell here.
    locked: Cell<bool>,
}

pub type MovableMutex = Mutex;

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {} // no threads on this platform

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { locked: Cell::new(false) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn lock(&self) {
        assert_eq!(self.locked.replace(true), false, "cannot recursively acquire mutex");
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        self.locked.set(false);
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        self.locked.replace(true) == false
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
