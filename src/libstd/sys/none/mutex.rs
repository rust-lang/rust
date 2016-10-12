use cell::Cell;

pub struct Mutex { locked: Cell<bool> }

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { locked: Cell::new(false) }
    }
    #[inline]
    pub unsafe fn init(&mut self) {}
    #[inline]
    pub unsafe fn lock(&self) {
        // Panic if trying to re-obtain lock
        debug_assert!(!self.locked.get());
        self.locked.set(true);
    }
    #[inline]
    pub unsafe fn unlock(&self) {
        debug_assert!(self.locked.get());
        self.locked.set(false);
    }
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let already_locked=self.locked.get();
        if !already_locked {
            self.lock()
        }
        !already_locked
    }
    #[inline]
    pub unsafe fn destroy(&self) {
        self.lock()
    }
}

pub struct ReentrantMutex(());

unsafe impl Send for ReentrantMutex {}
unsafe impl Sync for ReentrantMutex {}

impl ReentrantMutex {
    #[inline] pub unsafe fn uninitialized() -> ReentrantMutex { ReentrantMutex(()) }
    #[inline] pub unsafe fn init(&mut self) {}
    #[inline] pub unsafe fn lock(&self) {}
    #[inline] pub unsafe fn unlock(&self) {}
    #[inline] pub unsafe fn try_lock(&self) -> bool { true }
    #[inline] pub unsafe fn destroy(&self) {}
}
