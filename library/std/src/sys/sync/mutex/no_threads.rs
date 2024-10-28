use crate::cell::Cell;

pub struct Mutex {
    // This platform has no threads, so we can use a Cell here.
    locked: Cell<bool>,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {} // no threads on this platform

impl Mutex {
    #[inline]
    #[cfg_attr(bootstrap, rustc_const_stable(feature = "const_locks", since = "1.63.0"))]
    pub const fn new() -> Mutex {
        Mutex { locked: Cell::new(false) }
    }

    #[inline]
    pub fn lock(&self) {
        assert_eq!(self.locked.replace(true), false, "cannot recursively acquire mutex");
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        self.locked.set(false);
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        self.locked.replace(true) == false
    }
}
