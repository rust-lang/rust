use crate::cell::UnsafeCell;

pub struct Mutex {
    locked: UnsafeCell<bool>,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {} // no threads on wasm

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { locked: UnsafeCell::new(false) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
    }

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
    pub unsafe fn destroy(&self) {
    }
}
