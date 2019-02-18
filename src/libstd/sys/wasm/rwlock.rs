use crate::cell::UnsafeCell;

pub struct RWLock {
    mode: UnsafeCell<isize>,
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {} // no threads on wasm

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock {
            mode: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        let mode = self.mode.get();
        if *mode >= 0 {
            *mode += 1;
        } else {
            rtabort!("rwlock locked for writing");
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let mode = self.mode.get();
        if *mode >= 0 {
            *mode += 1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub unsafe fn write(&self) {
        let mode = self.mode.get();
        if *mode == 0 {
            *mode = -1;
        } else {
            rtabort!("rwlock locked for reading")
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let mode = self.mode.get();
        if *mode == 0 {
            *mode = -1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        *self.mode.get() -= 1;
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        *self.mode.get() += 1;
    }

    #[inline]
    pub unsafe fn destroy(&self) {
    }
}
