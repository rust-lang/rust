use crate::sync::atomic::AtomicU32;
use crate::sys::cloudabi::abi;
use crate::sys::rwlock::{self, RWLock};

extern "C" {
    #[thread_local]
    static __pthread_thread_id: abi::tid;
}

// Implement Mutex using an RWLock. This doesn't introduce any
// performance overhead in this environment, as the operations would be
// implemented identically.
pub struct Mutex(RWLock);

pub unsafe fn raw(m: &Mutex) -> *mut AtomicU32 {
    rwlock::raw(&m.0)
}

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex(RWLock::new())
    }

    pub unsafe fn init(&mut self) {
        // This function should normally reinitialize the mutex after
        // moving it to a different memory address. This implementation
        // does not require adjustments after moving.
    }

    pub unsafe fn try_lock(&self) -> bool {
        self.0.try_write()
    }

    pub unsafe fn lock(&self) {
        self.0.write()
    }

    pub unsafe fn unlock(&self) {
        self.0.write_unlock()
    }

    pub unsafe fn destroy(&self) {
        self.0.destroy()
    }
}

