use crate::arch::wasm32;
use crate::mem;
use crate::sync::atomic::{AtomicUsize, Ordering::SeqCst};

pub struct Mutex {
    locked: AtomicUsize,
}

pub type MovableMutex = Mutex;

// Mutexes have a pretty simple implementation where they contain an `i32`
// internally that is 0 when unlocked and 1 when the mutex is locked.
// Acquisition has a fast path where it attempts to cmpxchg the 0 to a 1, and
// if it fails it then waits for a notification. Releasing a lock is then done
// by swapping in 0 and then notifying any waiters, if present.

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { locked: AtomicUsize::new(0) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        // nothing to do
    }

    pub unsafe fn lock(&self) {
        while !self.try_lock() {
            // SAFETY: the caller must uphold the safety contract for `memory_atomic_wait32`.
            let val = unsafe {
                wasm32::memory_atomic_wait32(
                    self.ptr(),
                    1,  // we expect our mutex is locked
                    -1, // wait infinitely
                )
            };
            // we should have either woke up (0) or got a not-equal due to a
            // race (1). We should never time out (2)
            debug_assert!(val == 0 || val == 1);
        }
    }

    pub unsafe fn unlock(&self) {
        let prev = self.locked.swap(0, SeqCst);
        debug_assert_eq!(prev, 1);
        wasm32::memory_atomic_notify(self.ptr(), 1); // wake up one waiter, if any
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        self.locked.compare_exchange(0, 1, SeqCst, SeqCst).is_ok()
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        // nothing to do
    }

    #[inline]
    fn ptr(&self) -> *mut i32 {
        assert_eq!(mem::size_of::<usize>(), mem::size_of::<i32>());
        self.locked.as_mut_ptr() as *mut i32
    }
}
