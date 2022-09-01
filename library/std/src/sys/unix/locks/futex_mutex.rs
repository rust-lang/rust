use crate::sync::atomic::{
    AtomicU32,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::sys::futex::{futex_wait, futex_wake};

pub type MovableMutex = Mutex;

pub struct Mutex {
    /// 0: unlocked
    /// 1: locked, no other threads waiting
    /// 2: locked, and other threads waiting (contended)
    futex: AtomicU32,
}

impl Mutex {
    #[inline]
    pub const fn new() -> Self {
        Self { futex: AtomicU32::new(0) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        self.futex.compare_exchange(0, 1, Acquire, Relaxed).is_ok()
    }

    #[inline]
    pub unsafe fn lock(&self) {
        if self.futex.compare_exchange(0, 1, Acquire, Relaxed).is_err() {
            self.lock_contended();
        }
    }

    #[cold]
    fn lock_contended(&self) {
        // Spin first to speed things up if the lock is released quickly.
        let mut state = self.spin();

        // If it's unlocked now, attempt to take the lock
        // without marking it as contended.
        if state == 0 {
            match self.futex.compare_exchange(0, 1, Acquire, Relaxed) {
                Ok(_) => return, // Locked!
                Err(s) => state = s,
            }
        }

        loop {
            // Put the lock in contended state.
            // We avoid an unnecessary write if it as already set to 2,
            // to be friendlier for the caches.
            if state != 2 && self.futex.swap(2, Acquire) == 0 {
                // We changed it from 0 to 2, so we just successfully locked it.
                return;
            }

            // Wait for the futex to change state, assuming it is still 2.
            futex_wait(&self.futex, 2, None);

            // Spin again after waking up.
            state = self.spin();
        }
    }

    fn spin(&self) -> u32 {
        let mut spin = 100;
        loop {
            // We only use `load` (and not `swap` or `compare_exchange`)
            // while spinning, to be easier on the caches.
            let state = self.futex.load(Relaxed);

            // We stop spinning when the mutex is unlocked (0),
            // but also when it's contended (2).
            if state != 1 || spin == 0 {
                return state;
            }

            crate::hint::spin_loop();
            spin -= 1;
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        if self.futex.swap(0, Release) == 2 {
            // We only wake up one thread. When that thread locks the mutex, it
            // will mark the mutex as contended (2) (see lock_contended above),
            // which makes sure that any other waiting threads will also be
            // woken up eventually.
            self.wake();
        }
    }

    #[cold]
    fn wake(&self) {
        futex_wake(&self.futex);
    }
}
