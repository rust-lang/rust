use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sys::futex::{self, futex_wait, futex_wake};

type Futex = futex::SmallFutex;
type State = futex::SmallPrimitive;

pub struct Mutex {
    futex: Futex,
}

const UNLOCKED: State = 0;
const LOCKED: State = 1; // locked, no other threads waiting
const CONTENDED: State = 2; // locked, and other threads waiting (contended)

impl Mutex {
    #[inline]
    pub const fn new() -> Self {
        Self { futex: Futex::new(UNLOCKED) }
    }

    #[inline]
    // Make this a diagnostic item for Miri's concurrency model checker.
    #[cfg_attr(not(test), rustc_diagnostic_item = "sys_mutex_try_lock")]
    pub fn try_lock(&self) -> bool {
        self.futex.compare_exchange(UNLOCKED, LOCKED, Acquire, Relaxed).is_ok()
    }

    #[inline]
    // Make this a diagnostic item for Miri's concurrency model checker.
    #[cfg_attr(not(test), rustc_diagnostic_item = "sys_mutex_lock")]
    pub fn lock(&self) {
        if self.futex.compare_exchange(UNLOCKED, LOCKED, Acquire, Relaxed).is_err() {
            self.lock_contended();
        }
    }

    #[cold]
    fn lock_contended(&self) {
        // Spin first to speed things up if the lock is released quickly.
        let mut state = self.spin();

        // If it's unlocked now, attempt to take the lock
        // without marking it as contended.
        if state == UNLOCKED {
            match self.futex.compare_exchange(UNLOCKED, LOCKED, Acquire, Relaxed) {
                Ok(_) => return, // Locked!
                Err(s) => state = s,
            }
        }

        loop {
            // Put the lock in contended state.
            // We avoid an unnecessary write if it as already set to CONTENDED,
            // to be friendlier for the caches.
            if state != CONTENDED && self.futex.swap(CONTENDED, Acquire) == UNLOCKED {
                // We changed it from UNLOCKED to CONTENDED, so we just successfully locked it.
                return;
            }

            // Wait for the futex to change state, assuming it is still CONTENDED.
            futex_wait(&self.futex, CONTENDED, None);

            // Spin again after waking up.
            state = self.spin();
        }
    }

    fn spin(&self) -> State {
        let mut spin = 100;
        loop {
            // We only use `load` (and not `swap` or `compare_exchange`)
            // while spinning, to be easier on the caches.
            let state = self.futex.load(Relaxed);

            // We stop spinning when the mutex is UNLOCKED,
            // but also when it's CONTENDED.
            if state != LOCKED || spin == 0 {
                return state;
            }

            crate::hint::spin_loop();
            spin -= 1;
        }
    }

    #[inline]
    // Make this a diagnostic item for Miri's concurrency model checker.
    #[cfg_attr(not(test), rustc_diagnostic_item = "sys_mutex_unlock")]
    pub unsafe fn unlock(&self) {
        if self.futex.swap(UNLOCKED, Release) == CONTENDED {
            // We only wake up one thread. When that thread locks the mutex, it
            // will mark the mutex as CONTENDED (see lock_contended above),
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
