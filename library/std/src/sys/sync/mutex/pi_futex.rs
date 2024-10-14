use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sys::pi_futex as pi;

pub struct Mutex {
    futex: pi::Futex,
}

impl Mutex {
    #[inline]
    pub const fn new() -> Self {
        Self { futex: pi::Futex::new() }
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        self.futex.compare_exchange(pi::unlocked(), pi::locked(), Acquire, Relaxed).is_ok()
    }

    #[inline]
    pub fn lock(&self) {
        if self.futex.compare_exchange(pi::unlocked(), pi::locked(), Acquire, Relaxed).is_err() {
            self.lock_contended();
        }
    }

    #[cold]
    fn lock_contended(&self) {
        // Spin first to speed things up if the lock is released quickly.
        let state = self.spin();

        // If it's unlocked now, attempt to take the lock.
        if state == pi::unlocked() {
            if self.try_lock() {
                return;
            }
        };

        pi::futex_lock(&self.futex).expect("failed to lock mutex");

        let state = self.futex.load(Relaxed);
        if pi::is_owned_died(state) {
            panic!(
                "failed to lock mutex because the thread owning it finished without unlocking it"
            );
        }
    }

    fn spin(&self) -> pi::State {
        let mut spin = 100;
        loop {
            // We only use `load` (and not `swap` or `compare_exchange`)
            // while spinning, to be easier on the caches.
            let state = self.futex.load(Relaxed);

            // We stop spinning when the mutex is unlocked,
            // but also when it's contended.
            if state == pi::unlocked() || pi::is_contended(state) || spin == 0 {
                return state;
            }

            crate::hint::spin_loop();
            spin -= 1;
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        if self.futex.compare_exchange(pi::locked(), pi::unlocked(), Release, Relaxed).is_err() {
            // We only wake up one thread. When that thread locks the mutex,
            // the kernel will mark the mutex as contended automatically
            // (futex != pi::locked() in this case),
            // which makes sure that any other waiting threads will also be
            // woken up eventually.
            self.wake();
        }
    }

    #[cold]
    fn wake(&self) {
        pi::futex_unlock(&self.futex).unwrap();
    }
}
