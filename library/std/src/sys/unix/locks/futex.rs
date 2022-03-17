use crate::sync::atomic::{
    AtomicI32,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::sys::futex::{futex_wait, futex_wake, futex_wake_all};
use crate::time::Duration;

pub type MovableMutex = Mutex;
pub type MovableCondvar = Condvar;

pub struct Mutex {
    /// 0: unlocked
    /// 1: locked, no other threads waiting
    /// 2: locked, and other threads waiting (contended)
    futex: AtomicI32,
}

impl Mutex {
    pub const fn new() -> Self {
        Self { futex: AtomicI32::new(0) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn destroy(&self) {}

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

    fn lock_contended(&self) {
        loop {
            // Put the lock in contended state, if it wasn't already.
            if self.futex.swap(2, Acquire) == 0 {
                // It was unlocked, so we just locked it.
                return;
            }
            // Wait for the futex to change state.
            futex_wait(&self.futex, 2, None);
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        if self.futex.swap(0, Release) == 2 {
            // We only wake up one thread. When that thread locks the mutex, it
            // will mark the mutex as contended (2) (see lock_contended above),
            // which makes sure that any other waiting threads will also be
            // woken up eventually.
            futex_wake(&self.futex);
        }
    }
}

pub struct Condvar {
    // The value of this atomic is simply incremented on every notification.
    // This is used by `.wait()` to not miss any notifications after
    // unlocking the mutex and before waiting for notifications.
    futex: AtomicI32,
}

impl Condvar {
    #[inline]
    pub const fn new() -> Self {
        Self { futex: AtomicI32::new(0) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn destroy(&self) {}

    // All the memory orderings here are `Relaxed`,
    // because synchronization is done by unlocking and locking the mutex.

    #[inline]
    pub unsafe fn notify_one(&self) {
        self.futex.fetch_add(1, Relaxed);
        futex_wake(&self.futex);
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        self.futex.fetch_add(1, Relaxed);
        futex_wake_all(&self.futex);
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        self.wait_optional_timeout(mutex, None);
    }

    #[inline]
    pub unsafe fn wait_timeout(&self, mutex: &Mutex, timeout: Duration) -> bool {
        self.wait_optional_timeout(mutex, Some(timeout))
    }

    unsafe fn wait_optional_timeout(&self, mutex: &Mutex, timeout: Option<Duration>) -> bool {
        // Check the notification counter before we unlock the mutex.
        let futex_value = self.futex.load(Relaxed);

        // Unlock the mutex before going to sleep.
        mutex.unlock();

        // Wait, but only if there hasn't been any
        // notification since we unlocked the mutex.
        let r = futex_wait(&self.futex, futex_value, timeout);

        // Lock the mutex again.
        mutex.lock();

        r
    }
}
