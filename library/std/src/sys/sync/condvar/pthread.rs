#![forbid(unsafe_op_in_unsafe_fn)]

use crate::pin::Pin;
use crate::ptr;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering::Relaxed;
use crate::sys::pal::sync as pal;
use crate::sys::sync::{Mutex, OnceBox};
use crate::time::{Duration, Instant};

pub struct Condvar {
    cvar: OnceBox<pal::Condvar>,
    mutex: AtomicUsize,
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { cvar: OnceBox::new(), mutex: AtomicUsize::new(0) }
    }

    #[inline]
    fn get(&self) -> Pin<&pal::Condvar> {
        self.cvar.get_or_init(|| {
            let mut cvar = Box::pin(pal::Condvar::new());
            // SAFETY: we only call `init` once per `pal::Condvar`, namely here.
            unsafe { cvar.as_mut().init() };
            cvar
        })
    }

    #[inline]
    fn verify(&self, mutex: Pin<&pal::Mutex>) {
        let addr = ptr::from_ref::<pal::Mutex>(&mutex).addr();
        // Relaxed is okay here because we never read through `self.mutex`, and only use it to
        // compare addresses.
        match self.mutex.compare_exchange(0, addr, Relaxed, Relaxed) {
            Ok(_) => {}               // Stored the address
            Err(n) if n == addr => {} // Lost a race to store the same address
            _ => panic!("attempted to use a condition variable with two mutexes"),
        }
    }

    #[inline]
    pub fn notify_one(&self) {
        // SAFETY: we called `init` above.
        unsafe { self.get().notify_one() }
    }

    #[inline]
    pub fn notify_all(&self) {
        // SAFETY: we called `init` above.
        unsafe { self.get().notify_all() }
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        // SAFETY: the caller guarantees that the lock is owned, thus the mutex
        // must have been initialized already.
        let mutex = unsafe { mutex.pal.get_unchecked() };
        self.verify(mutex);
        // SAFETY: we called `init` above, we verified that this condition
        // variable is only used with `mutex` and the caller guarantees that
        // `mutex` is locked by the current thread.
        unsafe { self.get().wait(mutex) }
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        // SAFETY: the caller guarantees that the lock is owned, thus the mutex
        // must have been initialized already.
        let mutex = unsafe { mutex.pal.get_unchecked() };
        self.verify(mutex);

        if pal::Condvar::PRECISE_TIMEOUT {
            // SAFETY: we called `init` above, we verified that this condition
            // variable is only used with `mutex` and the caller guarantees that
            // `mutex` is locked by the current thread.
            unsafe { self.get().wait_timeout(mutex, dur) }
        } else {
            // Timeout reports are not reliable, so do the check ourselves.
            let now = Instant::now();
            // SAFETY: we called `init` above, we verified that this condition
            // variable is only used with `mutex` and the caller guarantees that
            // `mutex` is locked by the current thread.
            let woken = unsafe { self.get().wait_timeout(mutex, dur) };
            woken || now.elapsed() < dur
        }
    }
}
