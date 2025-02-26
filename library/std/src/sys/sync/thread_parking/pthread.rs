//! Thread parking without `futex` using the `pthread` synchronization primitives.

use crate::pin::Pin;
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sys::pal::sync::{Condvar, Mutex};
use crate::time::Duration;

const EMPTY: usize = 0;
const PARKED: usize = 1;
const NOTIFIED: usize = 2;

pub struct Parker {
    state: AtomicUsize,
    lock: Mutex,
    cvar: Condvar,
}

impl Parker {
    /// Constructs the UNIX parker in-place.
    ///
    /// # Safety
    /// The constructed parker must never be moved.
    pub unsafe fn new_in_place(parker: *mut Parker) {
        parker.write(Parker {
            state: AtomicUsize::new(EMPTY),
            lock: Mutex::new(),
            cvar: Condvar::new(),
        });

        Pin::new_unchecked(&mut (*parker).cvar).init();
    }

    fn lock(self: Pin<&Self>) -> Pin<&Mutex> {
        unsafe { self.map_unchecked(|p| &p.lock) }
    }

    fn cvar(self: Pin<&Self>) -> Pin<&Condvar> {
        unsafe { self.map_unchecked(|p| &p.cvar) }
    }

    // This implementation doesn't require `unsafe`, but other implementations
    // may assume this is only called by the thread that owns the Parker.
    //
    // For memory ordering, see futex.rs
    pub unsafe fn park(self: Pin<&Self>) {
        // If we were previously notified then we consume this notification and
        // return quickly.
        if self.state.compare_exchange(NOTIFIED, EMPTY, Acquire, Relaxed).is_ok() {
            return;
        }

        // Otherwise we need to coordinate going to sleep
        self.lock().lock();
        match self.state.compare_exchange(EMPTY, PARKED, Relaxed, Relaxed) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read here, even though we know it will be `NOTIFIED`.
                // This is because `unpark` may have been called again since we read
                // `NOTIFIED` in the `compare_exchange` above. We must perform an
                // acquire operation that synchronizes with that `unpark` to observe
                // any writes it made before the call to unpark. To do that we must
                // read from the write it made to `state`.
                let old = self.state.swap(EMPTY, Acquire);

                self.lock().unlock();

                assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
                return;
            } // should consume this notification, so prohibit spurious wakeups in next park.
            Err(_) => {
                self.lock().unlock();

                panic!("inconsistent park state")
            }
        }

        loop {
            self.cvar().wait(self.lock());

            match self.state.compare_exchange(NOTIFIED, EMPTY, Acquire, Relaxed) {
                Ok(_) => break, // got a notification
                Err(_) => {}    // spurious wakeup, go back to sleep
            }
        }

        self.lock().unlock();
    }

    // This implementation doesn't require `unsafe`, but other implementations
    // may assume this is only called by the thread that owns the Parker. Use
    // `Pin` to guarantee a stable address for the mutex and condition variable.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        // Like `park` above we have a fast path for an already-notified thread, and
        // afterwards we start coordinating for a sleep.
        // return quickly.
        if self.state.compare_exchange(NOTIFIED, EMPTY, Acquire, Relaxed).is_ok() {
            return;
        }

        self.lock().lock();
        match self.state.compare_exchange(EMPTY, PARKED, Relaxed, Relaxed) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read again here, see `park`.
                let old = self.state.swap(EMPTY, Acquire);
                self.lock().unlock();

                assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
                return;
            } // should consume this notification, so prohibit spurious wakeups in next park.
            Err(_) => {
                self.lock().unlock();
                panic!("inconsistent park_timeout state")
            }
        }

        // Wait with a timeout, and if we spuriously wake up or otherwise wake up
        // from a notification we just want to unconditionally set the state back to
        // empty, either consuming a notification or un-flagging ourselves as
        // parked.
        self.cvar().wait_timeout(self.lock(), dur);

        match self.state.swap(EMPTY, Acquire) {
            NOTIFIED => self.lock().unlock(), // got a notification, hurray!
            PARKED => self.lock().unlock(),   // no notification, alas
            n => {
                self.lock().unlock();
                panic!("inconsistent park_timeout state: {n}")
            }
        }
    }

    pub fn unpark(self: Pin<&Self>) {
        // To ensure the unparked thread will observe any writes we made
        // before this call, we must perform a release operation that `park`
        // can synchronize with. To do that we must write `NOTIFIED` even if
        // `state` is already `NOTIFIED`. That is why this must be a swap
        // rather than a compare-and-swap that returns if it reads `NOTIFIED`
        // on failure.
        match self.state.swap(NOTIFIED, Release) {
            EMPTY => return,    // no one was waiting
            NOTIFIED => return, // already unparked
            PARKED => {}        // gotta go wake someone up
            _ => panic!("inconsistent state in unpark"),
        }

        // There is a period between when the parked thread sets `state` to
        // `PARKED` (or last checked `state` in the case of a spurious wake
        // up) and when it actually waits on `cvar`. If we were to notify
        // during this period it would be ignored and then when the parked
        // thread went to sleep it would never wake up. Fortunately, it has
        // `lock` locked at this stage so we can acquire `lock` to wait until
        // it is ready to receive the notification.
        //
        // Releasing `lock` before the call to `notify_one` means that when the
        // parked thread wakes it doesn't get woken only to have to wait for us
        // to release `lock`.
        unsafe {
            self.lock().lock();
            self.lock().unlock();
            self.cvar().notify_one();
        }
    }
}
