//! Parker implementation based on a Mutex and Condvar.

use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering::SeqCst;
use crate::sync::{Condvar, Mutex};
use crate::time::Duration;

const EMPTY: usize = 0;
const PARKED: usize = 1;
const NOTIFIED: usize = 2;

pub struct Parker {
    state: AtomicUsize,
    lock: Mutex<()>,
    cvar: Condvar,
}

impl Parker {
    pub fn new() -> Self {
        Parker { state: AtomicUsize::new(EMPTY), lock: Mutex::new(()), cvar: Condvar::new() }
    }

    // This implementation doesn't require `unsafe`, but other implementations
    // may assume this is only called by the thread that owns the Parker.
    pub unsafe fn park(&self) {
        // If we were previously notified then we consume this notification and
        // return quickly.
        if self.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst).is_ok() {
            return;
        }

        // Otherwise we need to coordinate going to sleep
        let mut m = self.lock.lock().unwrap();
        match self.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read here, even though we know it will be `NOTIFIED`.
                // This is because `unpark` may have been called again since we read
                // `NOTIFIED` in the `compare_exchange` above. We must perform an
                // acquire operation that synchronizes with that `unpark` to observe
                // any writes it made before the call to unpark. To do that we must
                // read from the write it made to `state`.
                let old = self.state.swap(EMPTY, SeqCst);
                assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
                return;
            } // should consume this notification, so prohibit spurious wakeups in next park.
            Err(_) => panic!("inconsistent park state"),
        }
        loop {
            m = self.cvar.wait(m).unwrap();
            match self.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst) {
                Ok(_) => return, // got a notification
                Err(_) => {}     // spurious wakeup, go back to sleep
            }
        }
    }

    // This implementation doesn't require `unsafe`, but other implementations
    // may assume this is only called by the thread that owns the Parker.
    pub unsafe fn park_timeout(&self, dur: Duration) {
        // Like `park` above we have a fast path for an already-notified thread, and
        // afterwards we start coordinating for a sleep.
        // return quickly.
        if self.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst).is_ok() {
            return;
        }
        let m = self.lock.lock().unwrap();
        match self.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read again here, see `park`.
                let old = self.state.swap(EMPTY, SeqCst);
                assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
                return;
            } // should consume this notification, so prohibit spurious wakeups in next park.
            Err(_) => panic!("inconsistent park_timeout state"),
        }

        // Wait with a timeout, and if we spuriously wake up or otherwise wake up
        // from a notification we just want to unconditionally set the state back to
        // empty, either consuming a notification or un-flagging ourselves as
        // parked.
        let (_m, _result) = self.cvar.wait_timeout(m, dur).unwrap();
        match self.state.swap(EMPTY, SeqCst) {
            NOTIFIED => {} // got a notification, hurray!
            PARKED => {}   // no notification, alas
            n => panic!("inconsistent park_timeout state: {}", n),
        }
    }

    pub fn unpark(&self) {
        // To ensure the unparked thread will observe any writes we made
        // before this call, we must perform a release operation that `park`
        // can synchronize with. To do that we must write `NOTIFIED` even if
        // `state` is already `NOTIFIED`. That is why this must be a swap
        // rather than a compare-and-swap that returns if it reads `NOTIFIED`
        // on failure.
        match self.state.swap(NOTIFIED, SeqCst) {
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
        drop(self.lock.lock().unwrap());
        self.cvar.notify_one()
    }
}
