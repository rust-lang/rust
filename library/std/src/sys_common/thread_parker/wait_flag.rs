//! A wait-flag-based thread parker.
//!
//! Some operating systems provide low-level parking primitives like wait counts,
//! event flags or semaphores which are not susceptible to race conditions (meaning
//! the wakeup can occur before the wait operation). To implement the `std` thread
//! parker on top of these primitives, we only have to ensure that parking is fast
//! when the thread token is available, the atomic ordering guarantees are maintained
//! and spurious wakeups are minimized.
//!
//! To achieve this, this parker uses an atomic variable with three states: `EMPTY`,
//! `PARKED` and `NOTIFIED`:
//! * `EMPTY` means the token has not been made available, but the thread is not
//!    currently waiting on it.
//! * `PARKED` means the token is not available and the thread is parked.
//! * `NOTIFIED` means the token is available.
//!
//! `park` and `park_timeout` change the state from `EMPTY` to `PARKED` and from
//! `NOTIFIED` to `EMPTY`. If the state was `NOTIFIED`, the thread was unparked and
//! execution can continue without calling into the OS. If the state was `EMPTY`,
//! the token is not available and the thread waits on the primitive (here called
//! "wait flag").
//!
//! `unpark` changes the state to `NOTIFIED`. If the state was `PARKED`, the thread
//! is or will be sleeping on the wait flag, so we raise it.

use crate::pin::Pin;
use crate::sync::atomic::AtomicI8;
use crate::sync::atomic::Ordering::{Relaxed, SeqCst};
use crate::sys::wait_flag::WaitFlag;
use crate::time::Duration;

const EMPTY: i8 = 0;
const PARKED: i8 = -1;
const NOTIFIED: i8 = 1;

pub struct Parker {
    state: AtomicI8,
    wait_flag: WaitFlag,
}

impl Parker {
    /// Construct a parker for the current thread. The UNIX parker
    /// implementation requires this to happen in-place.
    pub unsafe fn new(parker: *mut Parker) {
        parker.write(Parker { state: AtomicI8::new(EMPTY), wait_flag: WaitFlag::new() })
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park(self: Pin<&Self>) {
        match self.state.fetch_sub(1, SeqCst) {
            // NOTIFIED => EMPTY
            NOTIFIED => return,
            // EMPTY => PARKED
            EMPTY => (),
            _ => panic!("inconsistent park state"),
        }

        // Avoid waking up from spurious wakeups (these are quite likely, see below).
        loop {
            self.wait_flag.wait();

            match self.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, Relaxed) {
                Ok(_) => return,
                Err(PARKED) => (),
                Err(_) => panic!("inconsistent park state"),
            }
        }
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        match self.state.fetch_sub(1, SeqCst) {
            NOTIFIED => return,
            EMPTY => (),
            _ => panic!("inconsistent park state"),
        }

        self.wait_flag.wait_timeout(dur);

        // Either a wakeup or a timeout occurred. Wakeups may be spurious, as there can be
        // a race condition when `unpark` is performed between receiving the timeout and
        // resetting the state, resulting in the eventflag being set unnecessarily. `park`
        // is protected against this by looping until the token is actually given, but
        // here we cannot easily tell.

        // Use `swap` to provide acquire ordering (not strictly necessary, but all other
        // implementations do).
        match self.state.swap(EMPTY, SeqCst) {
            NOTIFIED => (),
            PARKED => (),
            _ => panic!("inconsistent park state"),
        }
    }

    // This implementation doesn't require `Pin`, but other implementations do.
    pub fn unpark(self: Pin<&Self>) {
        let state = self.state.swap(NOTIFIED, SeqCst);

        if state == PARKED {
            self.wait_flag.raise();
        }
    }
}
