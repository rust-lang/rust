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
//! is or will be sleeping on the wait flag, so we raise it. Only the first thread
//! to call `unpark` will raise the wait flag, so spurious wakeups are avoided
//! (this is especially important for semaphores).

use crate::pin::Pin;
use crate::sync::atomic::AtomicI8;
use crate::sync::atomic::Ordering::SeqCst;
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
        // The state values are chosen so that this subtraction changes
        // `NOTIFIED` to `EMPTY` and `EMPTY` to `PARKED`.
        let state = self.state.fetch_sub(1, SeqCst);
        match state {
            EMPTY => (),
            NOTIFIED => return,
            _ => panic!("inconsistent park state"),
        }

        self.wait_flag.wait();

        // We need to do a load here to use `Acquire` ordering.
        self.state.swap(EMPTY, SeqCst);
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        let state = self.state.fetch_sub(1, SeqCst);
        match state {
            EMPTY => (),
            NOTIFIED => return,
            _ => panic!("inconsistent park state"),
        }

        let wakeup = self.wait_flag.wait_timeout(dur);
        let state = self.state.swap(EMPTY, SeqCst);
        if state == NOTIFIED && !wakeup {
            // The token was made available after the wait timed out, but before
            // we reset the state, so we need to reset the wait flag to avoid
            // spurious wakeups. This wait has no timeout, but we know it will
            // return quickly, as the unparking thread will definitely raise the
            // flag if it has not already done so.
            self.wait_flag.wait();
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
