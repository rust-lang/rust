//! Generic support for building blocking abstractions.

use crate::sync::atomic::{AtomicBool, Ordering};
use crate::sync::Arc;
use crate::thread::{self, Thread};
use crate::time::Instant;

struct Inner {
    thread: Thread,
    woken: AtomicBool,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

#[derive(Clone)]
pub struct SignalToken {
    inner: Arc<Inner>,
}

pub struct WaitToken {
    inner: Arc<Inner>,
}

impl !Send for WaitToken {}

impl !Sync for WaitToken {}

pub fn tokens() -> (WaitToken, SignalToken) {
    let inner = Arc::new(Inner { thread: thread::current(), woken: AtomicBool::new(false) });
    let wait_token = WaitToken { inner: inner.clone() };
    let signal_token = SignalToken { inner };
    (wait_token, signal_token)
}

impl SignalToken {
    pub fn signal(&self) -> bool {
        let wake = !self.inner.woken.compare_and_swap(false, true, Ordering::SeqCst);
        if wake {
            self.inner.thread.unpark();
        }
        wake
    }

    /// Converts to an unsafe usize value. Useful for storing in a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_to_usize(self) -> usize {
        Arc::into_raw(self.inner) as usize
    }

    /// Converts from an unsafe usize value. Useful for retrieving a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_from_usize(signal_ptr: usize) -> SignalToken {
        // SAFETY: this ok because of the internal represention of Arc, which
        // is a pointer and phantom data (so the size of a pointer -> a usize).
        //
        // NOTE: The call to `from_raw` is used in conjuction with the call to
        // `into_raw` made in the `SignalToken::cast_to_usize` method.
        SignalToken { inner: unsafe { Arc::from_raw(signal_ptr as *const _) } }
    }
}

impl WaitToken {
    pub fn wait(self) {
        while !self.inner.woken.load(Ordering::SeqCst) {
            thread::park()
        }
    }

    /// Returns `true` if we wake up normally.
    pub fn wait_max_until(self, end: Instant) -> bool {
        while !self.inner.woken.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now >= end {
                return false;
            }
            thread::park_timeout(end - now)
        }
        true
    }
}
