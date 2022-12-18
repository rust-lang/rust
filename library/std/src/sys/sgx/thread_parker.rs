//! Thread parking based on SGX events.

use super::abi::{thread, usercalls};
use crate::io::ErrorKind;
use crate::pin::Pin;
use crate::ptr::{self, NonNull};
use crate::sync::atomic::AtomicPtr;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::time::Duration;
use fortanix_sgx_abi::{EV_UNPARK, WAIT_INDEFINITE};

// The TCS structure must be page-aligned (this is checked by EENTER), so these cannot
// be valid pointers
const EMPTY: *mut u8 = ptr::invalid_mut(1);
const NOTIFIED: *mut u8 = ptr::invalid_mut(2);

pub struct Parker {
    /// The park state. One of EMPTY, NOTIFIED or a TCS address.
    /// A state change to NOTIFIED must be done with release ordering
    /// and be observed with acquire ordering so that operations after
    /// `thread::park` returns will not occur before the unpark message
    /// was sent.
    state: AtomicPtr<u8>,
}

impl Parker {
    /// Construct the thread parker. The UNIX parker implementation
    /// requires this to happen in-place.
    pub unsafe fn new(parker: *mut Parker) {
        unsafe { parker.write(Parker::new_internal()) }
    }

    pub(super) fn new_internal() -> Parker {
        Parker { state: AtomicPtr::new(EMPTY) }
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park(self: Pin<&Self>) {
        if self.state.load(Acquire) != NOTIFIED {
            let mut prev = EMPTY;
            loop {
                // Guard against changing TCS addresses by always setting the state to
                // the current value.
                let tcs = thread::current().as_ptr();
                if self.state.compare_exchange(prev, tcs, Relaxed, Acquire).is_ok() {
                    let event = usercalls::wait(EV_UNPARK, WAIT_INDEFINITE).unwrap();
                    assert!(event & EV_UNPARK == EV_UNPARK);
                    prev = tcs;
                } else {
                    // The state was definitely changed by another thread at this point.
                    // The only time this occurs is when the state is changed to NOTIFIED.
                    // We observed this change with acquire ordering, so we can simply
                    // change the state to EMPTY with a relaxed store.
                    break;
                }
            }
        }

        // At this point, the token was definately read with acquire ordering,
        // so this can be a relaxed store.
        self.state.store(EMPTY, Relaxed);
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        let timeout = u128::min(dur.as_nanos(), WAIT_INDEFINITE as u128 - 1) as u64;
        let tcs = thread::current().as_ptr();

        if self.state.load(Acquire) != NOTIFIED {
            if self.state.compare_exchange(EMPTY, tcs, Relaxed, Acquire).is_ok() {
                match usercalls::wait(EV_UNPARK, timeout) {
                    Ok(event) => assert!(event & EV_UNPARK == EV_UNPARK),
                    Err(e) => {
                        assert!(matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock))
                    }
                }

                // Swap to provide acquire ordering even if the timeout occurred
                // before the token was set. This situation can result in spurious
                // wakeups on the next call to `park_timeout`, but it is better to let
                // those be handled by the user than do some perhaps unnecessary, but
                // always expensive guarding.
                self.state.swap(EMPTY, Acquire);
                return;
            }
        }

        // The token was already read with `acquire` ordering, this can be a store.
        self.state.store(EMPTY, Relaxed);
    }

    // This implementation doesn't require `Pin`, but other implementations do.
    pub fn unpark(self: Pin<&Self>) {
        let state = self.state.swap(NOTIFIED, Release);

        if !matches!(state, EMPTY | NOTIFIED) {
            // There is a thread waiting, wake it up.
            let tcs = NonNull::new(state).unwrap();
            // This will fail if the thread has already terminated or its TCS is destroyed
            // by the time the signal is sent, but that is fine. If another thread receives
            // the same TCS, it will receive this notification as a spurious wakeup, but
            // all users of `wait` should and (internally) do guard against those where
            // necessary.
            let _ = usercalls::send(EV_UNPARK, Some(tcs));
        }
    }
}
