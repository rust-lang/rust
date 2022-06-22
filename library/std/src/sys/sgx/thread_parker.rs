//! Thread parking based on SGX events.

use super::abi::{thread, usercalls};
use crate::io::ErrorKind;
use crate::pin::Pin;
use crate::ptr::{self, NonNull};
use crate::sync::atomic::AtomicPtr;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::time::Duration;
use fortanix_sgx_abi::{EV_UNPARK, WAIT_INDEFINITE};

const EMPTY: *mut u8 = ptr::invalid_mut(0);
/// The TCS structure must be page-aligned, so this cannot be a valid pointer
const NOTIFIED: *mut u8 = ptr::invalid_mut(1);

pub struct Parker {
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
        let tcs = thread::current().as_ptr();

        if self.state.load(Acquire) != NOTIFIED {
            if self.state.compare_exchange(EMPTY, tcs, Acquire, Acquire).is_ok() {
                // Loop to guard against spurious wakeups.
                loop {
                    let event = usercalls::wait(EV_UNPARK, WAIT_INDEFINITE).unwrap();
                    assert!(event & EV_UNPARK == EV_UNPARK);
                    if self.state.load(Acquire) == NOTIFIED {
                        break;
                    }
                }
            }
        }

        // At this point, the token was definately read with acquire ordering,
        // so this can be a store.
        self.state.store(EMPTY, Relaxed);
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        let timeout = u128::min(dur.as_nanos(), WAIT_INDEFINITE as u128 - 1) as u64;
        let tcs = thread::current().as_ptr();

        if self.state.load(Acquire) != NOTIFIED {
            if self.state.compare_exchange(EMPTY, tcs, Acquire, Acquire).is_ok() {
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
            // This will fail if the thread has already terminated by the time the signal is send,
            // but that is OK.
            let _ = usercalls::send(EV_UNPARK, Some(tcs));
        }
    }
}
