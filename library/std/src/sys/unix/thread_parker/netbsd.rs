use crate::ffi::{c_int, c_void};
use crate::pin::Pin;
use crate::ptr::{null, null_mut};
use crate::sync::atomic::{
    AtomicU64,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::time::Duration;
use libc::{_lwp_self, clockid_t, lwpid_t, time_t, timespec, CLOCK_MONOTONIC};

extern "C" {
    fn ___lwp_park60(
        clock_id: clockid_t,
        flags: c_int,
        ts: *mut timespec,
        unpark: lwpid_t,
        hint: *const c_void,
        unparkhint: *const c_void,
    ) -> c_int;
    fn _lwp_unpark(lwp: lwpid_t, hint: *const c_void) -> c_int;
}

/// The thread is not parked and the token is not available.
///
/// Zero cannot be a valid LWP id, since it is used as empty value for the unpark
/// argument in _lwp_park.
const EMPTY: u64 = 0;
/// The token is available. Do not park anymore.
const NOTIFIED: u64 = u64::MAX;

pub struct Parker {
    /// The parker state. Contains either one of the two state values above or the LWP
    /// id of the parked thread.
    state: AtomicU64,
}

impl Parker {
    pub unsafe fn new(parker: *mut Parker) {
        parker.write(Parker { state: AtomicU64::new(EMPTY) })
    }

    // Does not actually need `unsafe` or `Pin`, but the pthread implementation does.
    pub unsafe fn park(self: Pin<&Self>) {
        // If the token has already been made available, we can skip
        // a bit of work, so check for it here.
        if self.state.load(Acquire) != NOTIFIED {
            let parked = _lwp_self() as u64;
            let hint = self.state.as_mut_ptr().cast();
            if self.state.compare_exchange(EMPTY, parked, Relaxed, Acquire).is_ok() {
                // Loop to guard against spurious wakeups.
                loop {
                    ___lwp_park60(0, 0, null_mut(), 0, hint, null());
                    if self.state.load(Acquire) == NOTIFIED {
                        break;
                    }
                }
            }
        }

        // At this point, the change to NOTIFIED has always been observed with acquire
        // ordering, so we can just use a relaxed store here (instead of a swap).
        self.state.store(EMPTY, Relaxed);
    }

    // Does not actually need `unsafe` or `Pin`, but the pthread implementation does.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        if self.state.load(Acquire) != NOTIFIED {
            let parked = _lwp_self() as u64;
            let hint = self.state.as_mut_ptr().cast();
            let mut timeout = timespec {
                // Saturate so that the operation will definitely time out
                // (even if it is after the heat death of the universe).
                tv_sec: dur.as_secs().try_into().ok().unwrap_or(time_t::MAX),
                tv_nsec: dur.subsec_nanos().into(),
            };

            if self.state.compare_exchange(EMPTY, parked, Relaxed, Acquire).is_ok() {
                // Timeout needs to be mutable since it is modified on NetBSD 9.0 and
                // above.
                ___lwp_park60(CLOCK_MONOTONIC, 0, &mut timeout, 0, hint, null());
                // Use a swap to get acquire ordering even if the token was set after
                // the timeout occurred.
                self.state.swap(EMPTY, Acquire);
                return;
            }
        }

        self.state.store(EMPTY, Relaxed);
    }

    // Does not actually need `Pin`, but the pthread implementation does.
    pub fn unpark(self: Pin<&Self>) {
        let state = self.state.swap(NOTIFIED, Release);
        if !matches!(state, EMPTY | NOTIFIED) {
            let lwp = state as lwpid_t;
            let hint = self.state.as_mut_ptr().cast();

            // If the parking thread terminated and did not actually park, this will
            // probably return an error, which is OK. In the worst case, another
            // thread has received the same LWP id. It will then receive a spurious
            // wakeup, but those are allowable per the API contract. The same reasoning
            // applies if a timeout occurred before this call, but the state was not
            // yet reset.

            // SAFETY:
            // The syscall has no invariants to hold. Only unsafe because it is an
            // extern function.
            unsafe {
                _lwp_unpark(lwp, hint);
            }
        }
    }
}
