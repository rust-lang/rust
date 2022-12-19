#![cfg(target_os = "netbsd")]

use crate::ffi::{c_int, c_void};
use crate::ptr::{null, null_mut};
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

pub type ThreadId = lwpid_t;

#[inline]
pub fn current() -> ThreadId {
    unsafe { _lwp_self() }
}

#[inline]
pub fn park() {
    unsafe {
        ___lwp_park60(0, 0, null_mut(), 0, null(), null());
    }
}

pub fn park_timeout(dur: Duration) {
    let mut timeout = timespec {
        // Saturate so that the operation will definitely time out
        // (even if it is after the heat death of the universe).
        tv_sec: dur.as_secs().try_into().ok().unwrap_or(time_t::MAX),
        tv_nsec: dur.subsec_nanos().into(),
    };

    // Timeout needs to be mutable since it is modified on NetBSD 9.0 and
    // above.
    unsafe {
        ___lwp_park60(CLOCK_MONOTONIC, 0, &mut timeout, 0, null(), null());
    }
}

#[inline]
pub fn unpark(tid: ThreadId) {
    unsafe {
        _lwp_unpark(tid, null());
    }
}
