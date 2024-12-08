// Only used on NetBSD. If other platforms start using id-based parking, use
// separate modules for each platform.
#![cfg(target_os = "netbsd")]

use libc::{_lwp_self, CLOCK_MONOTONIC, c_long, clockid_t, lwpid_t, time_t, timespec};

use crate::ffi::{c_int, c_void};
use crate::ptr;
use crate::time::Duration;

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
pub fn park(hint: usize) {
    unsafe {
        ___lwp_park60(0, 0, ptr::null_mut(), 0, ptr::without_provenance(hint), ptr::null());
    }
}

pub fn park_timeout(dur: Duration, hint: usize) {
    let mut timeout = timespec {
        // Saturate so that the operation will definitely time out
        // (even if it is after the heat death of the universe).
        tv_sec: dur.as_secs().try_into().ok().unwrap_or(time_t::MAX),
        tv_nsec: dur.subsec_nanos() as c_long,
    };

    // Timeout needs to be mutable since it is modified on NetBSD 9.0 and
    // above.
    unsafe {
        ___lwp_park60(
            CLOCK_MONOTONIC,
            0,
            &mut timeout,
            0,
            ptr::without_provenance(hint),
            ptr::null(),
        );
    }
}

#[inline]
pub fn unpark(tid: ThreadId, hint: usize) {
    unsafe {
        _lwp_unpark(tid, ptr::without_provenance(hint));
    }
}
