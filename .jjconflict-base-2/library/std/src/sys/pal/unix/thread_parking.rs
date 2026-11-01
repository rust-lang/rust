// Only used on NetBSD. If other platforms start using id-based parking, use
// separate modules for each platform.
#![cfg(target_os = "netbsd")]

use libc::{_lwp_park, _lwp_self, _lwp_unpark, CLOCK_MONOTONIC, c_long, lwpid_t, time_t, timespec};

use crate::ptr;
use crate::time::Duration;

pub type ThreadId = lwpid_t;

#[inline]
pub fn current() -> ThreadId {
    unsafe { _lwp_self() }
}

#[inline]
pub fn park(hint: usize) {
    unsafe {
        _lwp_park(0, 0, ptr::null_mut(), 0, ptr::without_provenance(hint), ptr::null_mut());
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
        _lwp_park(
            CLOCK_MONOTONIC,
            0,
            &mut timeout,
            0,
            ptr::without_provenance(hint),
            ptr::null_mut(),
        );
    }
}

#[inline]
pub fn unpark(tid: ThreadId, hint: usize) {
    unsafe {
        _lwp_unpark(tid, ptr::without_provenance(hint));
    }
}
