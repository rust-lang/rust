//! Thread parking for Darwin-based systems.
//!
//! Darwin actually has futex syscalls (`__ulock_wait`/`__ulock_wake`), but they
//! cannot be used in `std` because they are non-public (their use will lead to
//! rejection from the App Store).
//!
//! Therefore, we need to look for other synchronization primitives. Luckily, Darwin
//! supports semaphores, which allow us to implement the behavior we need with
//! only one primitive (as opposed to a mutex-condvar pair). We use the semaphore
//! provided by libdispatch, as the underlying Mach semaphore is only dubiously
//! public.

#![allow(non_camel_case_types)]

use crate::pin::Pin;
use crate::sync::atomic::AtomicI8;
use crate::sync::atomic::Ordering::{Acquire, Release};
use crate::time::Duration;

type dispatch_semaphore_t = *mut crate::ffi::c_void;
type dispatch_time_t = u64;

const DISPATCH_TIME_NOW: dispatch_time_t = 0;
const DISPATCH_TIME_FOREVER: dispatch_time_t = !0;

// Contained in libSystem.dylib, which is linked by default.
unsafe extern "C" {
    fn dispatch_time(when: dispatch_time_t, delta: i64) -> dispatch_time_t;
    fn dispatch_semaphore_create(val: isize) -> dispatch_semaphore_t;
    fn dispatch_semaphore_wait(dsema: dispatch_semaphore_t, timeout: dispatch_time_t) -> isize;
    fn dispatch_semaphore_signal(dsema: dispatch_semaphore_t) -> isize;
    fn dispatch_release(object: *mut crate::ffi::c_void);
}

const EMPTY: i8 = 0;
const NOTIFIED: i8 = 1;
const PARKED: i8 = -1;

pub struct Parker {
    semaphore: dispatch_semaphore_t,
    state: AtomicI8,
}

unsafe impl Sync for Parker {}
unsafe impl Send for Parker {}

impl Parker {
    pub unsafe fn new_in_place(parker: *mut Parker) {
        let semaphore = dispatch_semaphore_create(0);
        assert!(
            !semaphore.is_null(),
            "failed to create dispatch semaphore for thread synchronization"
        );
        parker.write(Parker { semaphore, state: AtomicI8::new(EMPTY) })
    }

    // Does not need `Pin`, but other implementation do.
    pub unsafe fn park(self: Pin<&Self>) {
        // The semaphore counter must be zero at this point, because unparking
        // threads will not actually increase it until we signalled that we
        // are waiting.

        // Change NOTIFIED to EMPTY and EMPTY to PARKED.
        if self.state.fetch_sub(1, Acquire) == NOTIFIED {
            return;
        }

        // Another thread may increase the semaphore counter from this point on.
        // If it is faster than us, we will decrement it again immediately below.
        // If we are faster, we wait.

        // Ensure that the semaphore counter has actually been decremented, even
        // if the call timed out for some reason.
        while dispatch_semaphore_wait(self.semaphore, DISPATCH_TIME_FOREVER) != 0 {}

        // At this point, the semaphore counter is zero again.

        // We were definitely woken up, so we don't need to check the state.
        // Still, we need to reset the state using a swap to observe the state
        // change with acquire ordering.
        self.state.swap(EMPTY, Acquire);
    }

    // Does not need `Pin`, but other implementation do.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        if self.state.fetch_sub(1, Acquire) == NOTIFIED {
            return;
        }

        let nanos = dur.as_nanos().try_into().unwrap_or(i64::MAX);
        let timeout = dispatch_time(DISPATCH_TIME_NOW, nanos);

        let timeout = dispatch_semaphore_wait(self.semaphore, timeout) != 0;

        let state = self.state.swap(EMPTY, Acquire);
        if state == NOTIFIED && timeout {
            // If the state was NOTIFIED but semaphore_wait returned without
            // decrementing the count because of a timeout, it means another
            // thread is about to call semaphore_signal. We must wait for that
            // to happen to ensure the semaphore count is reset.
            while dispatch_semaphore_wait(self.semaphore, DISPATCH_TIME_FOREVER) != 0 {}
        } else {
            // Either a timeout occurred and we reset the state before any thread
            // tried to wake us up, or we were woken up and reset the state,
            // making sure to observe the state change with acquire ordering.
            // Either way, the semaphore counter is now zero again.
        }
    }

    // Does not need `Pin`, but other implementation do.
    pub fn unpark(self: Pin<&Self>) {
        let state = self.state.swap(NOTIFIED, Release);
        if state == PARKED {
            unsafe {
                dispatch_semaphore_signal(self.semaphore);
            }
        }
    }
}

impl Drop for Parker {
    fn drop(&mut self) {
        // SAFETY:
        // We always ensure that the semaphore count is reset, so this will
        // never cause an exception.
        unsafe {
            dispatch_release(self.semaphore);
        }
    }
}
