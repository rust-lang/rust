use crate::convert::TryFrom;
use crate::ptr;
use crate::sync::atomic::{
    AtomicI8, AtomicUsize,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::sys::{c, dur2timeout};
use crate::time::Duration;

pub struct Parker {
    state: AtomicI8,
}

const PARKED: i8 = -1;
const EMPTY: i8 = 0;
const NOTIFIED: i8 = 1;

// Notes about memory ordering:
//
// Memory ordering is only relevant for the relative ordering of operations
// between different variables. Even Ordering::Relaxed guarantees a
// monotonic/consistent order when looking at just a single atomic variable.
//
// So, since this parker is just a single atomic variable, we only need to look
// at the ordering guarantees we need to provide to the 'outside world'.
//
// The only memory ordering guarantee that parking and unparking provide, is
// that things which happened before unpark() are visible on the thread
// returning from park() afterwards. Otherwise, it was effectively unparked
// before unpark() was called while still consuming the 'token'.
//
// In other words, unpark() needs to synchronize with the part of park() that
// consumes the token and returns.
//
// This is done with a release-acquire synchronization, by using
// Ordering::Release when writing NOTIFIED (the 'token') in unpark(), and using
// Ordering::Acquire when checking for this state in park().
impl Parker {
    pub fn new() -> Self {
        Self { state: AtomicI8::new(EMPTY) }
    }

    // Assumes this is only called by the thread that owns the Parker,
    // which means that `self.state != PARKED`.
    pub unsafe fn park(&self) {
        // Change NOTIFIED=>EMPTY or EMPTY=>PARKED, and directly return in the
        // first case.
        if self.state.fetch_sub(1, Acquire) == NOTIFIED {
            return;
        }

        loop {
            // Wait for something to happen.
            if c::WaitOnAddress::is_available() {
                c::WaitOnAddress(self.ptr(), &PARKED as *const _ as c::LPVOID, 1, c::INFINITE);
            } else {
                c::NtWaitForKeyedEvent(keyed_event_handle(), self.ptr(), 0, ptr::null_mut());
            }
            // Change NOTIFIED=>EMPTY and return in that case.
            if self.state.compare_and_swap(NOTIFIED, EMPTY, Acquire) == NOTIFIED {
                return;
            } else {
                // Spurious wake up. We loop to try again.
            }
        }
    }

    // Assumes this is only called by the thread that owns the Parker,
    // which means that `self.state != PARKED`.
    pub unsafe fn park_timeout(&self, timeout: Duration) {
        // Change NOTIFIED=>EMPTY or EMPTY=>PARKED, and directly return in the
        // first case.
        if self.state.fetch_sub(1, Acquire) == NOTIFIED {
            return;
        }

        if c::WaitOnAddress::is_available() {
            // Wait for something to happen, assuming it's still set to PARKED.
            c::WaitOnAddress(self.ptr(), &PARKED as *const _ as c::LPVOID, 1, dur2timeout(timeout));
            // Change NOTIFIED=>EMPTY and return in that case.
            if self.state.swap(EMPTY, Acquire) == NOTIFIED {
                return;
            } else {
                // Timeout or spurious wake up.
                // We return either way, because we can't easily tell if it was the
                // timeout or not.
            }
        } else {
            // Need to wait for unpark() using NtWaitForKeyedEvent.
            let handle = keyed_event_handle();

            // NtWaitForKeyedEvent uses a unit of 100ns, and uses negative values
            // to indicate the monotonic clock.
            let mut timeout = match i64::try_from((timeout.as_nanos() + 99) / 100) {
                Ok(t) => -t,
                Err(_) => i64::MIN,
            };

            // Wait for unpark() to produce this event.
            if c::NtWaitForKeyedEvent(handle, self.ptr(), 0, &mut timeout) == c::STATUS_SUCCESS {
                // Awoken by another thread.
                self.state.swap(EMPTY, Acquire);
            } else {
                // Not awoken by another thread (spurious or timeout).
                if self.state.swap(EMPTY, Acquire) == NOTIFIED {
                    // If the state is NOTIFIED, we *just* missed an unpark(),
                    // which is now waiting for us to wait for it.
                    // Wait for it to consume the event and unblock it.
                    c::NtWaitForKeyedEvent(handle, self.ptr(), 0, ptr::null_mut());
                }
            }
        }
    }

    pub fn unpark(&self) {
        // Change PARKED=>NOTIFIED, EMPTY=>NOTIFIED, or NOTIFIED=>NOTIFIED, and
        // wake the thread in the first case.
        //
        // Note that even NOTIFIED=>NOTIFIED results in a write. This is on
        // purpose, to make sure every unpark() has a release-acquire ordering
        // with park().
        if self.state.swap(NOTIFIED, Release) == PARKED {
            if c::WakeByAddressSingle::is_available() {
                unsafe {
                    c::WakeByAddressSingle(self.ptr());
                }
            } else {
                // If we run NtReleaseKeyedEvent before the waiting thread runs
                // NtWaitForKeyedEvent, this (shortly) blocks until we can wake it up.
                // If the waiting thread wakes up before we run NtReleaseKeyedEvent
                // (e.g. due to a timeout), this blocks until we do wake up a thread.
                // To prevent this thread from blocking indefinitely in that case,
                // park_impl() will, after seeing the state set to NOTIFIED after
                // waking up, call NtWaitForKeyedEvent again to unblock us.
                unsafe {
                    c::NtReleaseKeyedEvent(keyed_event_handle(), self.ptr(), 0, ptr::null_mut());
                }
            }
        }
    }

    fn ptr(&self) -> c::LPVOID {
        &self.state as *const _ as c::LPVOID
    }
}

fn keyed_event_handle() -> c::HANDLE {
    const INVALID: usize = !0;
    static HANDLE: AtomicUsize = AtomicUsize::new(INVALID);
    match HANDLE.load(Relaxed) {
        INVALID => {
            let mut handle = c::INVALID_HANDLE_VALUE;
            unsafe {
                match c::NtCreateKeyedEvent(
                    &mut handle,
                    c::GENERIC_READ | c::GENERIC_WRITE,
                    ptr::null_mut(),
                    0,
                ) {
                    c::STATUS_SUCCESS => {}
                    r => panic!("Unable to create keyed event handle: error {}", r),
                }
            }
            match HANDLE.compare_exchange(INVALID, handle as usize, Relaxed, Relaxed) {
                Ok(_) => handle,
                Err(h) => {
                    // Lost the race to another thread initializing HANDLE before we did.
                    // Closing our handle and using theirs instead.
                    unsafe {
                        c::CloseHandle(handle);
                    }
                    h as c::HANDLE
                }
            }
        }
        handle => handle as c::HANDLE,
    }
}
