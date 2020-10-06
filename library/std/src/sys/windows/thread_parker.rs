// Thread parker implementation for Windows.
//
// This uses WaitOnAddress and WakeByAddressSingle if available (Windows 8+).
// This modern API is exactly the same as the futex syscalls the Linux thread
// parker uses. When These APIs are available, the implementation of this
// thread parker matches the Linux thread parker exactly.
//
// However, when the modern API is not available, this implementation falls
// back to NT Keyed Events, which are similar, but have some important
// differences. These are available since Windows XP.
//
// WaitOnAddress first checks the state of the thread parker to make sure it no
// WakeByAddressSingle calls can be missed between updating the parker state
// and calling the function.
//
// NtWaitForKeyedEvent does not have this option, and unconditionally blocks
// without checking the parker state first. Instead, NtReleaseKeyedEvent
// (unlike WakeByAddressSingle) *blocks* until it woke up a thread waiting for
// it by NtWaitForKeyedEvent. This way, we can be sure no events are missed,
// but we need to be careful not to block unpark() if park_timeout() was woken
// up by a timeout instead of unpark().
//
// Unlike WaitOnAddress, NtWaitForKeyedEvent/NtReleaseKeyedEvent operate on a
// HANDLE (created with NtCreateKeyedEvent). This means that we can be sure
// a succesfully awoken park() was awoken by unpark() and not a
// NtReleaseKeyedEvent call from some other code, as these events are not only
// matched by the key (address of the parker (state)), but also by this HANDLE.
// We lazily allocate this handle the first time it is needed.
//
// The fast path (calling park() after unpark() was already called) and the
// possible states are the same for both implementations. This is used here to
// make sure the fast path does not even check which API to use, but can return
// right away, independent of the used API. Only the slow paths (which will
// actually block/wake a thread) check which API is available and have
// different implementations.
//
// Unfortunately, NT Keyed Events are an undocumented Windows API. However:
// - This API is relatively simple with obvious behaviour, and there are
//   several (unofficial) articles documenting the details. [1]
// - `parking_lot` has been using this API for years (on Windows versions
//   before Windows 8). [2] Many big projects extensively use parking_lot,
//   such as servo and the Rust compiler itself.
// - It is the underlying API used by Windows SRW locks and Windows critical
//   sections. [3] [4]
// - The source code of the implementations of Wine, ReactOs, and Windows XP
//   are available and match the expected behaviour.
// - The main risk with an undocumented API is that it might change in the
//   future. But since we only use it for older versions of Windows, that's not
//   a problem.
// - Even if these functions do not block or wake as we expect (which is
//   unlikely, see all previous points), this implementation would still be
//   memory safe. The NT Keyed Events API is only used to sleep/block in the
//   right place.
//
// [1]: http://www.locklessinc.com/articles/keyed_events/
// [2]: https://github.com/Amanieu/parking_lot/commit/43abbc964e
// [3]: https://docs.microsoft.com/en-us/archive/msdn-magazine/2012/november/windows-with-c-the-evolution-of-synchronization-in-windows-and-c
// [4]: Windows Internals, Part 1, ISBN 9780735671300

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
// Ordering::Acquire when reading this state in park() after waking up.
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

        if c::WaitOnAddress::is_available() {
            loop {
                // Wait for something to happen, assuming it's still set to PARKED.
                c::WaitOnAddress(self.ptr(), &PARKED as *const _ as c::LPVOID, 1, c::INFINITE);
                // Change NOTIFIED=>EMPTY but leave PARKED alone.
                if self.state.compare_and_swap(NOTIFIED, EMPTY, Acquire) == NOTIFIED {
                    // Actually woken up by unpark().
                    return;
                } else {
                    // Spurious wake up. We loop to try again.
                }
            }
        } else {
            // Wait for unpark() to produce this event.
            c::NtWaitForKeyedEvent(keyed_event_handle(), self.ptr(), 0, ptr::null_mut());
            // Set the state back to EMPTY (from either PARKED or NOTIFIED).
            // Note that we don't just write EMPTY, but use swap() to also
            // include an acquire-ordered read to synchronize with unpark()'s
            // release-ordered write.
            self.state.swap(EMPTY, Acquire);
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
            // Set the state back to EMPTY (from either PARKED or NOTIFIED).
            // Note that we don't just write EMPTY, but use swap() to also
            // include an acquire-ordered read to synchronize with unpark()'s
            // release-ordered write.
            if self.state.swap(EMPTY, Acquire) == NOTIFIED {
                // Actually woken up by unpark().
            } else {
                // Timeout or spurious wake up.
                // We return either way, because we can't easily tell if it was the
                // timeout or not.
            }
        } else {
            // Need to wait for unpark() using NtWaitForKeyedEvent.
            let handle = keyed_event_handle();

            // NtWaitForKeyedEvent uses a unit of 100ns, and uses negative
            // values to indicate a relative time on the monotonic clock.
            // This is documented here for the underlying KeWaitForSingleObject function:
            // https://docs.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/nf-wdm-kewaitforsingleobject
            let mut timeout = match i64::try_from((timeout.as_nanos() + 99) / 100) {
                Ok(t) => -t,
                Err(_) => i64::MIN,
            };

            // Wait for unpark() to produce this event.
            let unparked =
                c::NtWaitForKeyedEvent(handle, self.ptr(), 0, &mut timeout) == c::STATUS_SUCCESS;

            // Set the state back to EMPTY (from either PARKED or NOTIFIED).
            let prev_state = self.state.swap(EMPTY, Acquire);

            if !unparked && prev_state == NOTIFIED {
                // We were awoken by a timeout, not by unpark(), but the state
                // was set to NOTIFIED, which means we *just* missed an
                // unpark(), which is now blocked on us to wait for it.
                // Wait for it to consume the event and unblock that thread.
                c::NtWaitForKeyedEvent(handle, self.ptr(), 0, ptr::null_mut());
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
