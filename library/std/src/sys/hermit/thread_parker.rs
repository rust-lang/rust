// Hermit has efficient thread parking primitives in the form of the `block_current_task`/
// `wakeup_task` syscalls. `block_current_task` marks the current thread as blocked, which
// means the scheduler will not try to reschedule the task once it is switched away from.
// `wakeup_task` undoes this. Since Hermit is not pre-emptive, this means programs get to
// run code in between calling `block_current_task` and actually waiting (with a call to
// `yield_now`) without encountering any race conditions.
//
// The thread parker uses an atomic variable which is set one of three states:
// * EMPTY: the token has not been made available
// * NOTIFIED: the token is available
// * some pid: the thread with the specified PID is waiting or about to be waiting for
//   the token
// Since `wakeup_task` requires the task to be actually waiting, the state needs to
// be checked in between preparing to park and actually parking to avoid deadlocks.
// If the state has changed, the thread resets its own state by calling `wakeup_task`.

use super::abi;
use crate::pin::Pin;
use crate::sync::atomic::AtomicU64;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::time::{Duration, Instant};

// These values are larger than u32::MAX, so they never conflict with the thread's `pid`.
const EMPTY: u64 = 0x1_0000_0000;
const NOTIFIED: u64 = 0x2_0000_0000;

// Note about atomic memory orderings:
// Acquire ordering is necessary to obtain the token, as otherwise the parked thread
// could perform memory operations visible before it was unparked. Since once set,
// the token cannot be unset by other threads, the state can be reset with a relaxed
// store once it has been read with acquire ordering.
pub struct Parker {
    state: AtomicU64,
}

impl Parker {
    /// Construct the thread parker. The UNIX parker implementation
    /// requires this to happen in-place.
    pub unsafe fn new(parker: *mut Parker) {
        parker.write(Parker { state: AtomicU64::new(EMPTY) })
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park(self: Pin<&Self>) {
        if self.state.load(Acquire) == NOTIFIED {
            self.state.store(EMPTY, Relaxed);
            return;
        }

        let pid = abi::getpid();
        abi::block_current_task();
        if self.state.compare_exchange(EMPTY, pid as u64, Acquire, Acquire).is_ok() {
            // Loop to avoid spurious wakeups.
            loop {
                abi::yield_now();

                if self.state.load(Acquire) == NOTIFIED {
                    break;
                }

                abi::block_current_task();

                if self.state.load(Acquire) == NOTIFIED {
                    abi::wakeup_task(pid);
                    break;
                }
            }
        } else {
            abi::wakeup_task(pid);
        }

        self.state.store(EMPTY, Relaxed);
    }

    // This implementation doesn't require `unsafe` and `Pin`, but other implementations do.
    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        if self.state.load(Acquire) == NOTIFIED {
            self.state.store(EMPTY, Relaxed);
            return;
        }

        if dur < Duration::from_millis(1) {
            // Spin on the token for sub-millisecond parking.
            let now = Instant::now();
            let Some(deadline) = now.checked_add(dur) else { return; };
            loop {
                abi::yield_now();

                if self.state.load(Acquire) == NOTIFIED {
                    self.state.store(EMPTY, Relaxed);
                    return;
                } else if Instant::now() >= deadline {
                    // Swap to provide acquire ordering even if the timeout occurred
                    // before the token was set.
                    self.state.swap(EMPTY, Acquire);
                    return;
                }
            }
        } else {
            let timeout = dur.as_millis().try_into().unwrap_or(u64::MAX);
            let pid = abi::getpid();
            abi::block_current_task_with_timeout(timeout);
            if self.state.compare_exchange(EMPTY, pid as u64, Acquire, Acquire).is_ok() {
                abi::yield_now();

                // Swap to provide acquire ordering even if the timeout occurred
                // before the token was set. This situation can result in spurious
                // wakeups on the next call to `park_timeout`, but it is better to let
                // those be handled by the user rather than to do some perhaps unnecessary,
                // but always expensive guarding.
                self.state.swap(EMPTY, Acquire);
            } else {
                abi::wakeup_task(pid);
                self.state.store(EMPTY, Relaxed);
            }
        }
    }

    // This implementation doesn't require `Pin`, but other implementations do.
    pub fn unpark(self: Pin<&Self>) {
        // Use release ordering to synchonize with the parked thread.
        let state = self.state.swap(NOTIFIED, Release);

        if !matches!(state, EMPTY | NOTIFIED) {
            // SAFETY: `wakeup_task` is marked unsafe, but is actually safe to use
            unsafe {
                let pid = state as u32;
                // This is a noop if the task is not blocked or has terminated, but
                // that is fine.
                abi::wakeup_task(pid);
            }
        }
    }
}
