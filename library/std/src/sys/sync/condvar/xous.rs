use core::sync::atomic::{Atomic, AtomicUsize, Ordering};

use crate::os::xous::ffi::{blocking_scalar, scalar};
use crate::os::xous::services::{TicktimerScalar, ticktimer_server};
use crate::sys::sync::Mutex;
use crate::time::Duration;

// The implementation is inspired by Andrew D. Birrell's paper
// "Implementing Condition Variables with Semaphores"

const NOTIFY_TRIES: usize = 3;

pub struct Condvar {
    counter: Atomic<usize>,
    timed_out: Atomic<usize>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar { counter: AtomicUsize::new(0), timed_out: AtomicUsize::new(0) }
    }

    fn notify_some(&self, to_notify: usize) {
        // Assumption: The Mutex protecting this condvar is locked throughout the
        // entirety of this call, preventing calls to `wait` and `wait_timeout`.

        // Logic check: Ensure that there aren't any missing waiters. Remove any that
        // timed-out, ensuring the counter doesn't underflow.
        assert!(self.timed_out.load(Ordering::Relaxed) <= self.counter.load(Ordering::Relaxed));
        self.counter.fetch_sub(self.timed_out.swap(0, Ordering::Relaxed), Ordering::Relaxed);

        // Figure out how many threads to notify. Note that it is impossible for `counter`
        // to increase during this operation because Mutex is locked. However, it is
        // possible for `counter` to decrease due to a condvar timing out, in which
        // case the corresponding `timed_out` will increase accordingly.
        let Ok(waiter_count) =
            self.counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |counter| {
                if counter == 0 {
                    return None;
                } else {
                    Some(counter - counter.min(to_notify))
                }
            })
        else {
            // No threads are waiting on this condvar
            return;
        };

        let mut remaining_to_wake = waiter_count.min(to_notify);
        if remaining_to_wake == 0 {
            return;
        }
        for _wake_tries in 0..NOTIFY_TRIES {
            let result = blocking_scalar(
                ticktimer_server(),
                TicktimerScalar::NotifyCondition(self.index(), remaining_to_wake).into(),
            )
            .expect("failure to send NotifyCondition command");

            // Remove the list of waiters that were notified
            remaining_to_wake -= result[0];

            // Also remove the number of waiters that timed out. Clamp it to 0 in order to
            // ensure we don't wait forever in case the waiter woke up between the time
            // we counted the remaining waiters and now.
            remaining_to_wake =
                remaining_to_wake.saturating_sub(self.timed_out.swap(0, Ordering::Relaxed));
            if remaining_to_wake == 0 {
                return;
            }
            crate::thread::yield_now();
        }
    }

    pub fn notify_one(&self) {
        self.notify_some(1)
    }

    pub fn notify_all(&self) {
        self.notify_some(self.counter.load(Ordering::Relaxed))
    }

    fn index(&self) -> usize {
        core::ptr::from_ref(self).addr()
    }

    /// Unlock the given Mutex and wait for the notification. Wait at most
    /// `ms` milliseconds, or pass `0` to wait forever.
    ///
    /// Returns `true` if the condition was received, `false` if it timed out
    fn wait_ms(&self, mutex: &Mutex, ms: usize) -> bool {
        self.counter.fetch_add(1, Ordering::Relaxed);
        unsafe { mutex.unlock() };

        // Threading concern: There is a chance that the `notify` thread wakes up here before
        // we have a chance to wait for the condition. This is fine because we've recorded
        // the fact that we're waiting by incrementing the counter.
        let result = blocking_scalar(
            ticktimer_server(),
            TicktimerScalar::WaitForCondition(self.index(), ms).into(),
        );
        let awoken = result.expect("Ticktimer: failure to send WaitForCondition command")[0] == 0;

        // If we awoke due to a timeout, increment the `timed_out` counter so that the
        // main loop of `notify` knows there's a timeout.
        //
        // This is done with the Mutex still unlocked, because the Mutex might still
        // be locked by the `notify` process above.
        if !awoken {
            self.timed_out.fetch_add(1, Ordering::Relaxed);
        }

        unsafe { mutex.lock() };
        awoken
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        // Wait for 0 ms, which is a special case to "wait forever"
        self.wait_ms(mutex, 0);
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let mut millis = dur.as_millis() as usize;
        // Ensure we don't wait for 0 ms, which would cause us to wait forever
        if millis == 0 {
            millis = 1;
        }
        self.wait_ms(mutex, millis)
    }
}

impl Drop for Condvar {
    fn drop(&mut self) {
        let remaining_count = self.counter.load(Ordering::Relaxed);
        let timed_out = self.timed_out.load(Ordering::Relaxed);
        assert!(
            remaining_count - timed_out == 0,
            "counter was {} and timed_out was {} not 0",
            remaining_count,
            timed_out
        );
        scalar(ticktimer_server(), TicktimerScalar::FreeCondition(self.index()).into()).ok();
    }
}
