use crate::os::xous::ffi::{blocking_scalar, scalar};
use crate::os::xous::services::{TicktimerScalar, ticktimer_server};
use crate::pin::Pin;
use crate::ptr;
use crate::sync::atomic::AtomicI8;
use crate::sync::atomic::Ordering::{Acquire, Release};
use crate::time::Duration;

const NOTIFIED: i8 = 1;
const EMPTY: i8 = 0;
const PARKED: i8 = -1;

pub struct Parker {
    state: AtomicI8,
}

impl Parker {
    pub unsafe fn new_in_place(parker: *mut Parker) {
        unsafe { parker.write(Parker { state: AtomicI8::new(EMPTY) }) }
    }

    fn index(&self) -> usize {
        ptr::from_ref(self).addr()
    }

    pub unsafe fn park(self: Pin<&Self>) {
        // Change NOTIFIED to EMPTY and EMPTY to PARKED.
        let state = self.state.fetch_sub(1, Acquire);
        if state == NOTIFIED {
            // The state has gone from NOTIFIED (1) to EMPTY (0)
            return;
        }
        // The state has gone from EMPTY (0) to PARKED (-1)
        assert!(state == EMPTY);

        // The state is now PARKED (-1). Wait until the `unpark` wakes us up.
        blocking_scalar(
            ticktimer_server(),
            TicktimerScalar::WaitForCondition(self.index(), 0).into(),
        )
        .expect("failed to send WaitForCondition command");

        let state = self.state.swap(EMPTY, Acquire);
        assert!(state == NOTIFIED || state == PARKED);
    }

    pub unsafe fn park_timeout(self: Pin<&Self>, timeout: Duration) {
        // Change NOTIFIED to EMPTY and EMPTY to PARKED.
        let state = self.state.fetch_sub(1, Acquire);
        if state == NOTIFIED {
            // The state has gone from NOTIFIED (1) to EMPTY (0)
            return;
        }
        // The state has gone from EMPTY (0) to PARKED (-1)
        assert!(state == EMPTY);

        // A value of zero indicates an indefinite wait. Clamp the number of
        // milliseconds to the allowed range.
        let millis = usize::max(timeout.as_millis().try_into().unwrap_or(usize::MAX), 1);

        // The state is now PARKED (-1). Wait until the `unpark` wakes us up,
        // or things time out.
        let _was_timeout = blocking_scalar(
            ticktimer_server(),
            TicktimerScalar::WaitForCondition(self.index(), millis).into(),
        )
        .expect("failed to send WaitForCondition command")[0]
            != 0;

        let state = self.state.swap(EMPTY, Acquire);
        assert!(state == PARKED || state == NOTIFIED);
    }

    pub fn unpark(self: Pin<&Self>) {
        // If the state is already `NOTIFIED`, then another thread has
        // indicated it wants to wake up the target thread.
        //
        // If the state is `EMPTY` then there is nothing to wake up, and
        // the target thread will immediately exit from `park()` the
        // next time that function is called.
        if self.state.swap(NOTIFIED, Release) != PARKED {
            return;
        }

        // The thread is parked, wake it up. Keep trying until we wake something up.
        // This will happen when the `NotifyCondition` call returns the fact that
        // 1 condition was notified.
        // Alternately, keep going until the state is seen as `EMPTY`, indicating
        // the thread woke up and kept going. This can happen when the Park
        // times out before we can send the NotifyCondition message.
        while blocking_scalar(
            ticktimer_server(),
            TicktimerScalar::NotifyCondition(self.index(), 1).into(),
        )
        .expect("failed to send NotifyCondition command")[0]
            == 0
            && self.state.load(Acquire) != EMPTY
        {
            // The target thread hasn't yet hit the `WaitForCondition` call.
            // Yield to let the target thread run some more.
            crate::thread::yield_now();
        }
    }
}

impl Drop for Parker {
    fn drop(&mut self) {
        scalar(ticktimer_server(), TicktimerScalar::FreeCondition(self.index()).into()).ok();
    }
}
