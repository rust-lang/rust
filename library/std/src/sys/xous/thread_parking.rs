use crate::os::xous::ffi::{blocking_scalar, scalar};
use crate::os::xous::services::{ticktimer_server, TicktimerScalar};
use crate::pin::Pin;
use crate::ptr;
use crate::sync::atomic::{
    AtomicI8,
    Ordering::{Acquire, Release},
};
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
            return;
        }

        // The state was set to PARKED. Wait until the `unpark` wakes us up.
        blocking_scalar(
            ticktimer_server(),
            TicktimerScalar::WaitForCondition(self.index(), 0).into(),
        )
        .expect("failed to send WaitForCondition command");

        self.state.swap(EMPTY, Acquire);
    }

    pub unsafe fn park_timeout(self: Pin<&Self>, timeout: Duration) {
        // Change NOTIFIED to EMPTY and EMPTY to PARKED.
        let state = self.state.fetch_sub(1, Acquire);
        if state == NOTIFIED {
            return;
        }

        // A value of zero indicates an indefinite wait. Clamp the number of
        // milliseconds to the allowed range.
        let millis = usize::max(timeout.as_millis().try_into().unwrap_or(usize::MAX), 1);

        let was_timeout = blocking_scalar(
            ticktimer_server(),
            TicktimerScalar::WaitForCondition(self.index(), millis).into(),
        )
        .expect("failed to send WaitForCondition command")[0]
            != 0;

        let state = self.state.swap(EMPTY, Acquire);
        if was_timeout && state == NOTIFIED {
            // The state was set to NOTIFIED after we returned from the wait
            // but before we reset the state. Therefore, a wakeup is on its
            // way, which we need to consume here.
            // NOTICE: this is a priority hole.
            blocking_scalar(
                ticktimer_server(),
                TicktimerScalar::WaitForCondition(self.index(), 0).into(),
            )
            .expect("failed to send WaitForCondition command");
        }
    }

    pub fn unpark(self: Pin<&Self>) {
        let state = self.state.swap(NOTIFIED, Release);
        if state == PARKED {
            // The thread is parked, wake it up.
            blocking_scalar(
                ticktimer_server(),
                TicktimerScalar::NotifyCondition(self.index(), 1).into(),
            )
            .expect("failed to send NotifyCondition command");
        }
    }
}

impl Drop for Parker {
    fn drop(&mut self) {
        scalar(ticktimer_server(), TicktimerScalar::FreeCondition(self.index()).into()).ok();
    }
}
