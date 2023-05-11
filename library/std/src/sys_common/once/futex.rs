use crate::cell::Cell;
use crate::sync as public;
use crate::sync::atomic::{
    AtomicU32,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::sync::once::ExclusiveState;
use crate::sys::futex::{futex_wait, futex_wake_all};

// On some platforms, the OS is very nice and handles the waiter queue for us.
// This means we only need one atomic value with 5 states:

/// No initialization has run yet, and no thread is currently using the Once.
const INCOMPLETE: u32 = 0;
/// Some thread has previously attempted to initialize the Once, but it panicked,
/// so the Once is now poisoned. There are no other threads currently accessing
/// this Once.
const POISONED: u32 = 1;
/// Some thread is currently attempting to run initialization. It may succeed,
/// so all future threads need to wait for it to finish.
const RUNNING: u32 = 2;
/// Some thread is currently attempting to run initialization and there are threads
/// waiting for it to finish.
const QUEUED: u32 = 3;
/// Initialization has completed and all future calls should finish immediately.
const COMPLETE: u32 = 4;

// Threads wait by setting the state to QUEUED and calling `futex_wait` on the state
// variable. When the running thread finishes, it will wake all waiting threads using
// `futex_wake_all`.

pub struct OnceState {
    poisoned: bool,
    set_state_to: Cell<u32>,
}

impl OnceState {
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[inline]
    pub fn poison(&self) {
        self.set_state_to.set(POISONED);
    }
}

struct CompletionGuard<'a> {
    state: &'a AtomicU32,
    set_state_on_drop_to: u32,
}

impl<'a> Drop for CompletionGuard<'a> {
    fn drop(&mut self) {
        // Use release ordering to propagate changes to all threads checking
        // up on the Once. `futex_wake_all` does its own synchronization, hence
        // we do not need `AcqRel`.
        if self.state.swap(self.set_state_on_drop_to, Release) == QUEUED {
            futex_wake_all(&self.state);
        }
    }
}

pub struct Once {
    state: AtomicU32,
}

impl Once {
    #[inline]
    pub const fn new() -> Once {
        Once { state: AtomicU32::new(INCOMPLETE) }
    }

    #[inline]
    pub fn is_completed(&self) -> bool {
        // Use acquire ordering to make all initialization changes visible to the
        // current thread.
        self.state.load(Acquire) == COMPLETE
    }

    #[inline]
    pub(crate) fn state(&mut self) -> ExclusiveState {
        match *self.state.get_mut() {
            INCOMPLETE => ExclusiveState::Incomplete,
            POISONED => ExclusiveState::Poisoned,
            COMPLETE => ExclusiveState::Complete,
            _ => unreachable!("invalid Once state"),
        }
    }

    // This uses FnMut to match the API of the generic implementation. As this
    // implementation is quite light-weight, it is generic over the closure and
    // so avoids the cost of dynamic dispatch.
    #[cold]
    #[track_caller]
    pub fn call(&self, ignore_poisoning: bool, f: &mut impl FnMut(&public::OnceState)) {
        let mut state = self.state.load(Acquire);
        loop {
            match state {
                POISONED if !ignore_poisoning => {
                    // Panic to propagate the poison.
                    panic!("Once instance has previously been poisoned");
                }
                INCOMPLETE | POISONED => {
                    // Try to register the current thread as the one running.
                    if let Err(new) =
                        self.state.compare_exchange_weak(state, RUNNING, Acquire, Acquire)
                    {
                        state = new;
                        continue;
                    }
                    // `waiter_queue` will manage other waiting threads, and
                    // wake them up on drop.
                    let mut waiter_queue =
                        CompletionGuard { state: &self.state, set_state_on_drop_to: POISONED };
                    // Run the function, letting it know if we're poisoned or not.
                    let f_state = public::OnceState {
                        inner: OnceState {
                            poisoned: state == POISONED,
                            set_state_to: Cell::new(COMPLETE),
                        },
                    };
                    f(&f_state);
                    waiter_queue.set_state_on_drop_to = f_state.inner.set_state_to.get();
                    return;
                }
                RUNNING | QUEUED => {
                    // Set the state to QUEUED if it is not already.
                    if state == RUNNING
                        && let Err(new) = self.state.compare_exchange_weak(RUNNING, QUEUED, Relaxed, Acquire)
                    {
                        state = new;
                        continue;
                    }

                    futex_wait(&self.state, QUEUED, None);
                    state = self.state.load(Acquire);
                }
                COMPLETE => return,
                _ => unreachable!("state is never set to invalid values"),
            }
        }
    }
}
