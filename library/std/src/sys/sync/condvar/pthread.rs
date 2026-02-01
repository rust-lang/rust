#![forbid(unsafe_op_in_unsafe_fn)]

use crate::mem::DropGuard;
use crate::pin::Pin;
use crate::sys::pal::sync as pal;
use crate::sys::sync::{Mutex, OnceBox};
use crate::time::{Duration, Instant};

struct StateGuard<'a> {
    mutex: Pin<&'a pal::Mutex>,
}

impl<'a> Drop for StateGuard<'a> {
    fn drop(&mut self) {
        unsafe { self.mutex.unlock() };
    }
}

struct State {
    mutex: pal::Mutex,
    condvar: pal::Condvar,
}

impl State {
    fn condvar(self: Pin<&Self>) -> Pin<&pal::Condvar> {
        unsafe { self.map_unchecked(|this| &this.condvar) }
    }

    fn condvar_mut(self: Pin<&mut Self>) -> Pin<&mut pal::Condvar> {
        unsafe { self.map_unchecked_mut(|this| &mut this.condvar) }
    }

    /// Locks the `mutex` field and returns a [`StateGuard`] that unlocks the
    /// mutex when it is dropped.
    ///
    /// # Safety
    ///
    /// * The `mutex` field must not be locked by this thread.
    /// * Dismissing the guard leads to undefined behaviour when this `State`
    ///   is dropped, as it is undefined behaviour to destroy a locked mutex.
    unsafe fn lock(self: Pin<&Self>) -> StateGuard<'_> {
        let mutex = unsafe { self.map_unchecked(|this| &this.mutex) };
        unsafe { mutex.lock() };
        StateGuard { mutex }
    }
}

pub struct Condvar {
    state: OnceBox<State>,
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { state: OnceBox::new() }
    }

    #[inline]
    fn state(&self) -> Pin<&State> {
        self.state.get_or_init(|| {
            let mut state =
                Box::pin(State { mutex: pal::Mutex::new(), condvar: pal::Condvar::new() });

            // SAFETY: we only call `init` once per `pal::Condvar`, namely here.
            unsafe { state.as_mut().condvar_mut().init() };
            state
        })
    }

    pub fn notify_one(&self) {
        let state = self.state();
        // Notifications might be sent right after a mutex used with `wait` or
        // `wait_timeout` is unlocked. Waiting until the state mutex is
        // available ensures that the thread unlocking the mutex is enqueued
        // on the inner condition variable, as the mutex is only unlocked
        // with the state mutex held.
        //
        // Releasing the state mutex before issuing the notification stops
        // the awakened threads from having to wait on this thread unlocking
        // the mutex.
        //
        // SAFETY:
        // The functions in this module are never called recursively, so the
        // state mutex cannot be currently locked by this thread.
        drop(unsafe { state.lock() });
        // SAFETY: we called `init` above.
        unsafe { state.condvar().notify_one() }
    }

    pub fn notify_all(&self) {
        let state = self.state();
        // Notifications might be sent right after a mutex used with `wait` or
        // `wait_timeout` is unlocked. Waiting until the state mutex is
        // available ensures that the thread unlocking the mutex is enqueued
        // on the inner condition variable, as the mutex is only unlocked
        // with the state mutex held.
        //
        // Releasing the state mutex before issuing the notification stops
        // the awakened threads from having to wait on this thread unlocking
        // the mutex.
        //
        // SAFETY:
        // The functions in this module are never called recursively, so the
        // state mutex cannot be currently locked by this thread.
        drop(unsafe { state.lock() });
        // SAFETY: we called `init` above.
        unsafe { state.condvar().notify_all() }
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let state = self.state();

        // Ensure that the mutex is locked when this function returns or panics.
        // The relocking must occur after the state lock is unlocked to prevent
        // deadlocks, hence we scope the relock guard before the state lock guard.
        let relock;

        // Lock the state mutex before unlocking `mutex` to ensure that
        // notifications occurring before this thread is enqueued on the
        // condvar are not missed.
        //
        // SAFETY:
        // The functions in this module are never called recursively, so the
        // state mutex cannot be currently locked by this thread.
        let guard = unsafe { state.lock() };

        // SAFETY:
        // The caller must guarantee that `mutex` is currently locked by this
        // thread.
        unsafe { mutex.unlock() };
        relock = DropGuard::new(mutex, |mutex| mutex.lock());

        // SAFETY:
        // * `init` was called above
        // * the condition variable is only ever used with the state mutex
        // * the state mutex was locked above
        unsafe { state.condvar().wait(guard.mutex) };
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let state = self.state();

        // Ensure that the mutex is locked when this function returns or panics.
        // The relocking must occur after the state lock is unlocked to prevent
        // deadlocks, hence we scope the relock guard before the state lock guard.
        let relock;

        // Lock the state mutex before unlocking `mutex` to ensure that
        // notifications occurring before this thread is enqueued on the
        // condvar are not missed.
        //
        // SAFETY:
        // The functions in this module are never called recursively, so the
        // state mutex cannot be currently locked by this thread.
        let guard = unsafe { state.lock() };

        // SAFETY:
        // The caller must guarantee that `mutex` is currently locked by this
        // thread.
        unsafe { mutex.unlock() };
        relock = DropGuard::new(mutex, |mutex| mutex.lock());

        if pal::Condvar::PRECISE_TIMEOUT {
            // SAFETY:
            // * `init` was called above
            // * the condition variable is only ever used with the state mutex
            // * the state mutex was locked above
            unsafe { state.condvar().wait_timeout(guard.mutex, dur) }
        } else {
            // Timeout reports are not reliable, so do the check ourselves.
            let now = Instant::now();
            // SAFETY:
            // * `init` was called above
            // * the condition variable is only ever used with the state mutex
            // * the state mutex was locked above
            let woken = unsafe { state.condvar().wait_timeout(guard.mutex, dur) };
            woken || now.elapsed() < dur
        }
    }
}
