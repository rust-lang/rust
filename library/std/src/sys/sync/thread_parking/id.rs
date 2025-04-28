//! Thread parking using thread ids.
//!
//! Some platforms (notably NetBSD) have thread parking primitives whose semantics
//! match those offered by `thread::park`, with the difference that the thread to
//! be unparked is referenced by a platform-specific thread id. Since the thread
//! parker is constructed before that id is known, an atomic state variable is used
//! to manage the park state and propagate the thread id. This also avoids platform
//! calls in the case where `unpark` is called before `park`.

use crate::cell::UnsafeCell;
use crate::pin::Pin;
use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sync::atomic::{Atomic, AtomicI8, fence};
use crate::sys::thread_parking::{ThreadId, current, park, park_timeout, unpark};
use crate::time::Duration;

pub struct Parker {
    state: Atomic<i8>,
    tid: UnsafeCell<Option<ThreadId>>,
}

const PARKED: i8 = -1;
const EMPTY: i8 = 0;
const NOTIFIED: i8 = 1;

impl Parker {
    pub fn new() -> Parker {
        Parker { state: AtomicI8::new(EMPTY), tid: UnsafeCell::new(None) }
    }

    /// Creates a new thread parker. UNIX requires this to happen in-place.
    pub unsafe fn new_in_place(parker: *mut Parker) {
        parker.write(Parker::new())
    }

    /// # Safety
    /// * must always be called from the same thread
    /// * must be called before the state is set to PARKED
    unsafe fn init_tid(&self) {
        // The field is only ever written to from this thread, so we don't need
        // synchronization to read it here.
        if self.tid.get().read().is_none() {
            // Because this point is only reached once, before the state is set
            // to PARKED for the first time, the non-atomic write here can not
            // conflict with reads by other threads.
            self.tid.get().write(Some(current()));
            // Ensure that the write can be observed by all threads reading the
            // state. Synchronizes with the acquire barrier in `unpark`.
            fence(Release);
        }
    }

    pub unsafe fn park(self: Pin<&Self>) {
        self.init_tid();

        // Changes NOTIFIED to EMPTY and EMPTY to PARKED.
        let state = self.state.fetch_sub(1, Acquire);
        if state == EMPTY {
            // Loop to guard against spurious wakeups.
            // The state must be reset with acquire ordering to ensure that all
            // calls to `unpark` synchronize with this thread.
            while self.state.compare_exchange(NOTIFIED, EMPTY, Acquire, Relaxed).is_err() {
                park(self.state.as_ptr().addr());
            }
        }
    }

    pub unsafe fn park_timeout(self: Pin<&Self>, dur: Duration) {
        self.init_tid();

        let state = self.state.fetch_sub(1, Acquire).wrapping_sub(1);
        if state == PARKED {
            park_timeout(dur, self.state.as_ptr().addr());
            // Swap to ensure that we observe all state changes with acquire
            // ordering.
            self.state.swap(EMPTY, Acquire);
        }
    }

    pub fn unpark(self: Pin<&Self>) {
        let state = self.state.swap(NOTIFIED, Release);
        if state == PARKED {
            // Synchronize with the release fence in `init_tid` to observe the
            // write to `tid`.
            fence(Acquire);
            // # Safety
            // The thread id is initialized before the state is set to `PARKED`
            // for the first time and is not written to from that point on
            // (negating the need for an atomic read).
            let tid = unsafe { self.tid.get().read().unwrap_unchecked() };
            // It is possible that the waiting thread woke up because of a timeout
            // and terminated before this call is made. This call then returns an
            // error or wakes up an unrelated thread. The platform API and
            // environment does allow this, however.
            unpark(tid, self.state.as_ptr().addr());
        }
    }
}

unsafe impl Send for Parker {}
unsafe impl Sync for Parker {}
