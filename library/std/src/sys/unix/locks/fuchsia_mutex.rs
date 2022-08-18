//! A priority inheriting mutex for Fuchsia.
//!
//! This is a port of the [mutex in Fuchsia's libsync]. Contrary to the original,
//! it does not abort the process when reentrant locking is detected, but deadlocks.
//!
//! Priority inheritance is achieved by storing the owning thread's handle in an
//! atomic variable. Fuchsia's futex operations support setting an owner thread
//! for a futex, which can boost that thread's priority while the futex is waited
//! upon.
//!
//! libsync is licenced under the following BSD-style licence:
//!
//! Copyright 2016 The Fuchsia Authors.
//!
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are
//! met:
//!
//!    * Redistributions of source code must retain the above copyright
//!      notice, this list of conditions and the following disclaimer.
//!    * Redistributions in binary form must reproduce the above
//!      copyright notice, this list of conditions and the following
//!      disclaimer in the documentation and/or other materials provided
//!      with the distribution.
//!
//! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//! A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//! OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//! SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//! LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//! DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//! THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//!
//! [mutex in Fuchsia's libsync]: https://cs.opensource.google/fuchsia/fuchsia/+/main:zircon/system/ulib/sync/mutex.c

use crate::sync::atomic::{
    AtomicU32,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::sys::futex::zircon::{
    zx_futex_wait, zx_futex_wake_single_owner, zx_handle_t, zx_thread_self, ZX_ERR_BAD_HANDLE,
    ZX_ERR_BAD_STATE, ZX_ERR_INVALID_ARGS, ZX_ERR_TIMED_OUT, ZX_ERR_WRONG_TYPE, ZX_OK,
    ZX_TIME_INFINITE,
};

// The lowest two bits of a `zx_handle_t` are always set, so the lowest bit is used to mark the
// mutex as contested by clearing it.
const CONTESTED_BIT: u32 = 1;
// This can never be a valid `zx_handle_t`.
const UNLOCKED: u32 = 0;

pub type MovableMutex = Mutex;

pub struct Mutex {
    futex: AtomicU32,
}

#[inline]
fn to_state(owner: zx_handle_t) -> u32 {
    owner
}

#[inline]
fn to_owner(state: u32) -> zx_handle_t {
    state | CONTESTED_BIT
}

#[inline]
fn is_contested(state: u32) -> bool {
    state & CONTESTED_BIT == 0
}

#[inline]
fn mark_contested(state: u32) -> u32 {
    state & !CONTESTED_BIT
}

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { futex: AtomicU32::new(UNLOCKED) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {}

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let thread_self = zx_thread_self();
        self.futex.compare_exchange(UNLOCKED, to_state(thread_self), Acquire, Relaxed).is_ok()
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let thread_self = zx_thread_self();
        if let Err(state) =
            self.futex.compare_exchange(UNLOCKED, to_state(thread_self), Acquire, Relaxed)
        {
            self.lock_contested(state, thread_self);
        }
    }

    #[cold]
    fn lock_contested(&self, mut state: u32, thread_self: zx_handle_t) {
        let owned_state = mark_contested(to_state(thread_self));
        loop {
            // Mark the mutex as contested if it is not already.
            let contested = mark_contested(state);
            if is_contested(state)
                || self.futex.compare_exchange(state, contested, Relaxed, Relaxed).is_ok()
            {
                // The mutex has been marked as contested, wait for the state to change.
                unsafe {
                    match zx_futex_wait(
                        &self.futex,
                        AtomicU32::new(contested),
                        to_owner(state),
                        ZX_TIME_INFINITE,
                    ) {
                        ZX_OK | ZX_ERR_BAD_STATE | ZX_ERR_TIMED_OUT => (),
                        // Note that if a thread handle is reused after its associated thread
                        // exits without unlocking the mutex, an arbitrary thread's priority
                        // could be boosted by the wait, but there is currently no way to
                        // prevent that.
                        ZX_ERR_INVALID_ARGS | ZX_ERR_BAD_HANDLE | ZX_ERR_WRONG_TYPE => {
                            panic!(
                                "either the current thread is trying to lock a mutex it has
                                already locked, or the previous owner did not unlock the mutex
                                before exiting"
                            )
                        }
                        error => panic!("unexpected error in zx_futex_wait: {error}"),
                    }
                }
            }

            // The state has changed or a wakeup occurred, try to lock the mutex.
            match self.futex.compare_exchange(UNLOCKED, owned_state, Acquire, Relaxed) {
                Ok(_) => return,
                Err(updated) => state = updated,
            }
        }
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        if is_contested(self.futex.swap(UNLOCKED, Release)) {
            // The woken thread will mark the mutex as contested again,
            // and return here, waking until there are no waiters left,
            // in which case this is a noop.
            self.wake();
        }
    }

    #[cold]
    fn wake(&self) {
        unsafe {
            zx_futex_wake_single_owner(&self.futex);
        }
    }
}
