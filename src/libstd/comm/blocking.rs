// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generic support for building blocking abstractions.

use thread::Thread;
use sync::atomic::{AtomicBool, INIT_ATOMIC_BOOL, Ordering};
use sync::Arc;
use kinds::{Sync, Send};
use kinds::marker::{NoSend, NoSync};
use mem;
use clone::Clone;

struct Inner {
    thread: Thread,
    woken: AtomicBool,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

#[deriving(Clone)]
pub struct SignalToken {
    inner: Arc<Inner>,
}

pub struct WaitToken {
    inner: Arc<Inner>,
    no_send: NoSend,
    no_sync: NoSync,
}

pub fn tokens() -> (WaitToken, SignalToken) {
    let inner = Arc::new(Inner {
        thread: Thread::current(),
        woken: INIT_ATOMIC_BOOL,
    });
    let wait_token = WaitToken {
        inner: inner.clone(),
        no_send: NoSend,
        no_sync: NoSync,
    };
    let signal_token = SignalToken {
        inner: inner
    };
    (wait_token, signal_token)
}

impl SignalToken {
    pub fn signal(&self) -> bool {
        let wake = !self.inner.woken.compare_and_swap(false, true, Ordering::SeqCst);
        if wake {
            self.inner.thread.unpark();
        }
        wake
    }

    /// Convert to an unsafe uint value. Useful for storing in a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_to_uint(self) -> uint {
        mem::transmute(self.inner)
    }

    /// Convert from an unsafe uint value. Useful for retrieving a pipe's state
    /// flag.
    #[inline]
    pub unsafe fn cast_from_uint(signal_ptr: uint) -> SignalToken {
        SignalToken { inner: mem::transmute(signal_ptr) }
    }

}

impl WaitToken {
    pub fn wait(self) {
        while !self.inner.woken.load(Ordering::SeqCst) {
            Thread::park()
        }
    }
}
