// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the helper thread for the timer module
//!
//! This module contains the management necessary for the timer worker thread.
//! This thread is responsible for performing the send()s on channels for timers
//! that are using channels instead of a blocking call.
//!
//! The timer thread is lazily initialized, and it's shut down via the
//! `shutdown` function provided. It must be maintained as an invariant that
//! `shutdown` is only called when the entire program is finished. No new timers
//! can be created in the future and there must be no active timers at that
//! time.

use std::cast;
use std::rt;
use std::unstable::mutex::{Once, ONCE_INIT};

use bookkeeping;
use io::timer::{Req, Shutdown};
use task;

// You'll note that these variables are *not* protected by a lock. These
// variables are initialized with a Once before any Timer is created and are
// only torn down after everything else has exited. This means that these
// variables are read-only during use (after initialization) and both of which
// are safe to use concurrently.
static mut HELPER_CHAN: *mut SharedChan<Req> = 0 as *mut SharedChan<Req>;
static mut HELPER_SIGNAL: imp::signal = 0 as imp::signal;

pub fn boot(helper: fn(imp::signal, Port<Req>)) {
    static mut INIT: Once = ONCE_INIT;

    unsafe {
        INIT.doit(|| {
            let (msgp, msgc) = SharedChan::new();
            HELPER_CHAN = cast::transmute(~msgc);
            let (receive, send) = imp::new();
            HELPER_SIGNAL = send;

            do task::spawn {
                bookkeeping::decrement();
                helper(receive, msgp);
            }

            rt::at_exit(proc() { shutdown() });
        })
    }
}

pub fn send(req: Req) {
    unsafe {
        assert!(!HELPER_CHAN.is_null());
        (*HELPER_CHAN).send(req);
        imp::signal(HELPER_SIGNAL);
    }
}

fn shutdown() {
    // We want to wait for the entire helper task to exit, and in doing so it
    // will attempt to decrement the global task count. When the helper was
    // created, it decremented the count so it wouldn't count towards preventing
    // the program to exit, so here we pair that manual decrement with a manual
    // increment. We will then wait for the helper thread to exit by calling
    // wait_for_other_tasks.
    bookkeeping::increment();

    // Request a shutdown, and then wait for the task to exit
    send(Shutdown);
    bookkeeping::wait_for_other_tasks();

    // Clean up after ther helper thread
    unsafe {
        imp::close(HELPER_SIGNAL);
        let _chan: ~SharedChan<Req> = cast::transmute(HELPER_CHAN);
        HELPER_CHAN = 0 as *mut SharedChan<Req>;
        HELPER_SIGNAL = 0 as imp::signal;
    }
}

#[cfg(unix)]
mod imp {
    use std::libc;
    use std::os;

    use io::file::FileDesc;

    pub type signal = libc::c_int;

    pub fn new() -> (signal, signal) {
        let pipe = os::pipe();
        (pipe.input, pipe.out)
    }

    pub fn signal(fd: libc::c_int) {
        FileDesc::new(fd, false).inner_write([0]);
    }

    pub fn close(fd: libc::c_int) {
        let _fd = FileDesc::new(fd, true);
    }
}

#[cfg(windows)]
mod imp {
    use std::libc::{BOOL, LPCSTR, HANDLE, LPSECURITY_ATTRIBUTES, CloseHandle};
    use std::ptr;
    use std::libc;

    pub type signal = HANDLE;

    pub fn new() -> (HANDLE, HANDLE) {
        unsafe {
            let handle = CreateEventA(ptr::mut_null(), libc::FALSE, libc::FALSE,
                                      ptr::null());
            (handle, handle)
        }
    }

    pub fn signal(handle: HANDLE) {
        unsafe { SetEvent(handle); }
    }

    pub fn close(handle: HANDLE) {
        unsafe { CloseHandle(handle); }
    }

    extern "system" {
        fn CreateEventA(lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
                        bManualReset: BOOL,
                        bInitialState: BOOL,
                        lpName: LPCSTR) -> HANDLE;
        fn SetEvent(hEvent: HANDLE) -> BOOL;
    }
}
