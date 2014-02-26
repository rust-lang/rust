// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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
use std::rt::bookkeeping;
use std::rt;
use std::unstable::mutex::{StaticNativeMutex, NATIVE_MUTEX_INIT};

use io::timer::{Req, Shutdown};
use task;

// You'll note that these variables are *not* protected by a lock. These
// variables are initialized with a Once before any Timer is created and are
// only torn down after everything else has exited. This means that these
// variables are read-only during use (after initialization) and both of which
// are safe to use concurrently.
static mut HELPER_CHAN: *mut Sender<Req> = 0 as *mut Sender<Req>;
static mut HELPER_SIGNAL: imp::signal = 0 as imp::signal;

static mut TIMER_HELPER_EXIT: StaticNativeMutex = NATIVE_MUTEX_INIT;

pub fn boot(helper: fn(imp::signal, Receiver<Req>)) {
    static mut LOCK: StaticNativeMutex = NATIVE_MUTEX_INIT;
    static mut INITIALIZED: bool = false;

    unsafe {
        let mut _guard = LOCK.lock();
        if !INITIALIZED {
            let (tx, rx) = channel();
            // promote this to a shared channel
            drop(tx.clone());
            HELPER_CHAN = cast::transmute(~tx);
            let (receive, send) = imp::new();
            HELPER_SIGNAL = send;

            task::spawn(proc() {
                bookkeeping::decrement();
                helper(receive, rx);
                TIMER_HELPER_EXIT.lock().signal()
            });

            rt::at_exit(proc() { shutdown() });
            INITIALIZED = true;
        }
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
    // Request a shutdown, and then wait for the task to exit
    unsafe {
        let guard = TIMER_HELPER_EXIT.lock();
        send(Shutdown);
        guard.wait();
        drop(guard);
        TIMER_HELPER_EXIT.destroy();
    }


    // Clean up after ther helper thread
    unsafe {
        imp::close(HELPER_SIGNAL);
        let _chan: ~Sender<Req> = cast::transmute(HELPER_CHAN);
        HELPER_CHAN = 0 as *mut Sender<Req>;
        HELPER_SIGNAL = 0 as imp::signal;
    }
}

#[cfg(unix)]
mod imp {
    use libc;
    use std::os;

    use io::file::FileDesc;

    pub type signal = libc::c_int;

    pub fn new() -> (signal, signal) {
        let pipe = os::pipe();
        (pipe.input, pipe.out)
    }

    pub fn signal(fd: libc::c_int) {
        FileDesc::new(fd, false).inner_write([0]).unwrap();
    }

    pub fn close(fd: libc::c_int) {
        let _fd = FileDesc::new(fd, true);
    }
}

#[cfg(windows)]
mod imp {
    use libc::{BOOL, LPCSTR, HANDLE, LPSECURITY_ATTRIBUTES, CloseHandle};
    use std::ptr;
    use libc;

    pub type signal = HANDLE;

    pub fn new() -> (HANDLE, HANDLE) {
        unsafe {
            let handle = CreateEventA(ptr::mut_null(), libc::FALSE, libc::FALSE,
                                      ptr::null());
            (handle, handle)
        }
    }

    pub fn signal(handle: HANDLE) {
        assert!(unsafe { SetEvent(handle) != 0 });
    }

    pub fn close(handle: HANDLE) {
        assert!(unsafe { CloseHandle(handle) != 0 });
    }

    extern "system" {
        fn CreateEventA(lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
                        bManualReset: BOOL,
                        bInitialState: BOOL,
                        lpName: LPCSTR) -> HANDLE;
        fn SetEvent(hEvent: HANDLE) -> BOOL;
    }
}
