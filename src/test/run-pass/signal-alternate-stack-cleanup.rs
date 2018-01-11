// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Previously memory for alternate signal stack have been unmapped during
// main thread exit while still being in use by signal handlers. This test
// triggers this situation by sending signal from atexit handler.
//
// ignore-cloudabi no signal handling support
// ignore-wasm32-bare no libc
// ignore-windows

#![feature(libc)]
extern crate libc;

use libc::*;

unsafe extern fn signal_handler(signum: c_int, _: *mut siginfo_t, _: *mut c_void) {
    assert_eq!(signum, SIGWINCH);
}

extern fn send_signal() {
    unsafe {
        raise(SIGWINCH);
    }
}

fn main() {
    unsafe {
        // Install signal hander that runs on alternate signal stack.
        let mut action: sigaction = std::mem::zeroed();
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        action.sa_sigaction = signal_handler as sighandler_t;
        sigaction(SIGWINCH, &action, std::ptr::null_mut());

        // Send SIGWINCH on exit.
        atexit(send_signal);
    }
}

