// run-pass
// Previously memory for alternate signal stack have been unmapped during
// main thread exit while still being in use by signal handlers. This test
// triggers this situation by sending signal from atexit handler.
//
// ignore-wasm32-bare no libc
// ignore-windows
// ignore-sgx no libc
// ignore-vxworks no SIGWINCH in user space

#![feature(rustc_private)]
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
        // Install signal handler that runs on alternate signal stack.
        let mut action: sigaction = std::mem::zeroed();
        action.sa_flags = (SA_ONSTACK | SA_SIGINFO) as _;
        action.sa_sigaction = signal_handler as sighandler_t;
        sigaction(SIGWINCH, &action, std::ptr::null_mut());

        // Send SIGWINCH on exit.
        atexit(send_signal);
    }
}
