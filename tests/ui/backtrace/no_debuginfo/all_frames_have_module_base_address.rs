//@ compile-flags: -Cstrip=none -Cdebuginfo=none
//@ run-pass
#![feature(backtrace_frames)]
#![feature(backtrace_internals_accessors)]
use std::backtrace;

fn main() {
    let mut missing_base_addresses = 0;
    let btrace = backtrace::Backtrace::force_capture();
    let frames = btrace.frames();
    for frame in frames {
        if frame.module_base_address().is_none() {
            missing_base_addresses += 1;
        }
    }

    if cfg!(windows) {
        assert_eq!(missing_base_addresses, 0);
    }
}
