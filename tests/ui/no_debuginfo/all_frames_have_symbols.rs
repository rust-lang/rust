//@ compile-flags: -Cstrip=none -Cdebuginfo=none
//@ run-pass
#![feature(backtrace_frames)]
#![feature(backtrace_internals_accessors)]
use std::backtrace;
fn main() {
    let mut missing_symbols = 0;
    let mut has_symbols = 0;
    let btrace = backtrace::Backtrace::force_capture();
    let frames = btrace.frames();
    for frame in frames {
        let mut any = false;
        for sym in frame.symbols() {
            if sym.name().is_some() {
                any = true;
                break;
            }
        }
        if any {
            has_symbols += 1;
        } else if !frame.ip().is_null() {
            missing_symbols += 1;
        }
    }

    // FIXME(#346) currently on MinGW we can't symbolize kernel32.dll and other
    // system libraries, which means we miss the last few symbols.
    if cfg!(windows) && cfg!(target_env = "gnu") {
        assert!(missing_symbols < has_symbols && has_symbols > 4);
    } else {
        assert_eq!(missing_symbols, 0);
    }
}
