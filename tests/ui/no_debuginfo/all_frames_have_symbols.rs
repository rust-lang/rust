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
    let mut missing_symbol_indices = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
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
            missing_symbol_indices.push(i);
        }
    }

    // FIXME(#346) currently on MinGW we can't symbolize kernel32.dll and other
    // system libraries, which means we miss the last few symbols.
    if cfg!(windows) && cfg!(target_env = "gnu") {
        assert!(missing_symbols < has_symbols && has_symbols > 4);
    } else if cfg!(all(target_os = "linux", target_env = "gnu")) {
        //NOTE: The reason we allow one missing symbol is because the frame for the
        // `__libc_start_main` fn doesn't have a symbol. See the discussion in
        // #152860 for more details.
        assert!(missing_symbols < has_symbols && missing_symbols <= 1)
    } else {
        for i in missing_symbol_indices {
            eprintln!("missing symbol for frame {i}: {:#?}", frames[i]);
        }
        eprintln!("Full erroneous backtrace: {:#?}", btrace);
        assert_eq!(missing_symbols, 0);
    }
}
