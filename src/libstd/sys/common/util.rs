// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use env;
use fmt;
use io::prelude::*;
use sync::atomic::{self, Ordering};
use sys::stdio::Stderr;
use thread;

pub fn min_stack() -> usize {
    static MIN: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
    match MIN.load(Ordering::SeqCst) {
        0 => {}
        n => return n - 1,
    }
    let amt = env::var("RUST_MIN_STACK").ok().and_then(|s| s.parse().ok());
    let amt = amt.unwrap_or(2 * 1024 * 1024);
    // 0 is our sentinel value, so ensure that we'll never see 0 after
    // initialization has run
    MIN.store(amt + 1, Ordering::SeqCst);
    amt
}

pub fn dumb_print(args: fmt::Arguments) {
    let _ = Stderr::new().map(|mut stderr| stderr.write_fmt(args));
}

#[cfg(unix)]
pub fn abort(args: fmt::Arguments) -> ! {
    use libc;
    dumb_print(format_args!("fatal runtime error: {}\n", args));
    unsafe { libc::abort(); }
}

// On windows, use the processor-specific __fastfail mechanism
// https://msdn.microsoft.com/en-us/library/dn774154.aspx
#[cfg(all(windows, target_arch = "x86"))]
pub fn abort(args: fmt::Arguments) -> ! {
    dumb_print(format_args!("fatal runtime error: {}\n", args));
    unsafe {
        asm!("int $$0x29" :: "{ecx}"(7) ::: volatile); // 7 is FAST_FAIL_FATAL_APP_EXIT
        ::intrinsics::unreachable();
    }
}

#[cfg(all(windows, target_arch = "x86_64"))]
pub fn abort(args: fmt::Arguments) -> ! {
    dumb_print(format_args!("fatal runtime error: {}\n", args));
    unsafe {
        asm!("int $$0x29" :: "{rcx}"(7) ::: volatile); // 7 is FAST_FAIL_FATAL_APP_EXIT
        ::intrinsics::unreachable();
    }
}

#[allow(dead_code)] // stack overflow detection not enabled on all platforms
pub unsafe fn report_overflow() {
    dumb_print(format_args!("\nthread '{}' has overflowed its stack\n",
                            thread::current().name().unwrap_or("<unknown>")));
}
