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

// On Unix-like platforms, libc::abort will unregister signal handlers
// including the SIGABRT handler, preventing the abort from being blocked, and
// fclose streams, with the side effect of flushing them so libc bufferred
// output will be printed.  Additionally the shell will generally print a more
// understandable error message like "Abort trap" rather than "Illegal
// instruction" that intrinsics::abort would cause, as intrinsics::abort is
// implemented as an illegal instruction.
#[cfg(unix)]
unsafe fn abort_internal() -> ! {
    ::libc::abort()
}

// On Windows, use the processor-specific __fastfail mechanism.  In Windows 8
// and later, this will terminate the process immediately without running any
// in-process exception handlers.  In earlier versions of Windows, this
// sequence of instructions will be treated as an access violation,
// terminating the process but without necessarily bypassing all exception
// handlers.
//
// https://msdn.microsoft.com/en-us/library/dn774154.aspx
#[cfg(all(windows, any(target_arch = "x86", target_arch = "x86_64")))]
unsafe fn abort_internal() -> ! {
    asm!("int $$0x29" :: "{ecx}"(7) ::: volatile); // 7 is FAST_FAIL_FATAL_APP_EXIT
    ::intrinsics::unreachable();
}

// Other platforms should use the appropriate platform-specific mechanism for
// aborting the process.  If no platform-specific mechanism is available,
// ::intrinsics::abort() may be used instead.  The above implementations cover
// all targets currently supported by libstd.

pub fn abort(args: fmt::Arguments) -> ! {
    dumb_print(format_args!("fatal runtime error: {}\n", args));
    unsafe { abort_internal(); }
}

#[allow(dead_code)] // stack overflow detection not enabled on all platforms
pub unsafe fn report_overflow() {
    dumb_print(format_args!("\nthread '{}' has overflowed its stack\n",
                            thread::current().name().unwrap_or("<unknown>")));
}
