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
    use libc;
    libc::abort()
}

// On Windows, we want to avoid using libc, and there isn't a direct
// equivalent of libc::abort.  The __failfast intrinsic may be a reasonable
// substitute, but desireability of using it over the abort instrinsic is
// debateable; see https://github.com/rust-lang/rust/pull/31519 for details.
#[cfg(not(unix))]
unsafe fn abort_internal() -> ! {
    use intrinsics;
    intrinsics::abort()
}

pub fn abort(args: fmt::Arguments) -> ! {
    dumb_print(format_args!("fatal runtime error: {}\n", args));
    unsafe { abort_internal(); }
}

#[allow(dead_code)] // stack overflow detection not enabled on all platforms
pub unsafe fn report_overflow() {
    dumb_print(format_args!("\nthread '{}' has overflowed its stack\n",
                            thread::current().name().unwrap_or("<unknown>")));
}
