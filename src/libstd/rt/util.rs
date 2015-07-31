// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::prelude::*;

use env;
use fmt;
use intrinsics;
use sync::atomic::{self, Ordering};
use sys::stdio::Stderr;

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
    return amt;
}

// Indicates whether we should perform expensive sanity checks, including rtassert!
//
// FIXME: Once the runtime matures remove the `true` below to turn off rtassert,
//        etc.
pub const ENFORCE_SANITY: bool = true || !cfg!(rtopt) || cfg!(rtdebug) ||
                                  cfg!(rtassert);

pub fn dumb_print(args: fmt::Arguments) {
    let _ = Stderr::new().map(|mut stderr| stderr.write_fmt(args));
}

pub fn abort(args: fmt::Arguments) -> ! {
    rterrln!("fatal runtime error: {}", args);
    unsafe { intrinsics::abort(); }
}

pub unsafe fn report_overflow() {
    use thread;
    rterrln!("\nthread '{}' has overflowed its stack",
             thread::current().name().unwrap_or("<unknown>"));
}
