// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15677

use io::prelude::*;

use env;
use fmt;
use intrinsics;
use libc::uintptr_t;
use sync::atomic::{self, Ordering};
use sys::stdio::Stderr;

/// Dynamically inquire about whether we're running under V.
/// You should usually not use this unless your test definitely
/// can't run correctly un-altered. Valgrind is there to help
/// you notice weirdness in normal, un-doctored code paths!
pub fn running_on_valgrind() -> bool {
    extern {
        fn rust_running_on_valgrind() -> uintptr_t;
    }
    unsafe { rust_running_on_valgrind() != 0 }
}

/// Valgrind has a fixed-sized array (size around 2000) of segment descriptors
/// wired into it; this is a hard limit and requires rebuilding valgrind if you
/// want to go beyond it. Normally this is not a problem, but in some tests, we
/// produce a lot of threads casually.  Making lots of threads alone might not
/// be a problem _either_, except on OSX, the segments produced for new threads
/// _take a while_ to get reclaimed by the OS. Combined with the fact that libuv
/// schedulers fork off a separate thread for polling fsevents on OSX, we get a
/// perfect storm of creating "too many mappings" for valgrind to handle when
/// running certain stress tests in the runtime.
pub fn limit_thread_creation_due_to_osx_and_valgrind() -> bool {
    (cfg!(target_os="macos")) && running_on_valgrind()
}

pub fn min_stack() -> uint {
    static MIN: atomic::AtomicUsize = atomic::ATOMIC_USIZE_INIT;
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
    let _ = write!(&mut Stderr::new(), "{}", args);
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
