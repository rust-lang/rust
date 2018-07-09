// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt;
use io::prelude::*;
use sys::stdio::{Stderr, stderr_prints_nothing};
use thread;

#[cfg(feature = "backtrace")]
use sys_common::backtrace;

pub fn dumb_print(args: fmt::Arguments) {
    if stderr_prints_nothing() {
        return
    }
    let _ = Stderr::new().map(|mut stderr| stderr.write_fmt(args));
}

// Other platforms should use the appropriate platform-specific mechanism for
// aborting the process.  If no platform-specific mechanism is available,
// ::intrinsics::abort() may be used instead.  The above implementations cover
// all targets currently supported by libstd.

pub fn abort(args: fmt::Arguments) -> ! {
    dumb_print(format_args!("fatal runtime error: {}\n", args));
    unsafe { ::sys::abort_internal(); }
}

#[allow(dead_code)] // stack overflow detection not enabled on all platforms
pub unsafe fn report_overflow() {
    dumb_print(format_args!("\nthread '{}' has overflowed its stack\n",
                            thread::current().name().unwrap_or("<unknown>")));

    #[cfg(feature = "backtrace")]
    {
        let log_backtrace = backtrace::log_enabled();

        use sync::atomic::{AtomicBool, Ordering};

        static FIRST_OVERFLOW: AtomicBool = AtomicBool::new(true);

        if let Some(format) = log_backtrace {
            if let Ok(mut stderr) = Stderr::new() {
                let _ = backtrace::print(&mut stderr, format);
            }
        } else if FIRST_OVERFLOW.compare_and_swap(true, false, Ordering::SeqCst) {
            dumb_print(format_args!("note: Run with `RUST_BACKTRACE=1` for a backtrace."));
        }
    }
}
