// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![macro_escape]

/// Entry point of failure, for details, see std::macros
#[macro_export]
macro_rules! fail(
    () => (
        fail!("{}", "explicit failure")
    );
    ($msg:expr) => (
        fail!("{}", $msg)
    );
    ($fmt:expr, $($arg:tt)*) => ({
        // a closure can't have return type !, so we need a full
        // function to pass to format_args!, *and* we need the
        // file and line numbers right here; so an inner bare fn
        // is our only choice.
        //
        // LLVM doesn't tend to inline this, presumably because begin_unwind_fmt
        // is #[cold] and #[inline(never)] and because this is flagged as cold
        // as returning !. We really do want this to be inlined, however,
        // because it's just a tiny wrapper. Small wins (156K to 149K in size)
        // were seen when forcing this to be inlined, and that number just goes
        // up with the number of calls to fail!()
        #[inline(always)]
        fn run_fmt(fmt: &::std::fmt::Arguments) -> ! {
            ::core::failure::begin_unwind(fmt, file!(), line!())
        }
        format_args!(run_fmt, $fmt, $($arg)*)
    });
)

/// Runtime assertion, for details see std::macros
#[macro_export]
macro_rules! assert(
    ($cond:expr) => (
        if !$cond {
            fail!(concat!("assertion failed: ", stringify!($cond)))
        }
    );
    ($cond:expr, $($arg:tt)*) => (
        if !$cond {
            fail!($($arg)*)
        }
    );
)

/// Runtime assertion, disableable at compile time
#[macro_export]
macro_rules! debug_assert(
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { assert!($($arg)*); })
)
