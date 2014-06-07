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

/// Runtime assertion, only without `--cfg ndebug`
#[macro_export]
macro_rules! debug_assert(
    ($(a:tt)*) => ({
        if cfg!(not(ndebug)) {
            assert!($($a)*);
        }
    })
)

/// Runtime assertion for equality, for details see std::macros
#[macro_export]
macro_rules! assert_eq(
    ($cond1:expr, $cond2:expr) => ({
        let c1 = $cond1;
        let c2 = $cond2;
        if c1 != c2 || c2 != c1 {
            fail!("expressions not equal, left: {}, right: {}", c1, c2);
        }
    })
)

/// Runtime assertion for equality, only without `--cfg ndebug`
#[macro_export]
macro_rules! debug_assert_eq(
    ($($a:tt)*) => ({
        if cfg!(not(ndebug)) {
            assert_eq!($($a)*);
        }
    })
)

/// Runtime assertion, disableable at compile time
#[macro_export]
macro_rules! debug_assert(
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { assert!($($arg)*); })
)

/// Short circuiting evaluation on Err
#[macro_export]
macro_rules! try(
    ($e:expr) => (match $e { Ok(e) => e, Err(e) => return Err(e) })
)

/// Writing a formatted string into a writer
#[macro_export]
macro_rules! write(
    ($dst:expr, $($arg:tt)*) => (format_args_method!($dst, write_fmt, $($arg)*))
)

/// Writing a formatted string plus a newline into a writer
#[macro_export]
macro_rules! writeln(
    ($dst:expr, $fmt:expr $($arg:tt)*) => (
        write!($dst, concat!($fmt, "\n") $($arg)*)
    )
)

#[cfg(test)]
macro_rules! vec( ($($e:expr),*) => ({
    let mut _v = ::std::vec::Vec::new();
    $(_v.push($e);)*
    _v
}) )

#[cfg(test)]
macro_rules! format( ($($arg:tt)*) => (format_args!(::fmt::format, $($arg)*)) )

/// Write some formatted data into a stream.
///
/// Identical to the macro in `std::macros`
#[macro_export]
macro_rules! write(
    ($dst:expr, $($arg:tt)*) => ({
        format_args_method!($dst, write_fmt, $($arg)*)
    })
)
