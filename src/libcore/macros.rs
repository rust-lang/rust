// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Entry point of task panic, for details, see std::macros
#[macro_export]
macro_rules! panic {
    () => (
        panic!("explicit panic")
    );
    ($msg:expr) => ({
        static _MSG_FILE_LINE: (&'static str, &'static str, usize) = ($msg, file!(), line!());
        ::core::panicking::panic(&_MSG_FILE_LINE)
    });
    ($fmt:expr, $($arg:tt)*) => ({
        // The leading _'s are to avoid dead code warnings if this is
        // used inside a dead function. Just `#[allow(dead_code)]` is
        // insufficient, since the user may have
        // `#[forbid(dead_code)]` and which cannot be overridden.
        static _FILE_LINE: (&'static str, usize) = (file!(), line!());
        ::core::panicking::panic_fmt(format_args!($fmt, $($arg)*), &_FILE_LINE)
    });
}

/// Ensure that a boolean expression is `true` at runtime.
///
/// This will invoke the `panic!` macro if the provided expression cannot be
/// evaluated to `true` at runtime.
///
/// # Example
///
/// ```
/// // the panic message for these assertions is the stringified value of the
/// // expression given.
/// assert!(true);
/// # fn some_computation() -> bool { true }
/// assert!(some_computation());
///
/// // assert with a custom message
/// # let x = true;
/// assert!(x, "x wasn't true!");
/// # let a = 3i; let b = 27i;
/// assert!(a + b == 30, "a = {}, b = {}", a, b);
/// ```
#[macro_export]
macro_rules! assert {
    ($cond:expr) => (
        if !$cond {
            panic!(concat!("assertion failed: ", stringify!($cond)))
        }
    );
    ($cond:expr, $($arg:expr),+) => (
        if !$cond {
            panic!($($arg),+)
        }
    );
}

/// Asserts that two expressions are equal to each other, testing equality in
/// both directions.
///
/// On panic, this macro will print the values of the expressions.
///
/// # Example
///
/// ```
/// let a = 3i;
/// let b = 1i + 2i;
/// assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! assert_eq {
    ($left:expr , $right:expr) => ({
        match (&($left), &($right)) {
            (left_val, right_val) => {
                // check both directions of equality....
                if !((*left_val == *right_val) &&
                     (*right_val == *left_val)) {
                    panic!("assertion failed: `(left == right) && (right == left)` \
                           (left: `{:?}`, right: `{:?}`)", *left_val, *right_val)
                }
            }
        }
    })
}

/// Ensure that a boolean expression is `true` at runtime.
///
/// This will invoke the `panic!` macro if the provided expression cannot be
/// evaluated to `true` at runtime.
///
/// Unlike `assert!`, `debug_assert!` statements can be disabled by passing
/// `--cfg ndebug` to the compiler. This makes `debug_assert!` useful for
/// checks that are too expensive to be present in a release build but may be
/// helpful during development.
///
/// # Example
///
/// ```
/// // the panic message for these assertions is the stringified value of the
/// // expression given.
/// debug_assert!(true);
/// # fn some_expensive_computation() -> bool { true }
/// debug_assert!(some_expensive_computation());
///
/// // assert with a custom message
/// # let x = true;
/// debug_assert!(x, "x wasn't true!");
/// # let a = 3i; let b = 27i;
/// debug_assert!(a + b == 30, "a = {}, b = {}", a, b);
/// ```
#[macro_export]
macro_rules! debug_assert {
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { assert!($($arg)*); })
}

/// Asserts that two expressions are equal to each other, testing equality in
/// both directions.
///
/// On panic, this macro will print the values of the expressions.
///
/// Unlike `assert_eq!`, `debug_assert_eq!` statements can be disabled by
/// passing `--cfg ndebug` to the compiler. This makes `debug_assert_eq!`
/// useful for checks that are too expensive to be present in a release build
/// but may be helpful during development.
///
/// # Example
///
/// ```
/// let a = 3i;
/// let b = 1i + 2i;
/// debug_assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! debug_assert_eq {
    ($($arg:tt)*) => (if cfg!(not(ndebug)) { assert_eq!($($arg)*); })
}

/// Short circuiting evaluation on Err
///
/// `libstd` contains a more general `try!` macro that uses `FromError`.
#[macro_export]
macro_rules! try {
    ($e:expr) => ({
        use $crate::result::Result::{Ok, Err};

        match $e {
            Ok(e) => e,
            Err(e) => return Err(e),
        }
    })
}

/// Use the `format!` syntax to write data into a buffer of type `&mut Writer`.
/// See `std::fmt` for more information.
///
/// # Example
///
/// ```
/// # #![allow(unused_must_use)]
///
/// let mut w = Vec::new();
/// write!(&mut w, "test");
/// write!(&mut w, "formatted {}", "arguments");
/// ```
#[macro_export]
macro_rules! write {
    ($dst:expr, $($arg:tt)*) => ((&mut *$dst).write_fmt(format_args!($($arg)*)))
}

/// Equivalent to the `write!` macro, except that a newline is appended after
/// the message is written.
#[macro_export]
#[stable]
macro_rules! writeln {
    ($dst:expr, $fmt:expr) => (
        write!($dst, concat!($fmt, "\n"))
    );
    ($dst:expr, $fmt:expr, $($arg:tt)*) => (
        write!($dst, concat!($fmt, "\n"), $($arg)*)
    );
}

/// A utility macro for indicating unreachable code.
///
/// This is useful any time that the compiler can't determine that some code is unreachable. For
/// example:
///
/// * Match arms with guard conditions.
/// * Loops that dynamically terminate.
/// * Iterators that dynamically terminate.
///
/// # Panics
///
/// This will always panic.
///
/// # Examples
///
/// Match arms:
///
/// ```rust
/// fn foo(x: Option<int>) {
///     match x {
///         Some(n) if n >= 0 => println!("Some(Non-negative)"),
///         Some(n) if n <  0 => println!("Some(Negative)"),
///         Some(_)           => unreachable!(), // compile error if commented out
///         None              => println!("None")
///     }
/// }
/// ```
///
/// Iterators:
///
/// ```rust
/// fn divide_by_three(x: u32) -> u32 { // one of the poorest implementations of x/3
///     for i in std::iter::count(0_u32, 1) {
///         if 3*i < i { panic!("u32 overflow"); }
///         if x < 3*i { return i-1; }
///     }
///     unreachable!();
/// }
/// ```
#[macro_export]
macro_rules! unreachable {
    () => ({
        panic!("internal error: entered unreachable code")
    });
    ($msg:expr) => ({
        unreachable!("{}", $msg)
    });
    ($fmt:expr, $($arg:tt)*) => ({
        panic!(concat!("internal error: entered unreachable code: ", $fmt), $($arg)*)
    });
}

/// A standardised placeholder for marking unfinished code. It panics with the
/// message `"not yet implemented"` when executed.
#[macro_export]
macro_rules! unimplemented {
    () => (panic!("not yet implemented"))
}
