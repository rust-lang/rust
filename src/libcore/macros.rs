// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Entry point of thread panic, for details, see std::macros
#[macro_export]
#[allow_internal_unstable]
macro_rules! panic {
    () => (
        panic!("explicit panic")
    );
    ($msg:expr) => ({
        static _MSG_FILE_LINE: (&'static str, &'static str, u32) = ($msg, file!(), line!());
        $crate::panicking::panic(&_MSG_FILE_LINE)
    });
    ($fmt:expr, $($arg:tt)*) => ({
        // The leading _'s are to avoid dead code warnings if this is
        // used inside a dead function. Just `#[allow(dead_code)]` is
        // insufficient, since the user may have
        // `#[forbid(dead_code)]` and which cannot be overridden.
        static _FILE_LINE: (&'static str, u32) = (file!(), line!());
        $crate::panicking::panic_fmt(format_args!($fmt, $($arg)*), &_FILE_LINE)
    });
}

/// Ensure that a boolean expression is `true` at runtime.
///
/// This will invoke the `panic!` macro if the provided expression cannot be
/// evaluated to `true` at runtime.
///
/// This macro has a second version, where a custom panic message can be provided.
///
/// # Examples
///
/// ```
/// // the panic message for these assertions is the stringified value of the
/// // expression given.
/// assert!(true);
///
/// fn some_computation() -> bool { true } // a very simple function
///
/// assert!(some_computation());
///
/// // assert with a custom message
/// let x = true;
/// assert!(x, "x wasn't true!");
///
/// let a = 3; let b = 27;
/// assert!(a + b == 30, "a = {}, b = {}", a, b);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! assert {
    ($cond:expr) => (
        if !$cond {
            panic!(concat!("assertion failed: ", stringify!($cond)))
        }
    );
    ($cond:expr, $($arg:tt)+) => (
        if !$cond {
            panic!($($arg)+)
        }
    );
}

/// Asserts that two expressions are equal to each other.
///
/// On panic, this macro will print the values of the expressions with their
/// debug representations.
///
/// # Examples
///
/// ```
/// let a = 3;
/// let b = 1 + 2;
/// assert_eq!(a, b);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! assert_eq {
    ($left:expr , $right:expr) => ({
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    panic!("assertion failed: `(left == right)` \
                           (left: `{:?}`, right: `{:?}`)", left_val, right_val)
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
/// Like `assert!`, this macro also has a second version, where a custom panic
/// message can be provided.
///
/// Unlike `assert!`, `debug_assert!` statements are only enabled in non
/// optimized builds by default. An optimized build will omit all
/// `debug_assert!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development.
///
/// # Examples
///
/// ```
/// // the panic message for these assertions is the stringified value of the
/// // expression given.
/// debug_assert!(true);
///
/// fn some_expensive_computation() -> bool { true } // a very simple function
/// debug_assert!(some_expensive_computation());
///
/// // assert with a custom message
/// let x = true;
/// debug_assert!(x, "x wasn't true!");
///
/// let a = 3; let b = 27;
/// debug_assert!(a + b == 30, "a = {}, b = {}", a, b);
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
macro_rules! debug_assert {
    ($($arg:tt)*) => (if cfg!(debug_assertions) { assert!($($arg)*); })
}

/// Asserts that two expressions are equal to each other, testing equality in
/// both directions.
///
/// On panic, this macro will print the values of the expressions.
///
/// Unlike `assert_eq!`, `debug_assert_eq!` statements are only enabled in non
/// optimized builds by default. An optimized build will omit all
/// `debug_assert_eq!` statements unless `-C debug-assertions` is passed to the
/// compiler. This makes `debug_assert_eq!` useful for checks that are too
/// expensive to be present in a release build but may be helpful during
/// development.
///
/// # Examples
///
/// ```
/// let a = 3;
/// let b = 1 + 2;
/// debug_assert_eq!(a, b);
/// ```
#[macro_export]
macro_rules! debug_assert_eq {
    ($($arg:tt)*) => (if cfg!(debug_assertions) { assert_eq!($($arg)*); })
}

/// Short circuiting evaluation on Err
///
/// `libstd` contains a more general `try!` macro that uses `From<E>`.
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

/// Use the `format!` syntax to write data into a buffer.
///
/// This macro is typically used with a buffer of `&mut `[`Write`][write].
///
/// See [`std::fmt`][fmt] for more information on format syntax.
///
/// [fmt]: fmt/index.html
/// [write]: io/trait.Write.html
///
/// # Examples
///
/// ```
/// use std::io::Write;
///
/// let mut w = Vec::new();
/// write!(&mut w, "test").unwrap();
/// write!(&mut w, "formatted {}", "arguments").unwrap();
///
/// assert_eq!(w, b"testformatted arguments");
/// ```
#[macro_export]
macro_rules! write {
    ($dst:expr, $($arg:tt)*) => ($dst.write_fmt(format_args!($($arg)*)))
}

/// Use the `format!` syntax to write data into a buffer, appending a newline.
///
/// This macro is typically used with a buffer of `&mut `[`Write`][write].
///
/// See [`std::fmt`][fmt] for more information on format syntax.
///
/// [fmt]: fmt/index.html
/// [write]: io/trait.Write.html
///
/// # Examples
///
/// ```
/// use std::io::Write;
///
/// let mut w = Vec::new();
/// writeln!(&mut w, "test").unwrap();
/// writeln!(&mut w, "formatted {}", "arguments").unwrap();
///
/// assert_eq!(&w[..], "test\nformatted arguments\n".as_bytes());
/// ```
#[macro_export]
#[stable(feature = "rust1", since = "1.0.0")]
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
/// ```
/// # #[allow(dead_code)]
/// fn foo(x: Option<i32>) {
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
/// ```
/// # #[allow(dead_code)]
/// fn divide_by_three(x: u32) -> u32 { // one of the poorest implementations of x/3
///     for i in 0.. {
///         if 3*i < i { panic!("u32 overflow"); }
///         if x < 3*i { return i-1; }
///     }
///     unreachable!();
/// }
/// ```
#[macro_export]
#[unstable(feature = "core",
           reason = "relationship with panic is unclear",
           issue = "27701")]
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

/// A standardized placeholder for marking unfinished code. It panics with the
/// message `"not yet implemented"` when executed.
///
/// This can be useful if you are prototyping and are just looking to have your
/// code typecheck, or if you're implementing a trait that requires multiple
/// methods, and you're only planning on using one of them.
///
/// # Examples
///
/// Here's an example of some in-progress code. We have a trait `Foo`:
///
/// ```
/// trait Foo {
///     fn bar(&self);
///     fn baz(&self);
/// }
/// ```
///
/// We want to implement `Foo` on one of our types, but we also want to work on
/// just `bar()` first. In order for our code to compile, we need to implement
/// `baz()`, so we can use `unimplemented!`:
///
/// ```
/// # trait Foo {
/// #     fn bar(&self);
/// #     fn baz(&self);
/// # }
/// struct MyStruct;
///
/// impl Foo for MyStruct {
///     fn bar(&self) {
///         // implementation goes here
///     }
///
///     fn baz(&self) {
///         // let's not worry about implementing baz() for now
///         unimplemented!();
///     }
/// }
///
/// fn main() {
///     let s = MyStruct;
///     s.bar();
///
///     // we aren't even using baz() yet, so this is fine.
/// }
/// ```
#[macro_export]
#[unstable(feature = "core",
           reason = "relationship with panic is unclear",
           issue = "27701")]
macro_rules! unimplemented {
    () => (panic!("not yet implemented"))
}
