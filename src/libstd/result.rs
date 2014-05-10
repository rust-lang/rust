// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error handling with the `Result` type
//!
//! `Result<T>` is the type used for returning and propagating
//! errors. It is an enum with the variants, `Ok(T)`, representing
//! success and containing a value, and `Err(E)`, representing error
//! and containing an error value.
//!
//! ~~~
//! enum Result<T, E> {
//!    Ok(T),
//!    Err(E)
//! }
//! ~~~
//!
//! Functions return `Result` whenever errors are expected and
//! recoverable. In the `std` crate `Result` is most prominently used
//! for [I/O](../io/index.html).
//!
//! A simple function returning `Result` might be
//! defined and used like so:
//!
//! ~~~
//! #[deriving(Show)]
//! enum Version { Version1, Version2 }
//!
//! fn parse_version(header: &[u8]) -> Result<Version, &'static str> {
//!     if header.len() < 1 {
//!         return Err("invalid header length");
//!     }
//!     match header[0] {
//!         1 => Ok(Version1),
//!         2 => Ok(Version2),
//!         _ => Err("invalid version")
//!     }
//! }
//!
//! let version = parse_version(&[1, 2, 3, 4]);
//! match version {
//!     Ok(v) => {
//!         println!("working with version: {}", v);
//!     }
//!     Err(e) => {
//!         println!("error parsing header: {}", e);
//!     }
//! }
//! ~~~
//!
//! Pattern matching on `Result`s is clear and straightforward for
//! simple cases, but `Result` comes with some convenience methods
//! that make working it more succinct.
//!
//! ~~~
//! let good_result: Result<int, int> = Ok(10);
//! let bad_result: Result<int, int> = Err(10);
//!
//! // The `is_ok` and `is_err` methods do what they say.
//! assert!(good_result.is_ok() && !good_result.is_err());
//! assert!(bad_result.is_err() && !bad_result.is_ok());
//!
//! // `map` consumes the `Result` and produces another.
//! let good_result: Result<int, int> = good_result.map(|i| i + 1);
//! let bad_result: Result<int, int> = bad_result.map(|i| i - 1);
//!
//! // Use `and_then` to continue the computation.
//! let good_result: Result<bool, int> = good_result.and_then(|i| Ok(i == 11));
//!
//! // Use `or_else` to handle the error.
//! let bad_result: Result<int, int> = bad_result.or_else(|i| Ok(11));
//!
//! // Consume the result and return the contents with `unwrap`.
//! let final_awesome_result = good_result.ok().unwrap();
//! ~~~
//!
//! # Results must be used
//!
//! A common problem with using return values to indicate errors is
//! that it is easy to ignore the return value, thus failing to handle
//! the error. Result is annotated with the #[must_use] attribute,
//! which will cause the compiler to issue a warning when a Result
//! value is ignored. This makes `Result` especially useful with
//! functions that may encounter errors but don't otherwise return a
//! useful value.
//!
//! Consider the `write_line` method defined for I/O types
//! by the [`Writer`](../io/trait.Writer.html) trait:
//!
//! ~~~
//! use std::io::IoError;
//!
//! trait Writer {
//!     fn write_line(&mut self, s: &str) -> Result<(), IoError>;
//! }
//! ~~~
//!
//! *Note: The actual definition of `Writer` uses `IoResult`, which
//! is just a synonym for `Result<T, IoError>`.*
//!
//! This method doesn`t produce a value, but the write may
//! fail. It's crucial to handle the error case, and *not* write
//! something like this:
//!
//! ~~~ignore
//! use std::io::{File, Open, Write};
//!
//! let mut file = File::open_mode(&Path::new("valuable_data.txt"), Open, Write);
//! // If `write_line` errors, then we'll never know, because the return
//! // value is ignored.
//! file.write_line("important message");
//! drop(file);
//! ~~~
//!
//! If you *do* write that in Rust, the compiler will by give you a
//! warning (by default, controlled by the `unused_must_use` lint).
//!
//! You might instead, if you don't want to handle the error, simply
//! fail, by converting to an `Option` with `ok`, then asserting
//! success with `expect`. This will fail if the write fails, proving
//! a marginally useful message indicating why:
//!
//! ~~~no_run
//! use std::io::{File, Open, Write};
//!
//! let mut file = File::open_mode(&Path::new("valuable_data.txt"), Open, Write);
//! file.write_line("important message").ok().expect("failed to write message");
//! drop(file);
//! ~~~
//!
//! You might also simply assert success:
//!
//! ~~~no_run
//! # use std::io::{File, Open, Write};
//!
//! # let mut file = File::open_mode(&Path::new("valuable_data.txt"), Open, Write);
//! assert!(file.write_line("important message").is_ok());
//! # drop(file);
//! ~~~
//!
//! Or propagate the error up the call stack with `try!`:
//!
//! ~~~
//! # use std::io::{File, Open, Write, IoError};
//! fn write_message() -> Result<(), IoError> {
//!     let mut file = File::open_mode(&Path::new("valuable_data.txt"), Open, Write);
//!     try!(file.write_line("important message"));
//!     drop(file);
//!     return Ok(());
//! }
//! ~~~
//!
//! # The `try!` macro
//!
//! When writing code that calls many functions that return the
//! `Result` type, the error handling can be tedious.  The `try!`
//! macro hides some of the boilerplate of propagating errors up the
//! call stack.
//!
//! It replaces this:
//!
//! ~~~
//! use std::io::{File, Open, Write, IoError};
//!
//! struct Info { name: ~str, age: int, rating: int }
//!
//! fn write_info(info: &Info) -> Result<(), IoError> {
//!     let mut file = File::open_mode(&Path::new("my_best_friends.txt"), Open, Write);
//!     // Early return on error
//!     match file.write_line(format!("name: {}", info.name)) {
//!         Ok(_) => (),
//!         Err(e) => return Err(e)
//!     }
//!     match file.write_line(format!("age: {}", info.age)) {
//!         Ok(_) => (),
//!         Err(e) => return Err(e)
//!     }
//!     return file.write_line(format!("rating: {}", info.rating));
//! }
//! ~~~
//!
//! With this:
//!
//! ~~~
//! use std::io::{File, Open, Write, IoError};
//!
//! struct Info { name: ~str, age: int, rating: int }
//!
//! fn write_info(info: &Info) -> Result<(), IoError> {
//!     let mut file = File::open_mode(&Path::new("my_best_friends.txt"), Open, Write);
//!     // Early return on error
//!     try!(file.write_line(format!("name: {}", info.name)));
//!     try!(file.write_line(format!("age: {}", info.age)));
//!     try!(file.write_line(format!("rating: {}", info.rating)));
//!     return Ok(());
//! }
//! ~~~
//!
//! *It's much nicer!*
//!
//! Wrapping an expression in `try!` will result in the unwrapped
//! success (`Ok`) value, unless the result is `Err`, in which case
//! `Err` is returned early from the enclosing function. Its simple definition
//! makes it clear:
//!
//! ~~~
//! # #![feature(macro_rules)]
//! macro_rules! try(
//!     ($e:expr) => (match $e { Ok(e) => e, Err(e) => return Err(e) })
//! )
//! # fn main() { }
//! ~~~
//!
//! `try!` is imported by the prelude, and is available everywhere.
//!
//! # `Result` and `Option`
//!
//! The `Result` and [`Option`](../option/index.html) types are
//! similar and complementary: they are often employed to indicate a
//! lack of a return value; and they are trivially converted between
//! each other, so `Result`s are often handled by first converting to
//! `Option` with the [`ok`](../../core/result/enum.Result.html#method.ok) and
//! [`err`](../../core/result/enum.Result.html#method.ok) methods.
//!
//! Whereas `Option` only indicates the lack of a value, `Result` is
//! specifically for error reporting, and carries with it an error
//! value.  Sometimes `Option` is used for indicating errors, but this
//! is only for simple cases and is generally discouraged. Even when
//! there is no useful error value to return, prefer `Result<T, ()>`.
//!
//! Converting to an `Option` with `ok()` to handle an error:
//!
//! ~~~
//! use std::io::Timer;
//! let mut t = Timer::new().ok().expect("failed to create timer!");
//! ~~~
//!
//! # `Result` vs. `fail!`
//!
//! `Result` is for recoverable errors; `fail!` is for unrecoverable
//! errors. Callers should always be able to avoid failure if they
//! take the proper precautions, for example, calling `is_some()`
//! on an `Option` type before calling `unwrap`.
//!
//! The suitability of `fail!` as an error handling mechanism is
//! limited by Rust's lack of any way to "catch" and resume execution
//! from a thrown exception. Therefore using failure for error
//! handling requires encapsulating fallable code in a task. Calling
//! the `fail!` macro, or invoking `fail!` indirectly should be
//! avoided as an error reporting strategy. Failure is only for
//! unrecoverable errors and a failing task is typically the sign of
//! a bug.
//!
//! A module that instead returns `Results` is alerting the caller
//! that failure is possible, and providing precise control over how
//! it is handled.
//!
//! Furthermore, failure may not be recoverable at all, depending on
//! the context. The caller of `fail!` should assume that execution
//! will not resume after failure, that failure is catastrophic.

use fmt::Show;

pub use core::result::{Result, Ok, Err, collect, fold, fold_};

// FIXME: These traits should not exist. Once std::fmt is moved to libcore,
//        these can once again become inherent methods on Result.

/// Temporary trait for unwrapping a result
pub trait ResultUnwrap<T, E> {
    /// Unwraps a result, yielding the content of an `Ok`.
    ///
    /// Fails if the value is an `Err`.
    fn unwrap(self) -> T;
}

/// Temporary trait for unwrapping the error of a result
pub trait ResultUnwrapErr<T, E> {
    /// Unwraps a result, yielding the content of an `Err`.
    ///
    /// Fails if the value is an `Ok`.
    fn unwrap_err(self) -> E;
}

impl<T, E: Show> ResultUnwrap<T, E> for Result<T, E> {
    #[inline]
    fn unwrap(self) -> T {
        match self {
            Ok(t) => t,
            Err(e) =>
                fail!("called `Result::unwrap()` on an `Err` value: {}", e)
        }
    }
}

impl<T: Show, E> ResultUnwrapErr<T, E> for Result<T, E> {
    #[inline]
    fn unwrap_err(self) -> E {
        match self {
            Ok(t) =>
                fail!("called `Result::unwrap_err()` on an `Ok` value: {}", t),
            Err(e) => e
        }
    }
}
