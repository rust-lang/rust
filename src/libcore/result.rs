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
//! `Option` with the [`ok`](enum.Result.html#method.ok) and
//! [`err`](enum.Result.html#method.ok) methods.
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

use clone::Clone;
use cmp::Eq;
use iter::{Iterator, FromIterator};
use option::{None, Option, Some};

/// `Result` is a type that represents either success (`Ok`) or failure (`Err`).
///
/// See the [`std::result`](index.html) module documentation for details.
#[deriving(Clone, Eq, Ord, TotalEq, TotalOrd)]
#[must_use]
pub enum Result<T, E> {
    /// Contains the success value
    Ok(T),

    /// Contains the error value
    Err(E)
}

/////////////////////////////////////////////////////////////////////////////
// Type implementation
/////////////////////////////////////////////////////////////////////////////

impl<T, E> Result<T, E> {
    /////////////////////////////////////////////////////////////////////////
    // Querying the contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns true if the result is `Ok`
    ///
    /// # Example
    ///
    /// ~~~
    /// use std::io::{File, Open, Write};
    ///
    /// # fn do_not_run_example() { // creates a file
    /// let mut file = File::open_mode(&Path::new("secret.txt"), Open, Write);
    /// assert!(file.write_line("it's cold in here").is_ok());
    /// # }
    /// ~~~
    #[inline]
    pub fn is_ok(&self) -> bool {
        match *self {
            Ok(_) => true,
            Err(_) => false
        }
    }

    /// Returns true if the result is `Err`
    ///
    /// # Example
    ///
    /// ~~~
    /// use std::io::{File, Open, Read};
    ///
    /// // When opening with `Read` access, if the file does not exist
    /// // then `open_mode` returns an error.
    /// let bogus = File::open_mode(&Path::new("not_a_file.txt"), Open, Read);
    /// assert!(bogus.is_err());
    /// ~~~
    #[inline]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }


    /////////////////////////////////////////////////////////////////////////
    // Adapter for each variant
    /////////////////////////////////////////////////////////////////////////

    /// Convert from `Result<T, E>` to `Option<T>`
    ///
    /// Converts `self` into an `Option<T>`, consuming `self`,
    /// and discarding the error, if any.
    ///
    /// To convert to an `Option` without discarding the error value,
    /// use `as_ref` to first convert the `Result<T, E>` into a
    /// `Result<&T, &E>`.
    ///
    /// # Examples
    ///
    /// ~~~{.should_fail}
    /// use std::io::{File, IoResult};
    ///
    /// let bdays: IoResult<File> = File::open(&Path::new("important_birthdays.txt"));
    /// let bdays: File = bdays.ok().expect("unable to open birthday file");
    /// ~~~
    #[inline]
    pub fn ok(self) -> Option<T> {
        match self {
            Ok(x)  => Some(x),
            Err(_) => None,
        }
    }

    /// Convert from `Result<T, E>` to `Option<E>`
    ///
    /// Converts `self` into an `Option<T>`, consuming `self`,
    /// and discarding the value, if any.
    #[inline]
    pub fn err(self) -> Option<E> {
        match self {
            Ok(_)  => None,
            Err(x) => Some(x),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Convert from `Result<T, E>` to `Result<&T, &E>`
    ///
    /// Produces a new `Result`, containing a reference
    /// into the original, leaving the original in place.
    #[inline]
    pub fn as_ref<'r>(&'r self) -> Result<&'r T, &'r E> {
        match *self {
            Ok(ref x) => Ok(x),
            Err(ref x) => Err(x),
        }
    }

    /// Convert from `Result<T, E>` to `Result<&mut T, &mut E>`
    #[inline]
    pub fn as_mut<'r>(&'r mut self) -> Result<&'r mut T, &'r mut E> {
        match *self {
            Ok(ref mut x) => Ok(x),
            Err(ref mut x) => Err(x),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Transforming contained values
    /////////////////////////////////////////////////////////////////////////

    /// Maps a `Result<T, E>` to `Result<U, E>` by applying a function to an
    /// contained `Ok` value, leaving an `Err` value untouched.
    ///
    /// This function can be used to compose the results of two functions.
    ///
    /// # Examples
    ///
    /// Sum the lines of a buffer by mapping strings to numbers,
    /// ignoring I/O and parse errors:
    ///
    /// ~~~
    /// use std::io::{BufReader, IoResult};
    ///
    /// let buffer = "1\n2\n3\n4\n";
    /// let mut reader = BufReader::new(buffer.as_bytes());
    ///
    /// let mut sum = 0;
    ///
    /// while !reader.eof() {
    ///     let line: IoResult<~str> = reader.read_line();
    ///     // Convert the string line to a number using `map` and `from_str`
    ///     let val: IoResult<int> = line.map(|line| {
    ///         from_str::<int>(line).unwrap_or(0)
    ///     });
    ///     // Add the value if there were no errors, otherwise add 0
    ///     sum += val.ok().unwrap_or(0);
    /// }
    /// ~~~
    #[inline]
    pub fn map<U>(self, op: |T| -> U) -> Result<U,E> {
        match self {
          Ok(t) => Ok(op(t)),
          Err(e) => Err(e)
        }
    }

    /// Maps a `Result<T, E>` to `Result<T, F>` by applying a function to an
    /// contained `Err` value, leaving an `Ok` value untouched.
    ///
    /// This function can be used to pass through a successful result while handling
    /// an error.
    #[inline]
    pub fn map_err<F>(self, op: |E| -> F) -> Result<T,F> {
        match self {
          Ok(t) => Ok(t),
          Err(e) => Err(op(e))
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns `res` if the result is `Ok`, otherwise returns the `Err` value of `self`.
    #[inline]
    pub fn and<U>(self, res: Result<U, E>) -> Result<U, E> {
        match self {
            Ok(_) => res,
            Err(e) => Err(e),
        }
    }

    /// Calls `op` if the result is `Ok`, otherwise returns the `Err` value of `self`.
    ///
    /// This function can be used for control flow based on result values
    #[inline]
    pub fn and_then<U>(self, op: |T| -> Result<U, E>) -> Result<U, E> {
        match self {
            Ok(t) => op(t),
            Err(e) => Err(e),
        }
    }

    /// Returns `res` if the result is `Err`, otherwise returns the `Ok` value of `self`.
    #[inline]
    pub fn or(self, res: Result<T, E>) -> Result<T, E> {
        match self {
            Ok(_) => self,
            Err(_) => res,
        }
    }

    /// Calls `op` if the result is `Err`, otherwise returns the `Ok` value of `self`.
    ///
    /// This function can be used for control flow based on result values
    #[inline]
    pub fn or_else<F>(self, op: |E| -> Result<T, F>) -> Result<T, F> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => op(e),
        }
    }

    /// Unwraps a result, yielding the content of an `Ok`.
    /// Else it returns `optb`.
    #[inline]
    pub fn unwrap_or(self, optb: T) -> T {
        match self {
            Ok(t) => t,
            Err(_) => optb
        }
    }

    /// Unwraps a result, yielding the content of an `Ok`.
    /// If the value is an `Err` then it calls `op` with its value.
    #[inline]
    pub fn unwrap_or_handle(self, op: |E| -> T) -> T {
        match self {
            Ok(t) => t,
            Err(e) => op(e)
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Free functions
/////////////////////////////////////////////////////////////////////////////

/// Takes each element in the `Iterator`: if it is an `Err`, no further
/// elements are taken, and the `Err` is returned. Should no `Err` occur, a
/// vector containing the values of each `Result` is returned.
///
/// Here is an example which increments every integer in a vector,
/// checking for overflow:
///
///     fn inc_conditionally(x: uint) -> Result<uint, &'static str> {
///         if x == uint::MAX { return Err("overflow"); }
///         else { return Ok(x+1u); }
///     }
///     let v = [1u, 2, 3];
///     let res = collect(v.iter().map(|&x| inc_conditionally(x)));
///     assert!(res == Ok(~[2u, 3, 4]));
#[inline]
pub fn collect<T, E, Iter: Iterator<Result<T, E>>, V: FromIterator<T>>(iter: Iter) -> Result<V, E> {
    // FIXME(#11084): This should be twice as fast once this bug is closed.
    let mut iter = iter.scan(None, |state, x| {
        match x {
            Ok(x) => Some(x),
            Err(err) => {
                *state = Some(err);
                None
            }
        }
    });

    let v: V = FromIterator::from_iter(iter.by_ref());

    match iter.state {
        Some(err) => Err(err),
        None => Ok(v),
    }
}

/// Perform a fold operation over the result values from an iterator.
///
/// If an `Err` is encountered, it is immediately returned.
/// Otherwise, the folded value is returned.
#[inline]
pub fn fold<T,
            V,
            E,
            Iter: Iterator<Result<T, E>>>(
            mut iterator: Iter,
            mut init: V,
            f: |V, T| -> V)
            -> Result<V, E> {
    for t in iterator {
        match t {
            Ok(v) => init = f(init, v),
            Err(u) => return Err(u)
        }
    }
    Ok(init)
}

/// Perform a trivial fold operation over the result values
/// from an iterator.
///
/// If an `Err` is encountered, it is immediately returned.
/// Otherwise, a simple `Ok(())` is returned.
#[inline]
pub fn fold_<T,E,Iter:Iterator<Result<T,E>>>(iterator: Iter) -> Result<(),E> {
    fold(iterator, (), |_, _| ())
}

/////////////////////////////////////////////////////////////////////////////
// Tests
/////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use realstd::result::{collect, fold, fold_};
    use realstd::prelude::*;
    use realstd::iter::range;

    pub fn op1() -> Result<int, ~str> { Ok(666) }
    pub fn op2() -> Result<int, ~str> { Err("sadface".to_owned()) }

    #[test]
    pub fn test_and() {
        assert_eq!(op1().and(Ok(667)).unwrap(), 667);
        assert_eq!(op1().and(Err::<(), ~str>("bad".to_owned())).unwrap_err(), "bad".to_owned());

        assert_eq!(op2().and(Ok(667)).unwrap_err(), "sadface".to_owned());
        assert_eq!(op2().and(Err::<(), ~str>("bad".to_owned())).unwrap_err(), "sadface".to_owned());
    }

    #[test]
    pub fn test_and_then() {
        assert_eq!(op1().and_then(|i| Ok::<int, ~str>(i + 1)).unwrap(), 667);
        assert_eq!(op1().and_then(|_| Err::<int, ~str>("bad".to_owned())).unwrap_err(),
                   "bad".to_owned());

        assert_eq!(op2().and_then(|i| Ok::<int, ~str>(i + 1)).unwrap_err(),
                   "sadface".to_owned());
        assert_eq!(op2().and_then(|_| Err::<int, ~str>("bad".to_owned())).unwrap_err(),
                   "sadface".to_owned());
    }

    #[test]
    pub fn test_or() {
        assert_eq!(op1().or(Ok(667)).unwrap(), 666);
        assert_eq!(op1().or(Err("bad".to_owned())).unwrap(), 666);

        assert_eq!(op2().or(Ok(667)).unwrap(), 667);
        assert_eq!(op2().or(Err("bad".to_owned())).unwrap_err(), "bad".to_owned());
    }

    #[test]
    pub fn test_or_else() {
        assert_eq!(op1().or_else(|_| Ok::<int, ~str>(667)).unwrap(), 666);
        assert_eq!(op1().or_else(|e| Err::<int, ~str>(e + "!")).unwrap(), 666);

        assert_eq!(op2().or_else(|_| Ok::<int, ~str>(667)).unwrap(), 667);
        assert_eq!(op2().or_else(|e| Err::<int, ~str>(e + "!")).unwrap_err(),
                   "sadface!".to_owned());
    }

    #[test]
    pub fn test_impl_map() {
        assert_eq!(Ok::<~str, ~str>("a".to_owned()).map(|x| x + "b"), Ok("ab".to_owned()));
        assert_eq!(Err::<~str, ~str>("a".to_owned()).map(|x| x + "b"), Err("a".to_owned()));
    }

    #[test]
    pub fn test_impl_map_err() {
        assert_eq!(Ok::<~str, ~str>("a".to_owned()).map_err(|x| x + "b"), Ok("a".to_owned()));
        assert_eq!(Err::<~str, ~str>("a".to_owned()).map_err(|x| x + "b"), Err("ab".to_owned()));
    }

    #[test]
    fn test_collect() {
        let v: Result<~[int], ()> = collect(range(0, 0).map(|_| Ok::<int, ()>(0)));
        assert_eq!(v, Ok(box []));

        let v: Result<~[int], ()> = collect(range(0, 3).map(|x| Ok::<int, ()>(x)));
        assert_eq!(v, Ok(box [0, 1, 2]));

        let v: Result<~[int], int> = collect(range(0, 3)
                                             .map(|x| if x > 1 { Err(x) } else { Ok(x) }));
        assert_eq!(v, Err(2));

        // test that it does not take more elements than it needs
        let mut functions = [|| Ok(()), || Err(1), || fail!()];

        let v: Result<~[()], int> = collect(functions.mut_iter().map(|f| (*f)()));
        assert_eq!(v, Err(1));
    }

    #[test]
    fn test_fold() {
        assert_eq!(fold_(range(0, 0)
                        .map(|_| Ok::<(), ()>(()))),
                   Ok(()));
        assert_eq!(fold(range(0, 3)
                        .map(|x| Ok::<int, ()>(x)),
                        0, |a, b| a + b),
                   Ok(3));
        assert_eq!(fold_(range(0, 3)
                        .map(|x| if x > 1 { Err(x) } else { Ok(()) })),
                   Err(2));

        // test that it does not take more elements than it needs
        let mut functions = [|| Ok(()), || Err(1), || fail!()];

        assert_eq!(fold_(functions.mut_iter()
                        .map(|f| (*f)())),
                   Err(1));
    }

    #[test]
    pub fn test_to_str() {
        let ok: Result<int, ~str> = Ok(100);
        let err: Result<int, ~str> = Err("Err".to_owned());

        assert_eq!(ok.to_str(), "Ok(100)".to_owned());
        assert_eq!(err.to_str(), "Err(Err)".to_owned());
    }

    #[test]
    pub fn test_fmt_default() {
        let ok: Result<int, ~str> = Ok(100);
        let err: Result<int, ~str> = Err("Err".to_owned());

        assert_eq!(format!("{}", ok), "Ok(100)".to_owned());
        assert_eq!(format!("{}", err), "Err(Err)".to_owned());
    }

    #[test]
    pub fn test_unwrap_or() {
        let ok: Result<int, ~str> = Ok(100);
        let ok_err: Result<int, ~str> = Err("Err".to_owned());

        assert_eq!(ok.unwrap_or(50), 100);
        assert_eq!(ok_err.unwrap_or(50), 50);
    }

    #[test]
    pub fn test_unwrap_or_else() {
        fn handler(msg: ~str) -> int {
            if msg == "I got this.".to_owned() {
                50
            } else {
                fail!("BadBad")
            }
        }

        let ok: Result<int, ~str> = Ok(100);
        let ok_err: Result<int, ~str> = Err("I got this.".to_owned());

        assert_eq!(ok.unwrap_or_handle(handler), 100);
        assert_eq!(ok_err.unwrap_or_handle(handler), 50);
    }

    #[test]
    #[should_fail]
    pub fn test_unwrap_or_else_failure() {
        fn handler(msg: ~str) -> int {
            if msg == "I got this.".to_owned() {
                50
            } else {
                fail!("BadBad")
            }
        }

        let bad_err: Result<int, ~str> = Err("Unrecoverable mess.".to_owned());
        let _ : int = bad_err.unwrap_or_handle(handler);
    }
}
