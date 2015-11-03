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
//! `Result<T, E>` is the type used for returning and propagating
//! errors. It is an enum with the variants, `Ok(T)`, representing
//! success and containing a value, and `Err(E)`, representing error
//! and containing an error value.
//!
//! ```
//! # #[allow(dead_code)]
//! enum Result<T, E> {
//!    Ok(T),
//!    Err(E)
//! }
//! ```
//!
//! Functions return `Result` whenever errors are expected and
//! recoverable. In the `std` crate `Result` is most prominently used
//! for [I/O](../../std/io/index.html).
//!
//! A simple function returning `Result` might be
//! defined and used like so:
//!
//! ```
//! #[derive(Debug)]
//! enum Version { Version1, Version2 }
//!
//! fn parse_version(header: &[u8]) -> Result<Version, &'static str> {
//!     match header.get(0) {
//!         None => Err("invalid header length"),
//!         Some(&1) => Ok(Version::Version1),
//!         Some(&2) => Ok(Version::Version2),
//!         Some(_) => Err("invalid version")
//!     }
//! }
//!
//! let version = parse_version(&[1, 2, 3, 4]);
//! match version {
//!     Ok(v) => println!("working with version: {:?}", v),
//!     Err(e) => println!("error parsing header: {:?}", e),
//! }
//! ```
//!
//! Pattern matching on `Result`s is clear and straightforward for
//! simple cases, but `Result` comes with some convenience methods
//! that make working with it more succinct.
//!
//! ```
//! let good_result: Result<i32, i32> = Ok(10);
//! let bad_result: Result<i32, i32> = Err(10);
//!
//! // The `is_ok` and `is_err` methods do what they say.
//! assert!(good_result.is_ok() && !good_result.is_err());
//! assert!(bad_result.is_err() && !bad_result.is_ok());
//!
//! // `map` consumes the `Result` and produces another.
//! let good_result: Result<i32, i32> = good_result.map(|i| i + 1);
//! let bad_result: Result<i32, i32> = bad_result.map(|i| i - 1);
//!
//! // Use `and_then` to continue the computation.
//! let good_result: Result<bool, i32> = good_result.and_then(|i| Ok(i == 11));
//!
//! // Use `or_else` to handle the error.
//! let bad_result: Result<i32, i32> = bad_result.or_else(|i| Ok(i + 20));
//!
//! // Consume the result and return the contents with `unwrap`.
//! let final_awesome_result = good_result.unwrap();
//! ```
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
//! Consider the `write_all` method defined for I/O types
//! by the [`Write`](../../std/io/trait.Write.html) trait:
//!
//! ```
//! use std::io;
//!
//! trait Write {
//!     fn write_all(&mut self, bytes: &[u8]) -> Result<(), io::Error>;
//! }
//! ```
//!
//! *Note: The actual definition of `Write` uses `io::Result`, which
//! is just a synonym for `Result<T, io::Error>`.*
//!
//! This method doesn't produce a value, but the write may
//! fail. It's crucial to handle the error case, and *not* write
//! something like this:
//!
//! ```no_run
//! # #![allow(unused_must_use)] // \o/
//! use std::fs::File;
//! use std::io::prelude::*;
//!
//! let mut file = File::create("valuable_data.txt").unwrap();
//! // If `write_all` errors, then we'll never know, because the return
//! // value is ignored.
//! file.write_all(b"important message");
//! ```
//!
//! If you *do* write that in Rust, the compiler will give you a
//! warning (by default, controlled by the `unused_must_use` lint).
//!
//! You might instead, if you don't want to handle the error, simply
//! assert success with `expect`. This will panic if the
//! write fails, providing a marginally useful message indicating why:
//!
//! ```{.no_run}
//! use std::fs::File;
//! use std::io::prelude::*;
//!
//! let mut file = File::create("valuable_data.txt").unwrap();
//! file.write_all(b"important message").expect("failed to write message");
//! ```
//!
//! You might also simply assert success:
//!
//! ```{.no_run}
//! # use std::fs::File;
//! # use std::io::prelude::*;
//! # let mut file = File::create("valuable_data.txt").unwrap();
//! assert!(file.write_all(b"important message").is_ok());
//! ```
//!
//! Or propagate the error up the call stack with `try!`:
//!
//! ```
//! # use std::fs::File;
//! # use std::io::prelude::*;
//! # use std::io;
//! # #[allow(dead_code)]
//! fn write_message() -> io::Result<()> {
//!     let mut file = try!(File::create("valuable_data.txt"));
//!     try!(file.write_all(b"important message"));
//!     Ok(())
//! }
//! ```
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
//! ```
//! # #![allow(dead_code)]
//! use std::fs::File;
//! use std::io::prelude::*;
//! use std::io;
//!
//! struct Info {
//!     name: String,
//!     age: i32,
//!     rating: i32,
//! }
//!
//! fn write_info(info: &Info) -> io::Result<()> {
//!     let mut file = try!(File::create("my_best_friends.txt"));
//!     // Early return on error
//!     if let Err(e) = file.write_all(format!("name: {}\n", info.name).as_bytes()) {
//!         return Err(e)
//!     }
//!     if let Err(e) = file.write_all(format!("age: {}\n", info.age).as_bytes()) {
//!         return Err(e)
//!     }
//!     if let Err(e) = file.write_all(format!("rating: {}\n", info.rating).as_bytes()) {
//!         return Err(e)
//!     }
//!     Ok(())
//! }
//! ```
//!
//! With this:
//!
//! ```
//! # #![allow(dead_code)]
//! use std::fs::File;
//! use std::io::prelude::*;
//! use std::io;
//!
//! struct Info {
//!     name: String,
//!     age: i32,
//!     rating: i32,
//! }
//!
//! fn write_info(info: &Info) -> io::Result<()> {
//!     let mut file = try!(File::create("my_best_friends.txt"));
//!     // Early return on error
//!     try!(file.write_all(format!("name: {}\n", info.name).as_bytes()));
//!     try!(file.write_all(format!("age: {}\n", info.age).as_bytes()));
//!     try!(file.write_all(format!("rating: {}\n", info.rating).as_bytes()));
//!     Ok(())
//! }
//! ```
//!
//! *It's much nicer!*
//!
//! Wrapping an expression in `try!` will result in the unwrapped
//! success (`Ok`) value, unless the result is `Err`, in which case
//! `Err` is returned early from the enclosing function. Its simple definition
//! makes it clear:
//!
//! ```
//! macro_rules! try {
//!     ($e:expr) => (match $e { Ok(e) => e, Err(e) => return Err(e) })
//! }
//! ```
//!
//! `try!` is imported by the prelude and is available everywhere, but it can only
//! be used in functions that return `Result` because of the early return of
//! `Err` that it provides.

#![stable(feature = "rust1", since = "1.0.0")]

use self::Result::{Ok, Err};

use clone::Clone;
use fmt;
use iter::{Iterator, DoubleEndedIterator, FromIterator, ExactSizeIterator, IntoIterator};
use ops::FnOnce;
use option::Option::{self, None, Some};
use slice;

/// `Result` is a type that represents either success (`Ok`) or failure (`Err`).
///
/// See the [`std::result`](index.html) module documentation for details.
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Result<T, E> {
    /// Contains the success value
    #[stable(feature = "rust1", since = "1.0.0")]
    Ok(T),

    /// Contains the error value
    #[stable(feature = "rust1", since = "1.0.0")]
    Err(E)
}

/////////////////////////////////////////////////////////////////////////////
// Type implementation
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, E> Result<T, E> {
    /////////////////////////////////////////////////////////////////////////
    // Querying the contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns true if the result is `Ok`
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<i32, &str> = Ok(-3);
    /// assert_eq!(x.is_ok(), true);
    ///
    /// let x: Result<i32, &str> = Err("Some error message");
    /// assert_eq!(x.is_ok(), false);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_ok(&self) -> bool {
        match *self {
            Ok(_) => true,
            Err(_) => false
        }
    }

    /// Returns true if the result is `Err`
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<i32, &str> = Ok(-3);
    /// assert_eq!(x.is_err(), false);
    ///
    /// let x: Result<i32, &str> = Err("Some error message");
    /// assert_eq!(x.is_err(), true);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for each variant
    /////////////////////////////////////////////////////////////////////////

    /// Converts from `Result<T, E>` to `Option<T>`
    ///
    /// Converts `self` into an `Option<T>`, consuming `self`,
    /// and discarding the error, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// assert_eq!(x.ok(), Some(2));
    ///
    /// let x: Result<u32, &str> = Err("Nothing here");
    /// assert_eq!(x.ok(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ok(self) -> Option<T> {
        match self {
            Ok(x)  => Some(x),
            Err(_) => None,
        }
    }

    /// Converts from `Result<T, E>` to `Option<E>`
    ///
    /// Converts `self` into an `Option<E>`, consuming `self`,
    /// and discarding the success value, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// assert_eq!(x.err(), None);
    ///
    /// let x: Result<u32, &str> = Err("Nothing here");
    /// assert_eq!(x.err(), Some("Nothing here"));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn err(self) -> Option<E> {
        match self {
            Ok(_)  => None,
            Err(x) => Some(x),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Converts from `Result<T, E>` to `Result<&T, &E>`
    ///
    /// Produces a new `Result`, containing a reference
    /// into the original, leaving the original in place.
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// assert_eq!(x.as_ref(), Ok(&2));
    ///
    /// let x: Result<u32, &str> = Err("Error");
    /// assert_eq!(x.as_ref(), Err(&"Error"));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_ref(&self) -> Result<&T, &E> {
        match *self {
            Ok(ref x) => Ok(x),
            Err(ref x) => Err(x),
        }
    }

    /// Converts from `Result<T, E>` to `Result<&mut T, &mut E>`
    ///
    /// ```
    /// fn mutate(r: &mut Result<i32, i32>) {
    ///     match r.as_mut() {
    ///         Ok(&mut ref mut v) => *v = 42,
    ///         Err(&mut ref mut e) => *e = 0,
    ///     }
    /// }
    ///
    /// let mut x: Result<i32, i32> = Ok(2);
    /// mutate(&mut x);
    /// assert_eq!(x.unwrap(), 42);
    ///
    /// let mut x: Result<i32, i32> = Err(13);
    /// mutate(&mut x);
    /// assert_eq!(x.unwrap_err(), 0);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_mut(&mut self) -> Result<&mut T, &mut E> {
        match *self {
            Ok(ref mut x) => Ok(x),
            Err(ref mut x) => Err(x),
        }
    }

    /// Converts from `Result<T, E>` to `&[T]` (without copying)
    #[inline]
    #[unstable(feature = "as_slice", reason = "unsure of the utility here",
               issue = "27776")]
    #[deprecated(since = "1.4.0", reason = "niche API, unclear of usefulness")]
    #[allow(deprecated)]
    pub fn as_slice(&self) -> &[T] {
        match *self {
            Ok(ref x) => slice::ref_slice(x),
            Err(_) => {
                // work around lack of implicit coercion from fixed-size array to slice
                let emp: &[_] = &[];
                emp
            }
        }
    }

    /// Converts from `Result<T, E>` to `&mut [T]` (without copying)
    ///
    /// ```
    /// #![feature(as_slice)]
    /// # #![allow(deprecated)]
    ///
    /// let mut x: Result<&str, u32> = Ok("Gold");
    /// {
    ///     let v = x.as_mut_slice();
    ///     assert!(v == ["Gold"]);
    ///     v[0] = "Silver";
    ///     assert!(v == ["Silver"]);
    /// }
    /// assert_eq!(x, Ok("Silver"));
    ///
    /// let mut x: Result<&str, u32> = Err(45);
    /// assert!(x.as_mut_slice().is_empty());
    /// ```
    #[inline]
    #[unstable(feature = "as_slice",
               reason = "waiting for mut conventions",
               issue = "27776")]
    #[deprecated(since = "1.4.0", reason = "niche API, unclear of usefulness")]
    #[allow(deprecated)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match *self {
            Ok(ref mut x) => slice::mut_ref_slice(x),
            Err(_) => {
                // work around lack of implicit coercion from fixed-size array to slice
                let emp: &mut [_] = &mut [];
                emp
            }
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
    /// Print the numbers on each line of a string multiplied by two.
    ///
    /// ```
    /// let line = "1\n2\n3\n4\n";
    ///
    /// for num in line.lines() {
    ///     match num.parse::<i32>().map(|i| i * 2) {
    ///         Ok(n) => println!("{}", n),
    ///         Err(..) => {}
    ///     }
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map<U, F: FnOnce(T) -> U>(self, op: F) -> Result<U,E> {
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
    ///
    /// # Examples
    ///
    /// ```
    /// fn stringify(x: u32) -> String { format!("error code: {}", x) }
    ///
    /// let x: Result<u32, u32> = Ok(2);
    /// assert_eq!(x.map_err(stringify), Ok(2));
    ///
    /// let x: Result<u32, u32> = Err(13);
    /// assert_eq!(x.map_err(stringify), Err("error code: 13".to_string()));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map_err<F, O: FnOnce(E) -> F>(self, op: O) -> Result<T,F> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(op(e))
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Iterator constructors
    /////////////////////////////////////////////////////////////////////////

    /// Returns an iterator over the possibly contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(7);
    /// assert_eq!(x.iter().next(), Some(&7));
    ///
    /// let x: Result<u32, &str> = Err("nothing!");
    /// assert_eq!(x.iter().next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter { inner: self.as_ref().ok() }
    }

    /// Returns a mutable iterator over the possibly contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x: Result<u32, &str> = Ok(7);
    /// match x.iter_mut().next() {
    ///     Some(v) => *v = 40,
    ///     None => {},
    /// }
    /// assert_eq!(x, Ok(40));
    ///
    /// let mut x: Result<u32, &str> = Err("nothing!");
    /// assert_eq!(x.iter_mut().next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut { inner: self.as_mut().ok() }
    }

    ////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns `res` if the result is `Ok`, otherwise returns the `Err` value of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// let y: Result<&str, &str> = Err("late error");
    /// assert_eq!(x.and(y), Err("late error"));
    ///
    /// let x: Result<u32, &str> = Err("early error");
    /// let y: Result<&str, &str> = Ok("foo");
    /// assert_eq!(x.and(y), Err("early error"));
    ///
    /// let x: Result<u32, &str> = Err("not a 2");
    /// let y: Result<&str, &str> = Err("late error");
    /// assert_eq!(x.and(y), Err("not a 2"));
    ///
    /// let x: Result<u32, &str> = Ok(2);
    /// let y: Result<&str, &str> = Ok("different result type");
    /// assert_eq!(x.and(y), Ok("different result type"));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn and<U>(self, res: Result<U, E>) -> Result<U, E> {
        match self {
            Ok(_) => res,
            Err(e) => Err(e),
        }
    }

    /// Calls `op` if the result is `Ok`, otherwise returns the `Err` value of `self`.
    ///
    /// This function can be used for control flow based on result values.
    ///
    /// # Examples
    ///
    /// ```
    /// fn sq(x: u32) -> Result<u32, u32> { Ok(x * x) }
    /// fn err(x: u32) -> Result<u32, u32> { Err(x) }
    ///
    /// assert_eq!(Ok(2).and_then(sq).and_then(sq), Ok(16));
    /// assert_eq!(Ok(2).and_then(sq).and_then(err), Err(4));
    /// assert_eq!(Ok(2).and_then(err).and_then(sq), Err(2));
    /// assert_eq!(Err(3).and_then(sq).and_then(sq), Err(3));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn and_then<U, F: FnOnce(T) -> Result<U, E>>(self, op: F) -> Result<U, E> {
        match self {
            Ok(t) => op(t),
            Err(e) => Err(e),
        }
    }

    /// Returns `res` if the result is `Err`, otherwise returns the `Ok` value of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// let y: Result<u32, &str> = Err("late error");
    /// assert_eq!(x.or(y), Ok(2));
    ///
    /// let x: Result<u32, &str> = Err("early error");
    /// let y: Result<u32, &str> = Ok(2);
    /// assert_eq!(x.or(y), Ok(2));
    ///
    /// let x: Result<u32, &str> = Err("not a 2");
    /// let y: Result<u32, &str> = Err("late error");
    /// assert_eq!(x.or(y), Err("late error"));
    ///
    /// let x: Result<u32, &str> = Ok(2);
    /// let y: Result<u32, &str> = Ok(100);
    /// assert_eq!(x.or(y), Ok(2));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn or<F>(self, res: Result<T, F>) -> Result<T, F> {
        match self {
            Ok(v) => Ok(v),
            Err(_) => res,
        }
    }

    /// Calls `op` if the result is `Err`, otherwise returns the `Ok` value of `self`.
    ///
    /// This function can be used for control flow based on result values.
    ///
    /// # Examples
    ///
    /// ```
    /// fn sq(x: u32) -> Result<u32, u32> { Ok(x * x) }
    /// fn err(x: u32) -> Result<u32, u32> { Err(x) }
    ///
    /// assert_eq!(Ok(2).or_else(sq).or_else(sq), Ok(2));
    /// assert_eq!(Ok(2).or_else(err).or_else(sq), Ok(2));
    /// assert_eq!(Err(3).or_else(sq).or_else(err), Ok(9));
    /// assert_eq!(Err(3).or_else(err).or_else(err), Err(3));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn or_else<F, O: FnOnce(E) -> Result<T, F>>(self, op: O) -> Result<T, F> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => op(e),
        }
    }

    /// Unwraps a result, yielding the content of an `Ok`.
    /// Else it returns `optb`.
    ///
    /// # Examples
    ///
    /// ```
    /// let optb = 2;
    /// let x: Result<u32, &str> = Ok(9);
    /// assert_eq!(x.unwrap_or(optb), 9);
    ///
    /// let x: Result<u32, &str> = Err("error");
    /// assert_eq!(x.unwrap_or(optb), optb);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or(self, optb: T) -> T {
        match self {
            Ok(t) => t,
            Err(_) => optb
        }
    }

    /// Unwraps a result, yielding the content of an `Ok`.
    /// If the value is an `Err` then it calls `op` with its value.
    ///
    /// # Examples
    ///
    /// ```
    /// fn count(x: &str) -> usize { x.len() }
    ///
    /// assert_eq!(Ok(2).unwrap_or_else(count), 2);
    /// assert_eq!(Err("foo").unwrap_or_else(count), 3);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or_else<F: FnOnce(E) -> T>(self, op: F) -> T {
        match self {
            Ok(t) => t,
            Err(e) => op(e)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, E: fmt::Debug> Result<T, E> {
    /// Unwraps a result, yielding the content of an `Ok`.
    ///
    /// # Panics
    ///
    /// Panics if the value is an `Err`, with a panic message provided by the
    /// `Err`'s value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// assert_eq!(x.unwrap(), 2);
    /// ```
    ///
    /// ```{.should_panic}
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// x.unwrap(); // panics with `emergency failure`
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap(self) -> T {
        match self {
            Ok(t) => t,
            Err(e) =>
                panic!("called `Result::unwrap()` on an `Err` value: {:?}", e)
        }
    }

    /// Unwraps a result, yielding the content of an `Ok`.
    ///
    /// Panics if the value is an `Err`, with a panic message including the
    /// passed message, and the content of the `Err`.
    ///
    /// # Examples
    /// ```{.should_panic}
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// x.expect("Testing expect"); // panics with `Testing expect: emergency failure`
    /// ```
    #[inline]
    #[stable(feature = "result_expect", since = "1.4.0")]
    pub fn expect(self, msg: &str) -> T {
        match self {
            Ok(t) => t,
            Err(e) => panic!("{}: {:?}", msg, e),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug, E> Result<T, E> {
    /// Unwraps a result, yielding the content of an `Err`.
    ///
    /// # Panics
    ///
    /// Panics if the value is an `Ok`, with a custom panic message provided
    /// by the `Ok`'s value.
    ///
    /// # Examples
    ///
    /// ```{.should_panic}
    /// let x: Result<u32, &str> = Ok(2);
    /// x.unwrap_err(); // panics with `2`
    /// ```
    ///
    /// ```
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// assert_eq!(x.unwrap_err(), "emergency failure");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_err(self) -> E {
        match self {
            Ok(t) =>
                panic!("called `Result::unwrap_err()` on an `Ok` value: {:?}", t),
            Err(e) => e
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Trait implementations
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, E> IntoIterator for Result<T, E> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Returns a consuming iterator over the possibly contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(5);
    /// let v: Vec<u32> = x.into_iter().collect();
    /// assert_eq!(v, [5]);
    ///
    /// let x: Result<u32, &str> = Err("nothing!");
    /// let v: Vec<u32> = x.into_iter().collect();
    /// assert_eq!(v, []);
    /// ```
    #[inline]
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { inner: self.ok() }
    }
}

#[stable(since = "1.4.0", feature = "result_iter")]
impl<'a, T, E> IntoIterator for &'a Result<T, E> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(since = "1.4.0", feature = "result_iter")]
impl<'a, T, E> IntoIterator for &'a mut Result<T, E> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(mut self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

/////////////////////////////////////////////////////////////////////////////
// The Result Iterators
/////////////////////////////////////////////////////////////////////////////

/// An iterator over a reference to the `Ok` variant of a `Result`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> { inner: Option<&'a T> }

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> { self.inner.take() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() {1} else {0};
        (n, Some(n))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> { self.inner.take() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> { Iter { inner: self.inner } }
}

/// An iterator over a mutable reference to the `Ok` variant of a `Result`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> { inner: Option<&'a mut T> }

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> { self.inner.take() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() {1} else {0};
        (n, Some(n))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> { self.inner.take() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for IterMut<'a, T> {}

/// An iterator over the value in a `Ok` variant of a `Result`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> { inner: Option<T> }

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> { self.inner.take() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() {1} else {0};
        (n, Some(n))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> { self.inner.take() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {}

/////////////////////////////////////////////////////////////////////////////
// FromIterator
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, E, V: FromIterator<A>> FromIterator<Result<A, E>> for Result<V, E> {
    /// Takes each element in the `Iterator`: if it is an `Err`, no further
    /// elements are taken, and the `Err` is returned. Should no `Err` occur, a
    /// container with the values of each `Result` is returned.
    ///
    /// Here is an example which increments every integer in a vector,
    /// checking for overflow:
    ///
    /// ```
    /// use std::u32;
    ///
    /// let v = vec!(1, 2);
    /// let res: Result<Vec<u32>, &'static str> = v.iter().map(|&x: &u32|
    ///     if x == u32::MAX { Err("Overflow!") }
    ///     else { Ok(x + 1) }
    /// ).collect();
    /// assert!(res == Ok(vec!(2, 3)));
    /// ```
    #[inline]
    fn from_iter<I: IntoIterator<Item=Result<A, E>>>(iter: I) -> Result<V, E> {
        // FIXME(#11084): This could be replaced with Iterator::scan when this
        // performance bug is closed.

        struct Adapter<Iter, E> {
            iter: Iter,
            err: Option<E>,
        }

        impl<T, E, Iter: Iterator<Item=Result<T, E>>> Iterator for Adapter<Iter, E> {
            type Item = T;

            #[inline]
            fn next(&mut self) -> Option<T> {
                match self.iter.next() {
                    Some(Ok(value)) => Some(value),
                    Some(Err(err)) => {
                        self.err = Some(err);
                        None
                    }
                    None => None,
                }
            }
        }

        let mut adapter = Adapter { iter: iter.into_iter(), err: None };
        let v: V = FromIterator::from_iter(adapter.by_ref());

        match adapter.err {
            Some(err) => Err(err),
            None => Ok(v),
        }
    }
}
