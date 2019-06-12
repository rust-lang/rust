//! Error handling with the `Result` type.
//!
//! [`Result<T, E>`][`Result`] is the type used for returning and propagating
//! errors. It is an enum with the variants, [`Ok(T)`], representing
//! success and containing a value, and [`Err(E)`], representing error
//! and containing an error value.
//!
//! ```
//! # #[allow(dead_code)]
//! enum Result<T, E> {
//!    Ok(T),
//!    Err(E),
//! }
//! ```
//!
//! Functions return [`Result`] whenever errors are expected and
//! recoverable. In the `std` crate, [`Result`] is most prominently used
//! for [I/O](../../std/io/index.html).
//!
//! A simple function returning [`Result`] might be
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
//!         Some(_) => Err("invalid version"),
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
//! Pattern matching on [`Result`]s is clear and straightforward for
//! simple cases, but [`Result`] comes with some convenience methods
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
//! the error. [`Result`] is annotated with the `#[must_use]` attribute,
//! which will cause the compiler to issue a warning when a Result
//! value is ignored. This makes [`Result`] especially useful with
//! functions that may encounter errors but don't otherwise return a
//! useful value.
//!
//! Consider the [`write_all`] method defined for I/O types
//! by the [`Write`] trait:
//!
//! ```
//! use std::io;
//!
//! trait Write {
//!     fn write_all(&mut self, bytes: &[u8]) -> Result<(), io::Error>;
//! }
//! ```
//!
//! *Note: The actual definition of [`Write`] uses [`io::Result`], which
//! is just a synonym for [`Result`]`<T, `[`io::Error`]`>`.*
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
//! assert success with [`expect`]. This will panic if the
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
//! Or propagate the error up the call stack with [`?`]:
//!
//! ```
//! # use std::fs::File;
//! # use std::io::prelude::*;
//! # use std::io;
//! # #[allow(dead_code)]
//! fn write_message() -> io::Result<()> {
//!     let mut file = File::create("valuable_data.txt")?;
//!     file.write_all(b"important message")?;
//!     Ok(())
//! }
//! ```
//!
//! # The question mark operator, `?`
//!
//! When writing code that calls many functions that return the
//! [`Result`] type, the error handling can be tedious. The question mark
//! operator, [`?`], hides some of the boilerplate of propagating errors
//! up the call stack.
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
//!     // Early return on error
//!     let mut file = match File::create("my_best_friends.txt") {
//!            Err(e) => return Err(e),
//!            Ok(f) => f,
//!     };
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
//!     let mut file = File::create("my_best_friends.txt")?;
//!     // Early return on error
//!     file.write_all(format!("name: {}\n", info.name).as_bytes())?;
//!     file.write_all(format!("age: {}\n", info.age).as_bytes())?;
//!     file.write_all(format!("rating: {}\n", info.rating).as_bytes())?;
//!     Ok(())
//! }
//! ```
//!
//! *It's much nicer!*
//!
//! Ending the expression with [`?`] will result in the unwrapped
//! success ([`Ok`]) value, unless the result is [`Err`], in which case
//! [`Err`] is returned early from the enclosing function.
//!
//! [`?`] can only be used in functions that return [`Result`] because of the
//! early return of [`Err`] that it provides.
//!
//! [`expect`]: enum.Result.html#method.expect
//! [`Write`]: ../../std/io/trait.Write.html
//! [`write_all`]: ../../std/io/trait.Write.html#method.write_all
//! [`io::Result`]: ../../std/io/type.Result.html
//! [`?`]: ../../std/macro.try.html
//! [`Result`]: enum.Result.html
//! [`Ok(T)`]: enum.Result.html#variant.Ok
//! [`Err(E)`]: enum.Result.html#variant.Err
//! [`io::Error`]: ../../std/io/struct.Error.html
//! [`Ok`]: enum.Result.html#variant.Ok
//! [`Err`]: enum.Result.html#variant.Err

#![stable(feature = "rust1", since = "1.0.0")]

use crate::fmt;
use crate::iter::{FromIterator, FusedIterator, TrustedLen};
use crate::ops::{self, Deref};

/// `Result` is a type that represents either success ([`Ok`]) or failure ([`Err`]).
///
/// See the [`std::result`](index.html) module documentation for details.
///
/// [`Ok`]: enum.Result.html#variant.Ok
/// [`Err`]: enum.Result.html#variant.Err
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[must_use = "this `Result` may be an `Err` variant, which should be handled"]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Result<T, E> {
    /// Contains the success value
    #[stable(feature = "rust1", since = "1.0.0")]
    Ok(#[stable(feature = "rust1", since = "1.0.0")] T),

    /// Contains the error value
    #[stable(feature = "rust1", since = "1.0.0")]
    Err(#[stable(feature = "rust1", since = "1.0.0")] E),
}

/////////////////////////////////////////////////////////////////////////////
// Type implementation
/////////////////////////////////////////////////////////////////////////////

impl<T, E> Result<T, E> {
    /////////////////////////////////////////////////////////////////////////
    // Querying the contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns `true` if the result is [`Ok`].
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let x: Result<i32, &str> = Ok(-3);
    /// assert_eq!(x.is_ok(), true);
    ///
    /// let x: Result<i32, &str> = Err("Some error message");
    /// assert_eq!(x.is_ok(), false);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_ok(&self) -> bool {
        match *self {
            Ok(_) => true,
            Err(_) => false
        }
    }

    /// Returns `true` if the result is [`Err`].
    ///
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let x: Result<i32, &str> = Ok(-3);
    /// assert_eq!(x.is_err(), false);
    ///
    /// let x: Result<i32, &str> = Err("Some error message");
    /// assert_eq!(x.is_err(), true);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for each variant
    /////////////////////////////////////////////////////////////////////////

    /// Converts from `Result<T, E>` to [`Option<T>`].
    ///
    /// Converts `self` into an [`Option<T>`], consuming `self`,
    /// and discarding the error, if any.
    ///
    /// [`Option<T>`]: ../../std/option/enum.Option.html
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Converts from `Result<T, E>` to [`Option<E>`].
    ///
    /// Converts `self` into an [`Option<E>`], consuming `self`,
    /// and discarding the success value, if any.
    ///
    /// [`Option<E>`]: ../../std/option/enum.Option.html
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Converts from `&Result<T, E>` to `Result<&T, &E>`.
    ///
    /// Produces a new `Result`, containing a reference
    /// into the original, leaving the original in place.
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Converts from `&mut Result<T, E>` to `Result<&mut T, &mut E>`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// fn mutate(r: &mut Result<i32, i32>) {
    ///     match r.as_mut() {
    ///         Ok(v) => *v = 42,
    ///         Err(e) => *e = 0,
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

    /////////////////////////////////////////////////////////////////////////
    // Transforming contained values
    /////////////////////////////////////////////////////////////////////////

    /// Maps a `Result<T, E>` to `Result<U, E>` by applying a function to a
    /// contained [`Ok`] value, leaving an [`Err`] value untouched.
    ///
    /// This function can be used to compose the results of two functions.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
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

    /// Maps a `Result<T, E>` to `U` by applying a function to a
    /// contained [`Ok`] value, or a fallback function to a
    /// contained [`Err`] value.
    ///
    /// This function can be used to unpack a successful result
    /// while handling an error.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(result_map_or_else)]
    /// let k = 21;
    ///
    /// let x : Result<_, &str> = Ok("foo");
    /// assert_eq!(x.map_or_else(|e| k * 2, |v| v.len()), 3);
    ///
    /// let x : Result<&str, _> = Err("bar");
    /// assert_eq!(x.map_or_else(|e| k * 2, |v| v.len()), 42);
    /// ```
    #[inline]
    #[unstable(feature = "result_map_or_else", issue = "53268")]
    pub fn map_or_else<U, M: FnOnce(T) -> U, F: FnOnce(E) -> U>(self, fallback: F, map: M) -> U {
        self.map(map).unwrap_or_else(fallback)
    }

    /// Maps a `Result<T, E>` to `Result<T, F>` by applying a function to a
    /// contained [`Err`] value, leaving an [`Ok`] value untouched.
    ///
    /// This function can be used to pass through a successful result while handling
    /// an error.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
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
    /// The iterator yields one value if the result is [`Result::Ok`], otherwise none.
    ///
    /// # Examples
    ///
    /// Basic usage:
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
    pub fn iter(&self) -> Iter<'_, T> {
        Iter { inner: self.as_ref().ok() }
    }

    /// Returns a mutable iterator over the possibly contained value.
    ///
    /// The iterator yields one value if the result is [`Result::Ok`], otherwise none.
    ///
    /// # Examples
    ///
    /// Basic usage:
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
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut { inner: self.as_mut().ok() }
    }

    ////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns `res` if the result is [`Ok`], otherwise returns the [`Err`] value of `self`.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Calls `op` if the result is [`Ok`], otherwise returns the [`Err`] value of `self`.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// This function can be used for control flow based on `Result` values.
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Returns `res` if the result is [`Err`], otherwise returns the [`Ok`] value of `self`.
    ///
    /// Arguments passed to `or` are eagerly evaluated; if you are passing the
    /// result of a function call, it is recommended to use [`or_else`], which is
    /// lazily evaluated.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    /// [`or_else`]: #method.or_else
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Calls `op` if the result is [`Err`], otherwise returns the [`Ok`] value of `self`.
    ///
    /// This function can be used for control flow based on result values.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Unwraps a result, yielding the content of an [`Ok`].
    /// Else, it returns `optb`.
    ///
    /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`unwrap_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    /// [`unwrap_or_else`]: #method.unwrap_or_else
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    /// Unwraps a result, yielding the content of an [`Ok`].
    /// If the value is an [`Err`] then it calls `op` with its value.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
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

impl<T, E: fmt::Debug> Result<T, E> {
    /// Unwraps a result, yielding the content of an [`Ok`].
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Err`], with a panic message provided by the
    /// [`Err`]'s value.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
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
            Err(e) => unwrap_failed("called `Result::unwrap()` on an `Err` value", e),
        }
    }

    /// Unwraps a result, yielding the content of an [`Ok`].
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Err`], with a panic message including the
    /// passed message, and the content of the [`Err`].
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```{.should_panic}
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// x.expect("Testing expect"); // panics with `Testing expect: emergency failure`
    /// ```
    #[inline]
    #[stable(feature = "result_expect", since = "1.4.0")]
    pub fn expect(self, msg: &str) -> T {
        match self {
            Ok(t) => t,
            Err(e) => unwrap_failed(msg, e),
        }
    }
}

impl<T: fmt::Debug, E> Result<T, E> {
    /// Unwraps a result, yielding the content of an [`Err`].
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Ok`], with a custom panic message provided
    /// by the [`Ok`]'s value.
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
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
            Ok(t) => unwrap_failed("called `Result::unwrap_err()` on an `Ok` value", t),
            Err(e) => e,
        }
    }

    /// Unwraps a result, yielding the content of an [`Err`].
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Ok`], with a panic message including the
    /// passed message, and the content of the [`Ok`].
    ///
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```{.should_panic}
    /// let x: Result<u32, &str> = Ok(10);
    /// x.expect_err("Testing expect_err"); // panics with `Testing expect_err: 10`
    /// ```
    #[inline]
    #[stable(feature = "result_expect_err", since = "1.17.0")]
    pub fn expect_err(self, msg: &str) -> E {
        match self {
            Ok(t) => unwrap_failed(msg, t),
            Err(e) => e,
        }
    }
}

impl<T: Default, E> Result<T, E> {
    /// Returns the contained value or a default
    ///
    /// Consumes the `self` argument then, if [`Ok`], returns the contained
    /// value, otherwise if [`Err`], returns the default value for that
    /// type.
    ///
    /// # Examples
    ///
    /// Converts a string to an integer, turning poorly-formed strings
    /// into 0 (the default value for integers). [`parse`] converts
    /// a string to any other type that implements [`FromStr`], returning an
    /// [`Err`] on error.
    ///
    /// ```
    /// let good_year_from_input = "1909";
    /// let bad_year_from_input = "190blarg";
    /// let good_year = good_year_from_input.parse().unwrap_or_default();
    /// let bad_year = bad_year_from_input.parse().unwrap_or_default();
    ///
    /// assert_eq!(1909, good_year);
    /// assert_eq!(0, bad_year);
    /// ```
    ///
    /// [`parse`]: ../../std/primitive.str.html#method.parse
    /// [`FromStr`]: ../../std/str/trait.FromStr.html
    /// [`Ok`]: enum.Result.html#variant.Ok
    /// [`Err`]: enum.Result.html#variant.Err
    #[inline]
    #[stable(feature = "result_unwrap_or_default", since = "1.16.0")]
    pub fn unwrap_or_default(self) -> T {
        match self {
            Ok(x) => x,
            Err(_) => Default::default(),
        }
    }
}

#[unstable(feature = "inner_deref", reason = "newly added", issue = "50264")]
impl<T: Deref, E> Result<T, E> {
    /// Converts from `&Result<T, E>` to `Result<&T::Target, &E>`.
    ///
    /// Leaves the original Result in-place, creating a new one with a reference
    /// to the original one, additionally coercing the `Ok` arm of the Result via
    /// `Deref`.
    pub fn deref_ok(&self) -> Result<&T::Target, &E> {
        self.as_ref().map(|t| t.deref())
    }
}

#[unstable(feature = "inner_deref", reason = "newly added", issue = "50264")]
impl<T, E: Deref> Result<T, E> {
    /// Converts from `&Result<T, E>` to `Result<&T, &E::Target>`.
    ///
    /// Leaves the original Result in-place, creating a new one with a reference
    /// to the original one, additionally coercing the `Err` arm of the Result via
    /// `Deref`.
    pub fn deref_err(&self) -> Result<&T, &E::Target>
    {
        self.as_ref().map_err(|e| e.deref())
    }
}

#[unstable(feature = "inner_deref", reason = "newly added", issue = "50264")]
impl<T: Deref, E: Deref> Result<T, E> {
    /// Converts from `&Result<T, E>` to `Result<&T::Target, &E::Target>`.
    ///
    /// Leaves the original Result in-place, creating a new one with a reference
    /// to the original one, additionally coercing both the `Ok` and `Err` arms
    /// of the Result via `Deref`.
    pub fn deref(&self) -> Result<&T::Target, &E::Target>
    {
        self.as_ref().map(|t| t.deref()).map_err(|e| e.deref())
    }
}

impl<T, E> Result<Option<T>, E> {
    /// Transposes a `Result` of an `Option` into an `Option` of a `Result`.
    ///
    /// `Ok(None)` will be mapped to `None`.
    /// `Ok(Some(_))` and `Err(_)` will be mapped to `Some(Ok(_))` and `Some(Err(_))`.
    ///
    /// # Examples
    ///
    /// ```
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct SomeErr;
    ///
    /// let x: Result<Option<i32>, SomeErr> = Ok(Some(5));
    /// let y: Option<Result<i32, SomeErr>> = Some(Ok(5));
    /// assert_eq!(x.transpose(), y);
    /// ```
    #[inline]
    #[stable(feature = "transpose_result", since = "1.33.0")]
    pub fn transpose(self) -> Option<Result<T, E>> {
        match self {
            Ok(Some(x)) => Some(Ok(x)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// This is a separate function to reduce the code size of the methods
#[inline(never)]
#[cold]
fn unwrap_failed<E: fmt::Debug>(msg: &str, error: E) -> ! {
    panic!("{}: {:?}", msg, error)
}

/////////////////////////////////////////////////////////////////////////////
// Trait implementations
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone, E: Clone> Clone for Result<T, E> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Ok(x) => Ok(x.clone()),
            Err(x) => Err(x.clone()),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (Ok(to), Ok(from)) => to.clone_from(from),
            (Err(to), Err(from)) => to.clone_from(from),
            (to, from) => *to = from.clone(),
        }
    }
}


#[stable(feature = "rust1", since = "1.0.0")]
impl<T, E> IntoIterator for Result<T, E> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Returns a consuming iterator over the possibly contained value.
    ///
    /// The iterator yields one value if the result is [`Result::Ok`], otherwise none.
    ///
    /// # Examples
    ///
    /// Basic usage:
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

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

/////////////////////////////////////////////////////////////////////////////
// The Result Iterators
/////////////////////////////////////////////////////////////////////////////

/// An iterator over a reference to the [`Ok`] variant of a [`Result`].
///
/// The iterator yields one value if the result is [`Ok`], otherwise none.
///
/// Created by [`Result::iter`].
///
/// [`Ok`]: enum.Result.html#variant.Ok
/// [`Result`]: enum.Result.html
/// [`Result::iter`]: enum.Result.html#method.iter
#[derive(Debug)]
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
impl<T> ExactSizeIterator for Iter<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Iter<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A> TrustedLen for Iter<'_, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    #[inline]
    fn clone(&self) -> Self { Iter { inner: self.inner } }
}

/// An iterator over a mutable reference to the [`Ok`] variant of a [`Result`].
///
/// Created by [`Result::iter_mut`].
///
/// [`Ok`]: enum.Result.html#variant.Ok
/// [`Result`]: enum.Result.html
/// [`Result::iter_mut`]: enum.Result.html#method.iter_mut
#[derive(Debug)]
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
impl<T> ExactSizeIterator for IterMut<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for IterMut<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A> TrustedLen for IterMut<'_, A> {}

/// An iterator over the value in a [`Ok`] variant of a [`Result`].
///
/// The iterator yields one value if the result is [`Ok`], otherwise none.
///
/// This struct is created by the [`into_iter`] method on
/// [`Result`][`Result`] (provided by the [`IntoIterator`] trait).
///
/// [`Ok`]: enum.Result.html#variant.Ok
/// [`Result`]: enum.Result.html
/// [`into_iter`]: ../iter/trait.IntoIterator.html#tymethod.into_iter
/// [`IntoIterator`]: ../iter/trait.IntoIterator.html
#[derive(Clone, Debug)]
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

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for IntoIter<T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A> TrustedLen for IntoIter<A> {}

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
    /// let v = vec![1, 2];
    /// let res: Result<Vec<u32>, &'static str> = v.iter().map(|x: &u32|
    ///     x.checked_add(1).ok_or("Overflow!")
    /// ).collect();
    /// assert_eq!(res, Ok(vec![2, 3]));
    /// ```
    ///
    /// Here is another example that tries to subtract one from another list
    /// of integers, this time checking for underflow:
    ///
    /// ```
    /// let v = vec![1, 2, 0];
    /// let res: Result<Vec<u32>, &'static str> = v.iter().map(|x: &u32|
    ///     x.checked_sub(1).ok_or("Underflow!")
    /// ).collect();
    /// assert_eq!(res, Err("Underflow!"));
    /// ```
    ///
    /// Here is a variation on the previous example, showing that no
    /// further elements are taken from `iter` after the first `Err`.
    ///
    /// ```
    /// let v = vec![3, 2, 1, 10];
    /// let mut shared = 0;
    /// let res: Result<Vec<u32>, &'static str> = v.iter().map(|x: &u32| {
    ///     shared += x;
    ///     x.checked_sub(2).ok_or("Underflow!")
    /// }).collect();
    /// assert_eq!(res, Err("Underflow!"));
    /// assert_eq!(shared, 6);
    /// ```
    ///
    /// Since the third element caused an underflow, no further elements were taken,
    /// so the final value of `shared` is 6 (= `3 + 2 + 1`), not 16.
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

            fn size_hint(&self) -> (usize, Option<usize>) {
                let (_min, max) = self.iter.size_hint();
                (0, max)
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

#[unstable(feature = "try_trait", issue = "42327")]
impl<T,E> ops::Try for Result<T, E> {
    type Ok = T;
    type Error = E;

    #[inline]
    fn into_result(self) -> Self {
        self
    }

    #[inline]
    fn from_ok(v: T) -> Self {
        Ok(v)
    }

    #[inline]
    fn from_error(v: E) -> Self {
        Err(v)
    }
}
