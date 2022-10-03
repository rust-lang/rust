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
//!     Ok(v) => println!("working with version: {v:?}"),
//!     Err(e) => println!("error parsing header: {e:?}"),
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
//! is just a synonym for <code>[Result]<T, [io::Error]></code>.*
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
//! ```no_run
//! use std::fs::File;
//! use std::io::prelude::*;
//!
//! let mut file = File::create("valuable_data.txt").unwrap();
//! file.write_all(b"important message").expect("failed to write message");
//! ```
//!
//! You might also simply assert success:
//!
//! ```no_run
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
//! [`expect`]: Result::expect
//! [`Write`]: ../../std/io/trait.Write.html "io::Write"
//! [`write_all`]: ../../std/io/trait.Write.html#method.write_all "io::Write::write_all"
//! [`io::Result`]: ../../std/io/type.Result.html "io::Result"
//! [`?`]: core::ops::Try
//! [`Ok(T)`]: Ok
//! [`Err(E)`]: Err
//! [io::Error]: ../../std/io/struct.Error.html "io::Error"
//!
//! # Method overview
//!
//! In addition to working with pattern matching, [`Result`] provides a
//! wide variety of different methods.
//!
//! ## Querying the variant
//!
//! The [`is_ok`] and [`is_err`] methods return [`true`] if the [`Result`]
//! is [`Ok`] or [`Err`], respectively.
//!
//! [`is_err`]: Result::is_err
//! [`is_ok`]: Result::is_ok
//!
//! ## Adapters for working with references
//!
//! * [`as_ref`] converts from `&Result<T, E>` to `Result<&T, &E>`
//! * [`as_mut`] converts from `&mut Result<T, E>` to `Result<&mut T, &mut E>`
//! * [`as_deref`] converts from `&Result<T, E>` to `Result<&T::Target, &E>`
//! * [`as_deref_mut`] converts from `&mut Result<T, E>` to
//!   `Result<&mut T::Target, &mut E>`
//!
//! [`as_deref`]: Result::as_deref
//! [`as_deref_mut`]: Result::as_deref_mut
//! [`as_mut`]: Result::as_mut
//! [`as_ref`]: Result::as_ref
//!
//! ## Extracting contained values
//!
//! These methods extract the contained value in a [`Result<T, E>`] when it
//! is the [`Ok`] variant. If the [`Result`] is [`Err`]:
//!
//! * [`expect`] panics with a provided custom message
//! * [`unwrap`] panics with a generic message
//! * [`unwrap_or`] returns the provided default value
//! * [`unwrap_or_default`] returns the default value of the type `T`
//!   (which must implement the [`Default`] trait)
//! * [`unwrap_or_else`] returns the result of evaluating the provided
//!   function
//!
//! The panicking methods [`expect`] and [`unwrap`] require `E` to
//! implement the [`Debug`] trait.
//!
//! [`Debug`]: crate::fmt::Debug
//! [`expect`]: Result::expect
//! [`unwrap`]: Result::unwrap
//! [`unwrap_or`]: Result::unwrap_or
//! [`unwrap_or_default`]: Result::unwrap_or_default
//! [`unwrap_or_else`]: Result::unwrap_or_else
//!
//! These methods extract the contained value in a [`Result<T, E>`] when it
//! is the [`Err`] variant. They require `T` to implement the [`Debug`]
//! trait. If the [`Result`] is [`Ok`]:
//!
//! * [`expect_err`] panics with a provided custom message
//! * [`unwrap_err`] panics with a generic message
//!
//! [`Debug`]: crate::fmt::Debug
//! [`expect_err`]: Result::expect_err
//! [`unwrap_err`]: Result::unwrap_err
//!
//! ## Transforming contained values
//!
//! These methods transform [`Result`] to [`Option`]:
//!
//! * [`err`][Result::err] transforms [`Result<T, E>`] into [`Option<E>`],
//!   mapping [`Err(e)`] to [`Some(e)`] and [`Ok(v)`] to [`None`]
//! * [`ok`][Result::ok] transforms [`Result<T, E>`] into [`Option<T>`],
//!   mapping [`Ok(v)`] to [`Some(v)`] and [`Err(e)`] to [`None`]
//! * [`transpose`] transposes a [`Result`] of an [`Option`] into an
//!   [`Option`] of a [`Result`]
//!
// Do NOT add link reference definitions for `err` or `ok`, because they
// will generate numerous incorrect URLs for `Err` and `Ok` elsewhere, due
// to case folding.
//!
//! [`Err(e)`]: Err
//! [`Ok(v)`]: Ok
//! [`Some(e)`]: Option::Some
//! [`Some(v)`]: Option::Some
//! [`transpose`]: Result::transpose
//!
//! This method transforms the contained value of the [`Ok`] variant:
//!
//! * [`map`] transforms [`Result<T, E>`] into [`Result<U, E>`] by applying
//!   the provided function to the contained value of [`Ok`] and leaving
//!   [`Err`] values unchanged
//!
//! [`map`]: Result::map
//!
//! This method transforms the contained value of the [`Err`] variant:
//!
//! * [`map_err`] transforms [`Result<T, E>`] into [`Result<T, F>`] by
//!   applying the provided function to the contained value of [`Err`] and
//!   leaving [`Ok`] values unchanged
//!
//! [`map_err`]: Result::map_err
//!
//! These methods transform a [`Result<T, E>`] into a value of a possibly
//! different type `U`:
//!
//! * [`map_or`] applies the provided function to the contained value of
//!   [`Ok`], or returns the provided default value if the [`Result`] is
//!   [`Err`]
//! * [`map_or_else`] applies the provided function to the contained value
//!   of [`Ok`], or applies the provided default fallback function to the
//!   contained value of [`Err`]
//!
//! [`map_or`]: Result::map_or
//! [`map_or_else`]: Result::map_or_else
//!
//! ## Boolean operators
//!
//! These methods treat the [`Result`] as a boolean value, where [`Ok`]
//! acts like [`true`] and [`Err`] acts like [`false`]. There are two
//! categories of these methods: ones that take a [`Result`] as input, and
//! ones that take a function as input (to be lazily evaluated).
//!
//! The [`and`] and [`or`] methods take another [`Result`] as input, and
//! produce a [`Result`] as output. The [`and`] method can produce a
//! [`Result<U, E>`] value having a different inner type `U` than
//! [`Result<T, E>`]. The [`or`] method can produce a [`Result<T, F>`]
//! value having a different error type `F` than [`Result<T, E>`].
//!
//! | method  | self     | input     | output   |
//! |---------|----------|-----------|----------|
//! | [`and`] | `Err(e)` | (ignored) | `Err(e)` |
//! | [`and`] | `Ok(x)`  | `Err(d)`  | `Err(d)` |
//! | [`and`] | `Ok(x)`  | `Ok(y)`   | `Ok(y)`  |
//! | [`or`]  | `Err(e)` | `Err(d)`  | `Err(d)` |
//! | [`or`]  | `Err(e)` | `Ok(y)`   | `Ok(y)`  |
//! | [`or`]  | `Ok(x)`  | (ignored) | `Ok(x)`  |
//!
//! [`and`]: Result::and
//! [`or`]: Result::or
//!
//! The [`and_then`] and [`or_else`] methods take a function as input, and
//! only evaluate the function when they need to produce a new value. The
//! [`and_then`] method can produce a [`Result<U, E>`] value having a
//! different inner type `U` than [`Result<T, E>`]. The [`or_else`] method
//! can produce a [`Result<T, F>`] value having a different error type `F`
//! than [`Result<T, E>`].
//!
//! | method       | self     | function input | function result | output   |
//! |--------------|----------|----------------|-----------------|----------|
//! | [`and_then`] | `Err(e)` | (not provided) | (not evaluated) | `Err(e)` |
//! | [`and_then`] | `Ok(x)`  | `x`            | `Err(d)`        | `Err(d)` |
//! | [`and_then`] | `Ok(x)`  | `x`            | `Ok(y)`         | `Ok(y)`  |
//! | [`or_else`]  | `Err(e)` | `e`            | `Err(d)`        | `Err(d)` |
//! | [`or_else`]  | `Err(e)` | `e`            | `Ok(y)`         | `Ok(y)`  |
//! | [`or_else`]  | `Ok(x)`  | (not provided) | (not evaluated) | `Ok(x)`  |
//!
//! [`and_then`]: Result::and_then
//! [`or_else`]: Result::or_else
//!
//! ## Comparison operators
//!
//! If `T` and `E` both implement [`PartialOrd`] then [`Result<T, E>`] will
//! derive its [`PartialOrd`] implementation.  With this order, an [`Ok`]
//! compares as less than any [`Err`], while two [`Ok`] or two [`Err`]
//! compare as their contained values would in `T` or `E` respectively.  If `T`
//! and `E` both also implement [`Ord`], then so does [`Result<T, E>`].
//!
//! ```
//! assert!(Ok(1) < Err(0));
//! let x: Result<i32, ()> = Ok(0);
//! let y = Ok(1);
//! assert!(x < y);
//! let x: Result<(), i32> = Err(0);
//! let y = Err(1);
//! assert!(x < y);
//! ```
//!
//! ## Iterating over `Result`
//!
//! A [`Result`] can be iterated over. This can be helpful if you need an
//! iterator that is conditionally empty. The iterator will either produce
//! a single value (when the [`Result`] is [`Ok`]), or produce no values
//! (when the [`Result`] is [`Err`]). For example, [`into_iter`] acts like
//! [`once(v)`] if the [`Result`] is [`Ok(v)`], and like [`empty()`] if the
//! [`Result`] is [`Err`].
//!
//! [`Ok(v)`]: Ok
//! [`empty()`]: core::iter::empty
//! [`once(v)`]: core::iter::once
//!
//! Iterators over [`Result<T, E>`] come in three types:
//!
//! * [`into_iter`] consumes the [`Result`] and produces the contained
//!   value
//! * [`iter`] produces an immutable reference of type `&T` to the
//!   contained value
//! * [`iter_mut`] produces a mutable reference of type `&mut T` to the
//!   contained value
//!
//! See [Iterating over `Option`] for examples of how this can be useful.
//!
//! [Iterating over `Option`]: crate::option#iterating-over-option
//! [`into_iter`]: Result::into_iter
//! [`iter`]: Result::iter
//! [`iter_mut`]: Result::iter_mut
//!
//! You might want to use an iterator chain to do multiple instances of an
//! operation that can fail, but would like to ignore failures while
//! continuing to process the successful results. In this example, we take
//! advantage of the iterable nature of [`Result`] to select only the
//! [`Ok`] values using [`flatten`][Iterator::flatten].
//!
//! ```
//! # use std::str::FromStr;
//! let mut results = vec![];
//! let mut errs = vec![];
//! let nums: Vec<_> = ["17", "not a number", "99", "-27", "768"]
//!    .into_iter()
//!    .map(u8::from_str)
//!    // Save clones of the raw `Result` values to inspect
//!    .inspect(|x| results.push(x.clone()))
//!    // Challenge: explain how this captures only the `Err` values
//!    .inspect(|x| errs.extend(x.clone().err()))
//!    .flatten()
//!    .collect();
//! assert_eq!(errs.len(), 3);
//! assert_eq!(nums, [17, 99]);
//! println!("results {results:?}");
//! println!("errs {errs:?}");
//! println!("nums {nums:?}");
//! ```
//!
//! ## Collecting into `Result`
//!
//! [`Result`] implements the [`FromIterator`][impl-FromIterator] trait,
//! which allows an iterator over [`Result`] values to be collected into a
//! [`Result`] of a collection of each contained value of the original
//! [`Result`] values, or [`Err`] if any of the elements was [`Err`].
//!
//! [impl-FromIterator]: Result#impl-FromIterator%3CResult%3CA%2C%20E%3E%3E-for-Result%3CV%2C%20E%3E
//!
//! ```
//! let v = [Ok(2), Ok(4), Err("err!"), Ok(8)];
//! let res: Result<Vec<_>, &str> = v.into_iter().collect();
//! assert_eq!(res, Err("err!"));
//! let v = [Ok(2), Ok(4), Ok(8)];
//! let res: Result<Vec<_>, &str> = v.into_iter().collect();
//! assert_eq!(res, Ok(vec![2, 4, 8]));
//! ```
//!
//! [`Result`] also implements the [`Product`][impl-Product] and
//! [`Sum`][impl-Sum] traits, allowing an iterator over [`Result`] values
//! to provide the [`product`][Iterator::product] and
//! [`sum`][Iterator::sum] methods.
//!
//! [impl-Product]: Result#impl-Product%3CResult%3CU%2C%20E%3E%3E-for-Result%3CT%2C%20E%3E
//! [impl-Sum]: Result#impl-Sum%3CResult%3CU%2C%20E%3E%3E-for-Result%3CT%2C%20E%3E
//!
//! ```
//! let v = [Err("error!"), Ok(1), Ok(2), Ok(3), Err("foo")];
//! let res: Result<i32, &str> = v.into_iter().sum();
//! assert_eq!(res, Err("error!"));
//! let v = [Ok(1), Ok(2), Ok(21)];
//! let res: Result<i32, &str> = v.into_iter().product();
//! assert_eq!(res, Ok(42));
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::result::{IntoIter, Iter, IterMut, Result};

use crate::borrow::ToOwned;

impl<T: ?Sized, E> Result<&T, E> {
    /// Maps an `Result<&T>` to an `Result<T::Owned>` by cloning the contents of the
    /// result.
    ///
    /// In difference with [`cloned`], this method uses [`ToOwned`] trait, which allows
    /// to convert borrowed types to their owned variants. For example `Result<&[T], E>` to
    /// `Result<Vec<T>, E>`
    ///
    /// [`cloned`]: Result::cloned
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(result_owned)]
    ///
    /// let s = "string";
    /// let res_s: Result<&str, ()> = Ok(s);
    /// assert_eq!(res_s, Ok("string"));
    /// let owned: Result<String, ()> = res_s.owned();
    /// assert_eq!(owned, Ok(String::from("string")));
    /// ```
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "result_owned", issue = "none")]
    pub fn owned(self) -> Result<T::Owned, E>
    where
        T: ToOwned,
    {
        match self {
            Ok(x) => Ok(x.to_owned()),
            Err(e) => Err(e),
        }
    }
}

impl<T: ?Sized, E> Result<&mut T, E> {
    /// Maps an `Result<&mut T>` to an `Result<T::Owned>` by cloning the contents of the
    /// result.
    ///
    /// In difference with [`cloned`], this method uses [`ToOwned`] trait, which allows
    /// to convert borrowed types to their owned variants. For example `Result<&[T], E>` to
    /// `Result<Vec<T>, E>`
    ///
    /// [`cloned`]: Result::cloned
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(result_owned)]
    ///
    /// let s = "string";
    /// let res_s: Result<&str, ()> = Ok(s);
    /// assert_eq!(res_s, Ok("string"));
    /// let owned: Result<String, ()> = res_s.owned();
    /// assert_eq!(owned, Ok(String::from("string")));
    /// ```
    #[rustc_allow_incoherent_impl]
    #[unstable(feature = "result_owned", issue = "none")]
    pub fn owned(self) -> Result<T::Owned, E>
    where
        T: ToOwned,
    {
        match self {
            Ok(x) => Ok(x.to_owned()),
            Err(e) => Err(e),
        }
    }
}
