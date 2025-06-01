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
//! // The `is_ok` and `is_err` methods do what they say.
//! let good_result: Result<i32, i32> = Ok(10);
//! let bad_result: Result<i32, i32> = Err(10);
//! assert!(good_result.is_ok() && !good_result.is_err());
//! assert!(bad_result.is_err() && !bad_result.is_ok());
//!
//! // `map` and `map_err` consume the `Result` and produce another.
//! let good_result: Result<i32, i32> = good_result.map(|i| i + 1);
//! let bad_result: Result<i32, i32> = bad_result.map_err(|i| i - 1);
//! assert_eq!(good_result, Ok(11));
//! assert_eq!(bad_result, Err(9));
//!
//! // Use `and_then` to continue the computation.
//! let good_result: Result<bool, i32> = good_result.and_then(|i| Ok(i == 11));
//! assert_eq!(good_result, Ok(true));
//!
//! // Use `or_else` to handle the error.
//! let bad_result: Result<i32, i32> = bad_result.or_else(|i| Ok(i + 20));
//! assert_eq!(bad_result, Ok(29));
//!
//! // Consume the result and return the contents with `unwrap`.
//! let final_awesome_result = good_result.unwrap();
//! assert!(final_awesome_result)
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
//! Ending the expression with [`?`] will result in the [`Ok`]'s unwrapped value, unless the result
//! is [`Err`], in which case [`Err`] is returned early from the enclosing function.
//!
//! [`?`] can be used in functions that return [`Result`] because of the
//! early return of [`Err`] that it provides.
//!
//! [`expect`]: Result::expect
//! [`Write`]: ../../std/io/trait.Write.html "io::Write"
//! [`write_all`]: ../../std/io/trait.Write.html#method.write_all "io::Write::write_all"
//! [`io::Result`]: ../../std/io/type.Result.html "io::Result"
//! [`?`]: crate::ops::Try
//! [`Ok(T)`]: Ok
//! [`Err(E)`]: Err
//! [io::Error]: ../../std/io/struct.Error.html "io::Error"
//!
//! # Representation
//!
//! In some cases, [`Result<T, E>`] will gain the same size, alignment, and ABI
//! guarantees as [`Option<U>`] has. One of either the `T` or `E` type must be a
//! type that qualifies for the `Option` [representation guarantees][opt-rep],
//! and the *other* type must meet all of the following conditions:
//! * Is a zero-sized type with alignment 1 (a "1-ZST").
//! * Has no fields.
//! * Does not have the `#[non_exhaustive]` attribute.
//!
//! For example, `NonZeroI32` qualifies for the `Option` representation
//! guarantees, and `()` is a zero-sized type with alignment 1, no fields, and
//! it isn't `non_exhaustive`. This means that both `Result<NonZeroI32, ()>` and
//! `Result<(), NonZeroI32>` have the same size, alignment, and ABI guarantees
//! as `Option<NonZeroI32>`. The only difference is the implied semantics:
//! * `Option<NonZeroI32>` is "a non-zero i32 might be present"
//! * `Result<NonZeroI32, ()>` is "a non-zero i32 success result, if any"
//! * `Result<(), NonZeroI32>` is "a non-zero i32 error result, if any"
//!
//! [opt-rep]: ../option/index.html#representation "Option Representation"
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
//! The [`is_ok_and`] and [`is_err_and`] methods apply the provided function
//! to the contents of the [`Result`] to produce a boolean value. If the [`Result`] does not have the expected variant
//! then [`false`] is returned instead without executing the function.
//!
//! [`is_err`]: Result::is_err
//! [`is_ok`]: Result::is_ok
//! [`is_ok_and`]: Result::is_ok_and
//! [`is_err_and`]: Result::is_err_and
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
//! * [`unwrap_unchecked`] produces *[undefined behavior]*
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
//! [`unwrap_unchecked`]: Result::unwrap_unchecked
//! [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
//!
//! These methods extract the contained value in a [`Result<T, E>`] when it
//! is the [`Err`] variant. They require `T` to implement the [`Debug`]
//! trait. If the [`Result`] is [`Ok`]:
//!
//! * [`expect_err`] panics with a provided custom message
//! * [`unwrap_err`] panics with a generic message
//! * [`unwrap_err_unchecked`] produces *[undefined behavior]*
//!
//! [`Debug`]: crate::fmt::Debug
//! [`expect_err`]: Result::expect_err
//! [`unwrap_err`]: Result::unwrap_err
//! [`unwrap_err_unchecked`]: Result::unwrap_err_unchecked
//! [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
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
//! These methods transform the contained value of the [`Ok`] variant:
//!
//! * [`map`] transforms [`Result<T, E>`] into [`Result<U, E>`] by applying
//!   the provided function to the contained value of [`Ok`] and leaving
//!   [`Err`] values unchanged
//! * [`inspect`] takes ownership of the [`Result`], applies the
//!   provided function to the contained value by reference,
//!   and then returns the [`Result`]
//!
//! [`map`]: Result::map
//! [`inspect`]: Result::inspect
//!
//! These methods transform the contained value of the [`Err`] variant:
//!
//! * [`map_err`] transforms [`Result<T, E>`] into [`Result<T, F>`] by
//!   applying the provided function to the contained value of [`Err`] and
//!   leaving [`Ok`] values unchanged
//! * [`inspect_err`] takes ownership of the [`Result`], applies the
//!   provided function to the contained value of [`Err`] by reference,
//!   and then returns the [`Result`]
//!
//! [`map_err`]: Result::map_err
//! [`inspect_err`]: Result::inspect_err
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
//! [`empty()`]: crate::iter::empty
//! [`once(v)`]: crate::iter::once
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
//! [impl-FromIterator]: Result#impl-FromIterator%3CResult%3CA,+E%3E%3E-for-Result%3CV,+E%3E
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
//! [impl-Product]: Result#impl-Product%3CResult%3CU,+E%3E%3E-for-Result%3CT,+E%3E
//! [impl-Sum]: Result#impl-Sum%3CResult%3CU,+E%3E%3E-for-Result%3CT,+E%3E
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

use crate::iter::{self, FusedIterator, TrustedLen};
use crate::ops::{self, ControlFlow, Deref, DerefMut};
use crate::{convert, fmt, hint};

/// `Result` is a type that represents either success ([`Ok`]) or failure ([`Err`]).
///
/// See the [module documentation](self) for details.
#[doc(search_unbox)]
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[must_use = "this `Result` may be an `Err` variant, which should be handled"]
#[rustc_diagnostic_item = "Result"]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Result<T, E> {
    /// Contains the success value
    #[lang = "Ok"]
    #[stable(feature = "rust1", since = "1.0.0")]
    Ok(#[stable(feature = "rust1", since = "1.0.0")] T),

    /// Contains the error value
    #[lang = "Err"]
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
    /// # Examples
    ///
    /// ```
    /// let x: Result<i32, &str> = Ok(-3);
    /// assert_eq!(x.is_ok(), true);
    ///
    /// let x: Result<i32, &str> = Err("Some error message");
    /// assert_eq!(x.is_ok(), false);
    /// ```
    #[must_use = "if you intended to assert that this is ok, consider `.unwrap()` instead"]
    #[rustc_const_stable(feature = "const_result_basics", since = "1.48.0")]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn is_ok(&self) -> bool {
        matches!(*self, Ok(_))
    }

    /// Returns `true` if the result is [`Ok`] and the value inside of it matches a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// assert_eq!(x.is_ok_and(|x| x > 1), true);
    ///
    /// let x: Result<u32, &str> = Ok(0);
    /// assert_eq!(x.is_ok_and(|x| x > 1), false);
    ///
    /// let x: Result<u32, &str> = Err("hey");
    /// assert_eq!(x.is_ok_and(|x| x > 1), false);
    ///
    /// let x: Result<String, &str> = Ok("ownership".to_string());
    /// assert_eq!(x.as_ref().is_ok_and(|x| x.len() > 1), true);
    /// println!("still alive {:?}", x);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "is_some_and", since = "1.70.0")]
    pub fn is_ok_and(self, f: impl FnOnce(T) -> bool) -> bool {
        match self {
            Err(_) => false,
            Ok(x) => f(x),
        }
    }

    /// Returns `true` if the result is [`Err`].
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
    #[must_use = "if you intended to assert that this is err, consider `.unwrap_err()` instead"]
    #[rustc_const_stable(feature = "const_result_basics", since = "1.48.0")]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn is_err(&self) -> bool {
        !self.is_ok()
    }

    /// Returns `true` if the result is [`Err`] and the value inside of it matches a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// let x: Result<u32, Error> = Err(Error::new(ErrorKind::NotFound, "!"));
    /// assert_eq!(x.is_err_and(|x| x.kind() == ErrorKind::NotFound), true);
    ///
    /// let x: Result<u32, Error> = Err(Error::new(ErrorKind::PermissionDenied, "!"));
    /// assert_eq!(x.is_err_and(|x| x.kind() == ErrorKind::NotFound), false);
    ///
    /// let x: Result<u32, Error> = Ok(123);
    /// assert_eq!(x.is_err_and(|x| x.kind() == ErrorKind::NotFound), false);
    ///
    /// let x: Result<u32, String> = Err("ownership".to_string());
    /// assert_eq!(x.as_ref().is_err_and(|x| x.len() > 1), true);
    /// println!("still alive {:?}", x);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "is_some_and", since = "1.70.0")]
    pub fn is_err_and(self, f: impl FnOnce(E) -> bool) -> bool {
        match self {
            Ok(_) => false,
            Err(e) => f(e),
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for each variant
    /////////////////////////////////////////////////////////////////////////

    /// Converts from `Result<T, E>` to [`Option<T>`].
    ///
    /// Converts `self` into an [`Option<T>`], consuming `self`,
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
    #[rustc_diagnostic_item = "result_ok_method"]
    pub fn ok(self) -> Option<T> {
        match self {
            Ok(x) => Some(x),
            Err(_) => None,
        }
    }

    /// Converts from `Result<T, E>` to [`Option<E>`].
    ///
    /// Converts `self` into an [`Option<E>`], consuming `self`,
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
            Ok(_) => None,
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
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// assert_eq!(x.as_ref(), Ok(&2));
    ///
    /// let x: Result<u32, &str> = Err("Error");
    /// assert_eq!(x.as_ref(), Err(&"Error"));
    /// ```
    #[inline]
    #[rustc_const_stable(feature = "const_result_basics", since = "1.48.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn as_ref(&self) -> Result<&T, &E> {
        match *self {
            Ok(ref x) => Ok(x),
            Err(ref x) => Err(x),
        }
    }

    /// Converts from `&mut Result<T, E>` to `Result<&mut T, &mut E>`.
    ///
    /// # Examples
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
    #[rustc_const_stable(feature = "const_result", since = "1.83.0")]
    pub const fn as_mut(&mut self) -> Result<&mut T, &mut E> {
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
    /// # Examples
    ///
    /// Print the numbers on each line of a string multiplied by two.
    ///
    /// ```
    /// let line = "1\n2\n3\n4\n";
    ///
    /// for num in line.lines() {
    ///     match num.parse::<i32>().map(|i| i * 2) {
    ///         Ok(n) => println!("{n}"),
    ///         Err(..) => {}
    ///     }
    /// }
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map<U, F: FnOnce(T) -> U>(self, op: F) -> Result<U, E> {
        match self {
            Ok(t) => Ok(op(t)),
            Err(e) => Err(e),
        }
    }

    /// Returns the provided default (if [`Err`]), or
    /// applies a function to the contained value (if [`Ok`]).
    ///
    /// Arguments passed to `map_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`map_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`map_or_else`]: Result::map_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<_, &str> = Ok("foo");
    /// assert_eq!(x.map_or(42, |v| v.len()), 3);
    ///
    /// let x: Result<&str, _> = Err("bar");
    /// assert_eq!(x.map_or(42, |v| v.len()), 42);
    /// ```
    #[inline]
    #[stable(feature = "result_map_or", since = "1.41.0")]
    #[must_use = "if you don't need the returned value, use `if let` instead"]
    pub fn map_or<U, F: FnOnce(T) -> U>(self, default: U, f: F) -> U {
        match self {
            Ok(t) => f(t),
            Err(_) => default,
        }
    }

    /// Maps a `Result<T, E>` to `U` by applying fallback function `default` to
    /// a contained [`Err`] value, or function `f` to a contained [`Ok`] value.
    ///
    /// This function can be used to unpack a successful result
    /// while handling an error.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// let k = 21;
    ///
    /// let x : Result<_, &str> = Ok("foo");
    /// assert_eq!(x.map_or_else(|e| k * 2, |v| v.len()), 3);
    ///
    /// let x : Result<&str, _> = Err("bar");
    /// assert_eq!(x.map_or_else(|e| k * 2, |v| v.len()), 42);
    /// ```
    #[inline]
    #[stable(feature = "result_map_or_else", since = "1.41.0")]
    pub fn map_or_else<U, D: FnOnce(E) -> U, F: FnOnce(T) -> U>(self, default: D, f: F) -> U {
        match self {
            Ok(t) => f(t),
            Err(e) => default(e),
        }
    }

    /// Maps a `Result<T, E>` to a `U` by applying function `f` to the contained
    /// value if the result is [`Ok`], otherwise if [`Err`], returns the
    /// [default value] for the type `U`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(result_option_map_or_default)]
    ///
    /// let x: Result<_, &str> = Ok("foo");
    /// let y: Result<&str, _> = Err("bar");
    ///
    /// assert_eq!(x.map_or_default(|x| x.len()), 3);
    /// assert_eq!(y.map_or_default(|y| y.len()), 0);
    /// ```
    ///
    /// [default value]: Default::default
    #[inline]
    #[unstable(feature = "result_option_map_or_default", issue = "138099")]
    pub fn map_or_default<U, F>(self, f: F) -> U
    where
        U: Default,
        F: FnOnce(T) -> U,
    {
        match self {
            Ok(t) => f(t),
            Err(_) => U::default(),
        }
    }

    /// Maps a `Result<T, E>` to `Result<T, F>` by applying a function to a
    /// contained [`Err`] value, leaving an [`Ok`] value untouched.
    ///
    /// This function can be used to pass through a successful result while handling
    /// an error.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// fn stringify(x: u32) -> String { format!("error code: {x}") }
    ///
    /// let x: Result<u32, u32> = Ok(2);
    /// assert_eq!(x.map_err(stringify), Ok(2));
    ///
    /// let x: Result<u32, u32> = Err(13);
    /// assert_eq!(x.map_err(stringify), Err("error code: 13".to_string()));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map_err<F, O: FnOnce(E) -> F>(self, op: O) -> Result<T, F> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(op(e)),
        }
    }

    /// Calls a function with a reference to the contained value if [`Ok`].
    ///
    /// Returns the original result.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: u8 = "4"
    ///     .parse::<u8>()
    ///     .inspect(|x| println!("original: {x}"))
    ///     .map(|x| x.pow(3))
    ///     .expect("failed to parse number");
    /// ```
    #[inline]
    #[stable(feature = "result_option_inspect", since = "1.76.0")]
    pub fn inspect<F: FnOnce(&T)>(self, f: F) -> Self {
        if let Ok(ref t) = self {
            f(t);
        }

        self
    }

    /// Calls a function with a reference to the contained value if [`Err`].
    ///
    /// Returns the original result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::{fs, io};
    ///
    /// fn read() -> io::Result<String> {
    ///     fs::read_to_string("address.txt")
    ///         .inspect_err(|e| eprintln!("failed to read file: {e}"))
    /// }
    /// ```
    #[inline]
    #[stable(feature = "result_option_inspect", since = "1.76.0")]
    pub fn inspect_err<F: FnOnce(&E)>(self, f: F) -> Self {
        if let Err(ref e) = self {
            f(e);
        }

        self
    }

    /// Converts from `Result<T, E>` (or `&Result<T, E>`) to `Result<&<T as Deref>::Target, &E>`.
    ///
    /// Coerces the [`Ok`] variant of the original [`Result`] via [`Deref`](crate::ops::Deref)
    /// and returns the new [`Result`].
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<String, u32> = Ok("hello".to_string());
    /// let y: Result<&str, &u32> = Ok("hello");
    /// assert_eq!(x.as_deref(), y);
    ///
    /// let x: Result<String, u32> = Err(42);
    /// let y: Result<&str, &u32> = Err(&42);
    /// assert_eq!(x.as_deref(), y);
    /// ```
    #[inline]
    #[stable(feature = "inner_deref", since = "1.47.0")]
    pub fn as_deref(&self) -> Result<&T::Target, &E>
    where
        T: Deref,
    {
        self.as_ref().map(|t| t.deref())
    }

    /// Converts from `Result<T, E>` (or `&mut Result<T, E>`) to `Result<&mut <T as DerefMut>::Target, &mut E>`.
    ///
    /// Coerces the [`Ok`] variant of the original [`Result`] via [`DerefMut`](crate::ops::DerefMut)
    /// and returns the new [`Result`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = "HELLO".to_string();
    /// let mut x: Result<String, u32> = Ok("hello".to_string());
    /// let y: Result<&mut str, &mut u32> = Ok(&mut s);
    /// assert_eq!(x.as_deref_mut().map(|x| { x.make_ascii_uppercase(); x }), y);
    ///
    /// let mut i = 42;
    /// let mut x: Result<String, u32> = Err(42);
    /// let y: Result<&mut str, &mut u32> = Err(&mut i);
    /// assert_eq!(x.as_deref_mut().map(|x| { x.make_ascii_uppercase(); x }), y);
    /// ```
    #[inline]
    #[stable(feature = "inner_deref", since = "1.47.0")]
    pub fn as_deref_mut(&mut self) -> Result<&mut T::Target, &mut E>
    where
        T: DerefMut,
    {
        self.as_mut().map(|t| t.deref_mut())
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

    /////////////////////////////////////////////////////////////////////////
    // Extract a value
    /////////////////////////////////////////////////////////////////////////

    /// Returns the contained [`Ok`] value, consuming the `self` value.
    ///
    /// Because this function may panic, its use is generally discouraged.
    /// Instead, prefer to use pattern matching and handle the [`Err`]
    /// case explicitly, or call [`unwrap_or`], [`unwrap_or_else`], or
    /// [`unwrap_or_default`].
    ///
    /// [`unwrap_or`]: Result::unwrap_or
    /// [`unwrap_or_else`]: Result::unwrap_or_else
    /// [`unwrap_or_default`]: Result::unwrap_or_default
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Err`], with a panic message including the
    /// passed message, and the content of the [`Err`].
    ///
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// x.expect("Testing expect"); // panics with `Testing expect: emergency failure`
    /// ```
    ///
    /// # Recommended Message Style
    ///
    /// We recommend that `expect` messages are used to describe the reason you
    /// _expect_ the `Result` should be `Ok`.
    ///
    /// ```should_panic
    /// let path = std::env::var("IMPORTANT_PATH")
    ///     .expect("env variable `IMPORTANT_PATH` should be set by `wrapper_script.sh`");
    /// ```
    ///
    /// **Hint**: If you're having trouble remembering how to phrase expect
    /// error messages remember to focus on the word "should" as in "env
    /// variable should be set by blah" or "the given binary should be available
    /// and executable by the current user".
    ///
    /// For more detail on expect message styles and the reasoning behind our recommendation please
    /// refer to the section on ["Common Message
    /// Styles"](../../std/error/index.html#common-message-styles) in the
    /// [`std::error`](../../std/error/index.html) module docs.
    #[inline]
    #[track_caller]
    #[stable(feature = "result_expect", since = "1.4.0")]
    pub fn expect(self, msg: &str) -> T
    where
        E: fmt::Debug,
    {
        match self {
            Ok(t) => t,
            Err(e) => unwrap_failed(msg, &e),
        }
    }

    /// Returns the contained [`Ok`] value, consuming the `self` value.
    ///
    /// Because this function may panic, its use is generally discouraged.
    /// Panics are meant for unrecoverable errors, and
    /// [may abort the entire program][panic-abort].
    ///
    /// Instead, prefer to use [the `?` (try) operator][try-operator], or pattern matching
    /// to handle the [`Err`] case explicitly, or call [`unwrap_or`],
    /// [`unwrap_or_else`], or [`unwrap_or_default`].
    ///
    /// [panic-abort]: https://doc.rust-lang.org/book/ch09-01-unrecoverable-errors-with-panic.html
    /// [try-operator]: https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html#a-shortcut-for-propagating-errors-the--operator
    /// [`unwrap_or`]: Result::unwrap_or
    /// [`unwrap_or_else`]: Result::unwrap_or_else
    /// [`unwrap_or_default`]: Result::unwrap_or_default
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Err`], with a panic message provided by the
    /// [`Err`]'s value.
    ///
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
    /// ```should_panic
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// x.unwrap(); // panics with `emergency failure`
    /// ```
    #[inline(always)]
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap(self) -> T
    where
        E: fmt::Debug,
    {
        match self {
            Ok(t) => t,
            Err(e) => unwrap_failed("called `Result::unwrap()` on an `Err` value", &e),
        }
    }

    /// Returns the contained [`Ok`] value or a default
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
    /// [`parse`]: str::parse
    /// [`FromStr`]: crate::str::FromStr
    #[inline]
    #[stable(feature = "result_unwrap_or_default", since = "1.16.0")]
    pub fn unwrap_or_default(self) -> T
    where
        T: Default,
    {
        match self {
            Ok(x) => x,
            Err(_) => Default::default(),
        }
    }

    /// Returns the contained [`Err`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Ok`], with a panic message including the
    /// passed message, and the content of the [`Ok`].
    ///
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// let x: Result<u32, &str> = Ok(10);
    /// x.expect_err("Testing expect_err"); // panics with `Testing expect_err: 10`
    /// ```
    #[inline]
    #[track_caller]
    #[stable(feature = "result_expect_err", since = "1.17.0")]
    pub fn expect_err(self, msg: &str) -> E
    where
        T: fmt::Debug,
    {
        match self {
            Ok(t) => unwrap_failed(msg, &t),
            Err(e) => e,
        }
    }

    /// Returns the contained [`Err`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is an [`Ok`], with a custom panic message provided
    /// by the [`Ok`]'s value.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// let x: Result<u32, &str> = Ok(2);
    /// x.unwrap_err(); // panics with `2`
    /// ```
    ///
    /// ```
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// assert_eq!(x.unwrap_err(), "emergency failure");
    /// ```
    #[inline]
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_err(self) -> E
    where
        T: fmt::Debug,
    {
        match self {
            Ok(t) => unwrap_failed("called `Result::unwrap_err()` on an `Ok` value", &t),
            Err(e) => e,
        }
    }

    /// Returns the contained [`Ok`] value, but never panics.
    ///
    /// Unlike [`unwrap`], this method is known to never panic on the
    /// result types it is implemented for. Therefore, it can be used
    /// instead of `unwrap` as a maintainability safeguard that will fail
    /// to compile if the error type of the `Result` is later changed
    /// to an error that can actually occur.
    ///
    /// [`unwrap`]: Result::unwrap
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(never_type)]
    /// # #![feature(unwrap_infallible)]
    ///
    /// fn only_good_news() -> Result<String, !> {
    ///     Ok("this is fine".into())
    /// }
    ///
    /// let s: String = only_good_news().into_ok();
    /// println!("{s}");
    /// ```
    #[unstable(feature = "unwrap_infallible", reason = "newly added", issue = "61695")]
    #[inline]
    pub fn into_ok(self) -> T
    where
        E: Into<!>,
    {
        match self {
            Ok(x) => x,
            Err(e) => e.into(),
        }
    }

    /// Returns the contained [`Err`] value, but never panics.
    ///
    /// Unlike [`unwrap_err`], this method is known to never panic on the
    /// result types it is implemented for. Therefore, it can be used
    /// instead of `unwrap_err` as a maintainability safeguard that will fail
    /// to compile if the ok type of the `Result` is later changed
    /// to a type that can actually occur.
    ///
    /// [`unwrap_err`]: Result::unwrap_err
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(never_type)]
    /// # #![feature(unwrap_infallible)]
    ///
    /// fn only_bad_news() -> Result<!, String> {
    ///     Err("Oops, it failed".into())
    /// }
    ///
    /// let error: String = only_bad_news().into_err();
    /// println!("{error}");
    /// ```
    #[unstable(feature = "unwrap_infallible", reason = "newly added", issue = "61695")]
    #[inline]
    pub fn into_err(self) -> E
    where
        T: Into<!>,
    {
        match self {
            Ok(x) => x.into(),
            Err(e) => e,
        }
    }

    ////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns `res` if the result is [`Ok`], otherwise returns the [`Err`] value of `self`.
    ///
    /// Arguments passed to `and` are eagerly evaluated; if you are passing the
    /// result of a function call, it is recommended to use [`and_then`], which is
    /// lazily evaluated.
    ///
    /// [`and_then`]: Result::and_then
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

    /// Calls `op` if the result is [`Ok`], otherwise returns the [`Err`] value of `self`.
    ///
    ///
    /// This function can be used for control flow based on `Result` values.
    ///
    /// # Examples
    ///
    /// ```
    /// fn sq_then_to_string(x: u32) -> Result<String, &'static str> {
    ///     x.checked_mul(x).map(|sq| sq.to_string()).ok_or("overflowed")
    /// }
    ///
    /// assert_eq!(Ok(2).and_then(sq_then_to_string), Ok(4.to_string()));
    /// assert_eq!(Ok(1_000_000).and_then(sq_then_to_string), Err("overflowed"));
    /// assert_eq!(Err("not a number").and_then(sq_then_to_string), Err("not a number"));
    /// ```
    ///
    /// Often used to chain fallible operations that may return [`Err`].
    ///
    /// ```
    /// use std::{io::ErrorKind, path::Path};
    ///
    /// // Note: on Windows "/" maps to "C:\"
    /// let root_modified_time = Path::new("/").metadata().and_then(|md| md.modified());
    /// assert!(root_modified_time.is_ok());
    ///
    /// let should_fail = Path::new("/bad/path").metadata().and_then(|md| md.modified());
    /// assert!(should_fail.is_err());
    /// assert_eq!(should_fail.unwrap_err().kind(), ErrorKind::NotFound);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_confusables("flat_map", "flatmap")]
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
    /// [`or_else`]: Result::or_else
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

    /// Calls `op` if the result is [`Err`], otherwise returns the [`Ok`] value of `self`.
    ///
    /// This function can be used for control flow based on result values.
    ///
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

    /// Returns the contained [`Ok`] value or a provided default.
    ///
    /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`unwrap_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`unwrap_or_else`]: Result::unwrap_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// let default = 2;
    /// let x: Result<u32, &str> = Ok(9);
    /// assert_eq!(x.unwrap_or(default), 9);
    ///
    /// let x: Result<u32, &str> = Err("error");
    /// assert_eq!(x.unwrap_or(default), default);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Ok(t) => t,
            Err(_) => default,
        }
    }

    /// Returns the contained [`Ok`] value or computes it from a closure.
    ///
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
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or_else<F: FnOnce(E) -> T>(self, op: F) -> T {
        match self {
            Ok(t) => t,
            Err(e) => op(e),
        }
    }

    /// Returns the contained [`Ok`] value, consuming the `self` value,
    /// without checking that the value is not an [`Err`].
    ///
    /// # Safety
    ///
    /// Calling this method on an [`Err`] is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<u32, &str> = Ok(2);
    /// assert_eq!(unsafe { x.unwrap_unchecked() }, 2);
    /// ```
    ///
    /// ```no_run
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// unsafe { x.unwrap_unchecked(); } // Undefined behavior!
    /// ```
    #[inline]
    #[track_caller]
    #[stable(feature = "option_result_unwrap_unchecked", since = "1.58.0")]
    pub unsafe fn unwrap_unchecked(self) -> T {
        match self {
            Ok(t) => t,
            // SAFETY: the safety contract must be upheld by the caller.
            Err(_) => unsafe { hint::unreachable_unchecked() },
        }
    }

    /// Returns the contained [`Err`] value, consuming the `self` value,
    /// without checking that the value is not an [`Ok`].
    ///
    /// # Safety
    ///
    /// Calling this method on an [`Ok`] is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let x: Result<u32, &str> = Ok(2);
    /// unsafe { x.unwrap_err_unchecked() }; // Undefined behavior!
    /// ```
    ///
    /// ```
    /// let x: Result<u32, &str> = Err("emergency failure");
    /// assert_eq!(unsafe { x.unwrap_err_unchecked() }, "emergency failure");
    /// ```
    #[inline]
    #[track_caller]
    #[stable(feature = "option_result_unwrap_unchecked", since = "1.58.0")]
    pub unsafe fn unwrap_err_unchecked(self) -> E {
        match self {
            // SAFETY: the safety contract must be upheld by the caller.
            Ok(_) => unsafe { hint::unreachable_unchecked() },
            Err(e) => e,
        }
    }
}

impl<T, E> Result<&T, E> {
    /// Maps a `Result<&T, E>` to a `Result<T, E>` by copying the contents of the
    /// `Ok` part.
    ///
    /// # Examples
    ///
    /// ```
    /// let val = 12;
    /// let x: Result<&i32, i32> = Ok(&val);
    /// assert_eq!(x, Ok(&12));
    /// let copied = x.copied();
    /// assert_eq!(copied, Ok(12));
    /// ```
    #[inline]
    #[stable(feature = "result_copied", since = "1.59.0")]
    #[rustc_const_stable(feature = "const_result", since = "1.83.0")]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    pub const fn copied(self) -> Result<T, E>
    where
        T: Copy,
    {
        // FIXME(const-hack): this implementation, which sidesteps using `Result::map` since it's not const
        // ready yet, should be reverted when possible to avoid code repetition
        match self {
            Ok(&v) => Ok(v),
            Err(e) => Err(e),
        }
    }

    /// Maps a `Result<&T, E>` to a `Result<T, E>` by cloning the contents of the
    /// `Ok` part.
    ///
    /// # Examples
    ///
    /// ```
    /// let val = 12;
    /// let x: Result<&i32, i32> = Ok(&val);
    /// assert_eq!(x, Ok(&12));
    /// let cloned = x.cloned();
    /// assert_eq!(cloned, Ok(12));
    /// ```
    #[inline]
    #[stable(feature = "result_cloned", since = "1.59.0")]
    pub fn cloned(self) -> Result<T, E>
    where
        T: Clone,
    {
        self.map(|t| t.clone())
    }
}

impl<T, E> Result<&mut T, E> {
    /// Maps a `Result<&mut T, E>` to a `Result<T, E>` by copying the contents of the
    /// `Ok` part.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut val = 12;
    /// let x: Result<&mut i32, i32> = Ok(&mut val);
    /// assert_eq!(x, Ok(&mut 12));
    /// let copied = x.copied();
    /// assert_eq!(copied, Ok(12));
    /// ```
    #[inline]
    #[stable(feature = "result_copied", since = "1.59.0")]
    #[rustc_const_stable(feature = "const_result", since = "1.83.0")]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    pub const fn copied(self) -> Result<T, E>
    where
        T: Copy,
    {
        // FIXME(const-hack): this implementation, which sidesteps using `Result::map` since it's not const
        // ready yet, should be reverted when possible to avoid code repetition
        match self {
            Ok(&mut v) => Ok(v),
            Err(e) => Err(e),
        }
    }

    /// Maps a `Result<&mut T, E>` to a `Result<T, E>` by cloning the contents of the
    /// `Ok` part.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut val = 12;
    /// let x: Result<&mut i32, i32> = Ok(&mut val);
    /// assert_eq!(x, Ok(&mut 12));
    /// let cloned = x.cloned();
    /// assert_eq!(cloned, Ok(12));
    /// ```
    #[inline]
    #[stable(feature = "result_cloned", since = "1.59.0")]
    pub fn cloned(self) -> Result<T, E>
    where
        T: Clone,
    {
        self.map(|t| t.clone())
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
    #[rustc_const_stable(feature = "const_result", since = "1.83.0")]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    pub const fn transpose(self) -> Option<Result<T, E>> {
        match self {
            Ok(Some(x)) => Some(Ok(x)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl<T, E> Result<Result<T, E>, E> {
    /// Converts from `Result<Result<T, E>, E>` to `Result<T, E>`
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Result<Result<&'static str, u32>, u32> = Ok(Ok("hello"));
    /// assert_eq!(Ok("hello"), x.flatten());
    ///
    /// let x: Result<Result<&'static str, u32>, u32> = Ok(Err(6));
    /// assert_eq!(Err(6), x.flatten());
    ///
    /// let x: Result<Result<&'static str, u32>, u32> = Err(6);
    /// assert_eq!(Err(6), x.flatten());
    /// ```
    ///
    /// Flattening only removes one level of nesting at a time:
    ///
    /// ```
    /// let x: Result<Result<Result<&'static str, u32>, u32>, u32> = Ok(Ok(Ok("hello")));
    /// assert_eq!(Ok(Ok("hello")), x.flatten());
    /// assert_eq!(Ok("hello"), x.flatten().flatten());
    /// ```
    #[inline]
    #[stable(feature = "result_flattening", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    #[rustc_const_stable(feature = "result_flattening", since = "CURRENT_RUSTC_VERSION")]
    pub const fn flatten(self) -> Result<T, E> {
        // FIXME(const-hack): could be written with `and_then`
        match self {
            Ok(inner) => inner,
            Err(e) => Err(e),
        }
    }
}

// This is a separate function to reduce the code size of the methods
#[cfg(not(feature = "panic_immediate_abort"))]
#[inline(never)]
#[cold]
#[track_caller]
fn unwrap_failed(msg: &str, error: &dyn fmt::Debug) -> ! {
    panic!("{msg}: {error:?}")
}

// This is a separate function to avoid constructing a `dyn Debug`
// that gets immediately thrown away, since vtables don't get cleaned up
// by dead code elimination if a trait object is constructed even if it goes
// unused
#[cfg(feature = "panic_immediate_abort")]
#[inline]
#[cold]
#[track_caller]
fn unwrap_failed<T>(_msg: &str, _error: &T) -> ! {
    panic!()
}

/////////////////////////////////////////////////////////////////////////////
// Trait implementations
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T, E> Clone for Result<T, E>
where
    T: Clone,
    E: Clone,
{
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

#[unstable(feature = "ergonomic_clones", issue = "132290")]
impl<T, E> crate::clone::UseCloned for Result<T, E>
where
    T: crate::clone::UseCloned,
    E: crate::clone::UseCloned,
{
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
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    inner: Option<&'a T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.inner.take()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        self.inner.take()
    }
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
    fn clone(&self) -> Self {
        Iter { inner: self.inner }
    }
}

/// An iterator over a mutable reference to the [`Ok`] variant of a [`Result`].
///
/// Created by [`Result::iter_mut`].
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    inner: Option<&'a mut T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        self.inner.take()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        self.inner.take()
    }
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
/// [`Result`] (provided by the [`IntoIterator`] trait).
///
/// [`into_iter`]: IntoIterator::into_iter
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    inner: Option<T>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.take()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.inner.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.inner.take()
    }
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
    fn from_iter<I: IntoIterator<Item = Result<A, E>>>(iter: I) -> Result<V, E> {
        iter::try_process(iter.into_iter(), |i| i.collect())
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T, E> ops::Try for Result<T, E> {
    type Output = T;
    type Residual = Result<convert::Infallible, E>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(v) => ControlFlow::Continue(v),
            Err(e) => ControlFlow::Break(Err(e)),
        }
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T, E, F: From<E>> ops::FromResidual<Result<convert::Infallible, E>> for Result<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: Result<convert::Infallible, E>) -> Self {
        match residual {
            Err(e) => Err(From::from(e)),
        }
    }
}
#[diagnostic::do_not_recommend]
#[unstable(feature = "try_trait_v2_yeet", issue = "96374")]
impl<T, E, F: From<E>> ops::FromResidual<ops::Yeet<E>> for Result<T, F> {
    #[inline]
    fn from_residual(ops::Yeet(e): ops::Yeet<E>) -> Self {
        Err(From::from(e))
    }
}

#[unstable(feature = "try_trait_v2_residual", issue = "91285")]
impl<T, E> ops::Residual<T> for Result<convert::Infallible, E> {
    type TryType = Result<T, E>;
}
