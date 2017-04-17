// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits for conversions between types.
//!
//! The traits in this module provide a general way to talk about conversions
//! from one type to another. They follow the standard Rust conventions of
//! `as`/`into`/`from`.
//!
//! Like many traits, these are often used as bounds for generic functions, to
//! support arguments of multiple types.
//!
//! - Implement the `As*` traits for reference-to-reference conversions
//! - Implement the [`Into`] trait when you want to consume the value in the conversion
//! - The [`From`] trait is the most flexible, useful for value _and_ reference conversions
//! - The [`TryFrom`] and [`TryInto`] traits behave like [`From`] and [`Into`], but allow for the
//!   conversion to fail
//!
//! As a library author, you should prefer implementing [`From<T>`][`From`] or
//! [`TryFrom<T>`][`TryFrom`] rather than [`Into<U>`][`Into`] or [`TryInto<U>`][`TryInto`],
//! as [`From`] and [`TryFrom`] provide greater flexibility and offer
//! equivalent [`Into`] or [`TryInto`] implementations for free, thanks to a
//! blanket implementation in the standard library.
//!
//! # Generic Implementations
//!
//! - [`AsRef`] and [`AsMut`] auto-dereference if the inner type is a reference
//! - [`From`]`<U> for T` implies [`Into`]`<T> for U`
//! - [`TryFrom`]`<U> for T` implies [`TryInto`]`<T> for U`
//! - [`From`] and [`Into`] are reflexive, which means that all types can
//!   `into` themselves and `from` themselves
//!
//! See each trait for usage examples.
//!
//! [`Into`]: trait.Into.html
//! [`From`]: trait.From.html
//! [`TryFrom`]: trait.TryFrom.html
//! [`TryInto`]: trait.TryInto.html
//! [`AsRef`]: trait.AsRef.html
//! [`AsMut`]: trait.AsMut.html

#![stable(feature = "rust1", since = "1.0.0")]

use str::FromStr;

/// A cheap reference-to-reference conversion. Used to convert a value to a
/// reference value within generic code.
///
/// `AsRef` is very similar to, but serves a slightly different purpose than,
/// [`Borrow`].
///
/// `AsRef` is to be used when wishing to convert to a reference of another
/// type.
/// `Borrow` is more related to the notion of taking the reference. It is
/// useful when wishing to abstract over the type of reference
/// (`&T`, `&mut T`) or allow both the referenced and owned type to be treated
/// in the same manner.
///
/// The key difference between the two traits is the intention:
///
/// - Use `AsRef` when goal is to simply convert into a reference
/// - Use `Borrow` when goal is related to writing code that is agnostic to the
///   type of borrow and if is reference or value
///
/// See [the book][book] for a more detailed comparison.
///
/// [book]: ../../book/borrow-and-asref.html
/// [`Borrow`]: ../../std/borrow/trait.Borrow.html
///
/// **Note: this trait must not fail**. If the conversion can fail, use a
/// dedicated method which returns an [`Option<T>`] or a [`Result<T, E>`].
///
/// [`Option<T>`]: ../../std/option/enum.Option.html
/// [`Result<T, E>`]: ../../std/result/enum.Result.html
///
/// # Generic Implementations
///
/// - `AsRef` auto-dereferences if the inner type is a reference or a mutable
///   reference (e.g.: `foo.as_ref()` will work the same if `foo` has type
///   `&mut Foo` or `&&mut Foo`)
///
/// # Examples
///
/// Both [`String`] and `&str` implement `AsRef<str>`:
///
/// [`String`]: ../../std/string/struct.String.html
///
/// ```
/// fn is_hello<T: AsRef<str>>(s: T) {
///    assert_eq!("hello", s.as_ref());
/// }
///
/// let s = "hello";
/// is_hello(s);
///
/// let s = "hello".to_string();
/// is_hello(s);
/// ```
///
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsRef<T: ?Sized> {
    /// Performs the conversion.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_ref(&self) -> &T;
}

/// A cheap, mutable reference-to-mutable reference conversion.
///
/// This trait is similar to `AsRef` but used for converting between mutable
/// references.
///
/// **Note: this trait must not fail**. If the conversion can fail, use a
/// dedicated method which returns an [`Option<T>`] or a [`Result<T, E>`].
///
/// [`Option<T>`]: ../../std/option/enum.Option.html
/// [`Result<T, E>`]: ../../std/result/enum.Result.html
///
/// # Generic Implementations
///
/// - `AsMut` auto-dereferences if the inner type is a reference or a mutable
///   reference (e.g.: `foo.as_ref()` will work the same if `foo` has type
///   `&mut Foo` or `&&mut Foo`)
///
/// # Examples
///
/// [`Box<T>`] implements `AsMut<T>`:
///
/// [`Box<T>`]: ../../std/boxed/struct.Box.html
///
/// ```
/// fn add_one<T: AsMut<u64>>(num: &mut T) {
///     *num.as_mut() += 1;
/// }
///
/// let mut boxed_num = Box::new(0);
/// add_one(&mut boxed_num);
/// assert_eq!(*boxed_num, 1);
/// ```
///
///
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsMut<T: ?Sized> {
    /// Performs the conversion.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_mut(&mut self) -> &mut T;
}

/// A conversion that consumes `self`, which may or may not be expensive. The
/// reciprocal of [`From`][From].
///
/// **Note: this trait must not fail**. If the conversion can fail, use
/// [`TryInto`] or a dedicated method which returns an [`Option<T>`] or a
/// [`Result<T, E>`].
///
/// Library authors should not directly implement this trait, but should prefer
/// implementing the [`From`][From] trait, which offers greater flexibility and
/// provides an equivalent `Into` implementation for free, thanks to a blanket
/// implementation in the standard library.
///
/// # Generic Implementations
///
/// - [`From<T>`][From]` for U` implies `Into<U> for T`
/// - [`into`] is reflexive, which means that `Into<T> for T` is implemented
///
/// # Examples
///
/// [`String`] implements `Into<Vec<u8>>`:
///
/// ```
/// fn is_hello<T: Into<Vec<u8>>>(s: T) {
///    let bytes = b"hello".to_vec();
///    assert_eq!(bytes, s.into());
/// }
///
/// let s = "hello".to_string();
/// is_hello(s);
/// ```
///
/// [`TryInto`]: trait.TryInto.html
/// [`Option<T>`]: ../../std/option/enum.Option.html
/// [`Result<T, E>`]: ../../std/result/enum.Result.html
/// [`String`]: ../../std/string/struct.String.html
/// [From]: trait.From.html
/// [`into`]: trait.Into.html#tymethod.into
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Into<T>: Sized {
    /// Performs the conversion.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into(self) -> T;
}

/// Simple and safe type conversions in to `Self`. It is the reciprocal of
/// `Into`.
///
/// This trait is useful when performing error handling as described by
/// [the book][book] and is closely related to the `?` operator.
///
/// When constructing a function that is capable of failing the return type
/// will generally be of the form `Result<T, E>`.
///
/// The `From` trait allows for simplification of error handling by providing a
/// means of returning a single error type that encapsulates numerous possible
/// erroneous situations.
///
/// This trait is not limited to error handling, rather the general case for
/// this trait would be in any type conversions to have an explicit definition
/// of how they are performed.
///
/// **Note: this trait must not fail**. If the conversion can fail, use
/// [`TryFrom`] or a dedicated method which returns an [`Option<T>`] or a
/// [`Result<T, E>`].
///
/// # Generic Implementations
///
/// - `From<T> for U` implies [`Into<U>`]` for T`
/// - [`from`] is reflexive, which means that `From<T> for T` is implemented
///
/// # Examples
///
/// [`String`] implements `From<&str>`:
///
/// ```
/// let string = "hello".to_string();
/// let other_string = String::from("hello");
///
/// assert_eq!(string, other_string);
/// ```
///
/// An example usage for error handling:
///
/// ```
/// use std::io::{self, Read};
/// use std::num;
///
/// enum CliError {
///     IoError(io::Error),
///     ParseError(num::ParseIntError),
/// }
///
/// impl From<io::Error> for CliError {
///     fn from(error: io::Error) -> Self {
///         CliError::IoError(error)
///     }
/// }
///
/// impl From<num::ParseIntError> for CliError {
///     fn from(error: num::ParseIntError) -> Self {
///         CliError::ParseError(error)
///     }
/// }
///
/// fn open_and_parse_file(file_name: &str) -> Result<i32, CliError> {
///     let mut file = std::fs::File::open("test")?;
///     let mut contents = String::new();
///     file.read_to_string(&mut contents)?;
///     let num: i32 = contents.trim().parse()?;
///     Ok(num)
/// }
/// ```
///
/// [`TryFrom`]: trait.TryFrom.html
/// [`Option<T>`]: ../../std/option/enum.Option.html
/// [`Result<T, E>`]: ../../std/result/enum.Result.html
/// [`String`]: ../../std/string/struct.String.html
/// [`Into<U>`]: trait.Into.html
/// [`from`]: trait.From.html#tymethod.from
/// [book]: ../../book/error-handling.html
#[stable(feature = "rust1", since = "1.0.0")]
pub trait From<T>: Sized {
    /// Performs the conversion.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from(T) -> Self;
}

/// An attempted conversion that consumes `self`, which may or may not be
/// expensive.
///
/// Library authors should not directly implement this trait, but should prefer
/// implementing the [`TryFrom`] trait, which offers greater flexibility and
/// provides an equivalent `TryInto` implementation for free, thanks to a
/// blanket implementation in the standard library.
///
/// [`TryFrom`]: trait.TryFrom.html
#[unstable(feature = "try_from", issue = "33417")]
pub trait TryInto<T>: Sized {
    /// The type returned in the event of a conversion error.
    type Error;

    /// Performs the conversion.
    fn try_into(self) -> Result<T, Self::Error>;
}

/// Attempt to construct `Self` via a conversion.
#[unstable(feature = "try_from", issue = "33417")]
pub trait TryFrom<T>: Sized {
    /// The type returned in the event of a conversion error.
    type Error;

    /// Performs the conversion.
    fn try_from(value: T) -> Result<Self, Self::Error>;
}

////////////////////////////////////////////////////////////////////////////////
// GENERIC IMPLS
////////////////////////////////////////////////////////////////////////////////

// As lifts over &
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized, U: ?Sized> AsRef<U> for &'a T where T: AsRef<U>
{
    fn as_ref(&self) -> &U {
        <T as AsRef<U>>::as_ref(*self)
    }
}

// As lifts over &mut
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized, U: ?Sized> AsRef<U> for &'a mut T where T: AsRef<U>
{
    fn as_ref(&self) -> &U {
        <T as AsRef<U>>::as_ref(*self)
    }
}

// FIXME (#23442): replace the above impls for &/&mut with the following more general one:
// // As lifts over Deref
// impl<D: ?Sized + Deref, U: ?Sized> AsRef<U> for D where D::Target: AsRef<U> {
//     fn as_ref(&self) -> &U {
//         self.deref().as_ref()
//     }
// }

// AsMut lifts over &mut
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized, U: ?Sized> AsMut<U> for &'a mut T where T: AsMut<U>
{
    fn as_mut(&mut self) -> &mut U {
        (*self).as_mut()
    }
}

// FIXME (#23442): replace the above impl for &mut with the following more general one:
// // AsMut lifts over DerefMut
// impl<D: ?Sized + Deref, U: ?Sized> AsMut<U> for D where D::Target: AsMut<U> {
//     fn as_mut(&mut self) -> &mut U {
//         self.deref_mut().as_mut()
//     }
// }

// From implies Into
#[stable(feature = "rust1", since = "1.0.0")]
impl<T, U> Into<U> for T where U: From<T>
{
    fn into(self) -> U {
        U::from(self)
    }
}

// From (and thus Into) is reflexive
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> From<T> for T {
    fn from(t: T) -> T { t }
}


// TryFrom implies TryInto
#[unstable(feature = "try_from", issue = "33417")]
impl<T, U> TryInto<U> for T where U: TryFrom<T>
{
    type Error = U::Error;

    fn try_into(self) -> Result<U, U::Error> {
        U::try_from(self)
    }
}

////////////////////////////////////////////////////////////////////////////////
// CONCRETE IMPLS
////////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> AsRef<[T]> for [T] {
    fn as_ref(&self) -> &[T] {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> AsMut<[T]> for [T] {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<str> for str {
    #[inline]
    fn as_ref(&self) -> &str {
        self
    }
}

// FromStr implies TryFrom<&str>
#[unstable(feature = "try_from", issue = "33417")]
impl<'a, T> TryFrom<&'a str> for T where T: FromStr
{
    type Error = <T as FromStr>::Err;

    fn try_from(s: &'a str) -> Result<T, Self::Error> {
        FromStr::from_str(s)
    }
}
