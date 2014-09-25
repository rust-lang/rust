// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `Default` trait for types which may have meaningful default values.
//!
//! Sometimes, you want to fall back to some kind of default value, and
//! don't particularly care what it is. This comes up often with `struct`s
//! that define a set of options:
//!
//! ```
//! struct SomeOptions {
//!     foo: int,
//!     bar: f32,
//! }
//! ```
//!
//! How can we define some default values? You can use `Default`:
//!
//! ```
//! use std::default::Default;
//!
//! #[deriving(Default)]
//! struct SomeOptions {
//!     foo: int,
//!     bar: f32,
//! }
//!
//!
//! fn main() {
//!     let options: SomeOptions = Default::default();
//! }
//! ```
//!
//! Now, you get all of the default values. Rust implements `Default` for various primitives types.
//! If you have your own type, you need to implement `Default` yourself:
//!
//! ```
//! use std::default::Default;
//!
//! enum Kind {
//!     A,
//!     B,
//!     C,
//! }
//!
//! impl Default for Kind {
//!     fn default() -> Kind { A }
//! }
//!
//! #[deriving(Default)]
//! struct SomeOptions {
//!     foo: int,
//!     bar: f32,
//!     baz: Kind,
//! }
//!
//!
//! fn main() {
//!     let options: SomeOptions = Default::default();
//! }
//! ```
//!
//! If you want to override a particular option, but still retain the other defaults:
//!
//! ```
//! # use std::default::Default;
//! # #[deriving(Default)]
//! # struct SomeOptions {
//! #     foo: int,
//! #     bar: f32,
//! # }
//! fn main() {
//!     let options = SomeOptions { foo: 42, ..Default::default() };
//! }
//! ```

#![stable]

/// A trait that types which have a useful default value should implement.
///
/// A struct can derive default implementations of `Default` for basic types using
/// `#[deriving(Default)]`.
///
/// # Examples
///
/// ```
/// #[deriving(Default)]
/// struct SomeOptions {
///     foo: int,
///     bar: f32,
/// }
/// ```
pub trait Default {
    /// Returns the "default value" for a type.
    ///
    /// Default values are often some kind of initial value, identity value, or anything else that
    /// may make sense as a default.
    ///
    /// # Examples
    ///
    /// Using built-in default values:
    ///
    /// ```
    /// use std::default::Default;
    ///
    /// let i: i8 = Default::default();
    /// let (x, y): (Option<String>, f64) = Default::default();
    /// let (a, b, (c, d)): (int, uint, (bool, bool)) = Default::default();
    /// ```
    ///
    /// Making your own:
    ///
    /// ```
    /// use std::default::Default;
    ///
    /// enum Kind {
    ///     A,
    ///     B,
    ///     C,
    /// }
    ///
    /// impl Default for Kind {
    ///     fn default() -> Kind { A }
    /// }
    /// ```
    fn default() -> Self;
}

macro_rules! default_impl(
    ($t:ty, $v:expr) => {
        impl Default for $t {
            #[inline]
            fn default() -> $t { $v }
        }
    }
)

default_impl!((), ())
default_impl!(bool, false)
default_impl!(char, '\x00')

default_impl!(uint, 0u)
default_impl!(u8,  0u8)
default_impl!(u16, 0u16)
default_impl!(u32, 0u32)
default_impl!(u64, 0u64)

default_impl!(int, 0i)
default_impl!(i8,  0i8)
default_impl!(i16, 0i16)
default_impl!(i32, 0i32)
default_impl!(i64, 0i64)

default_impl!(f32, 0.0f32)
default_impl!(f64, 0.0f64)
