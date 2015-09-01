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
//!     foo: i32,
//!     bar: f32,
//! }
//! ```
//!
//! How can we define some default values? You can use `Default`:
//!
//! ```
//! #[derive(Default)]
//! struct SomeOptions {
//!     foo: i32,
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
//! enum Kind {
//!     A,
//!     B,
//!     C,
//! }
//!
//! impl Default for Kind {
//!     fn default() -> Kind { Kind::A }
//! }
//!
//! #[derive(Default)]
//! struct SomeOptions {
//!     foo: i32,
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
//! # #[derive(Default)]
//! # struct SomeOptions {
//! #     foo: i32,
//! #     bar: f32,
//! # }
//! fn main() {
//!     let options = SomeOptions { foo: 42, ..Default::default() };
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

/// A trait for giving a type a useful default value.
///
/// A struct can derive default implementations of `Default` for basic types using
/// `#[derive(Default)]`.
///
/// # Examples
///
/// ```
/// #[derive(Default)]
/// struct SomeOptions {
///     foo: i32,
///     bar: f32,
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
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
    /// let i: i8 = Default::default();
    /// let (x, y): (Option<String>, f64) = Default::default();
    /// let (a, b, (c, d)): (i32, u32, (bool, bool)) = Default::default();
    /// ```
    ///
    /// Making your own:
    ///
    /// ```
    /// enum Kind {
    ///     A,
    ///     B,
    ///     C,
    /// }
    ///
    /// impl Default for Kind {
    ///     fn default() -> Kind { Kind::A }
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> Self;
}

macro_rules! default_impl {
    ($t:ty, $v:expr) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Default for $t {
            #[inline]
            #[stable(feature = "rust1", since = "1.0.0")]
            fn default() -> $t { $v }
        }
    }
}

default_impl! { (), () }
default_impl! { bool, false }
default_impl! { char, '\x00' }

default_impl! { usize, 0 }
default_impl! { u8, 0 }
default_impl! { u16, 0 }
default_impl! { u32, 0 }
default_impl! { u64, 0 }

default_impl! { isize, 0 }
default_impl! { i8, 0 }
default_impl! { i16, 0 }
default_impl! { i32, 0 }
default_impl! { i64, 0 }

default_impl! { f32, 0.0f32 }
default_impl! { f64, 0.0f64 }
