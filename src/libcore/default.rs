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

#![stable(feature = "rust1", since = "1.0.0")]

/// A trait for giving a type a useful default value.
///
/// Sometimes, you want to fall back to some kind of default value, and
/// don't particularly care what it is. This comes up often with `struct`s
/// that define a set of options:
///
/// ```
/// # #[allow(dead_code)]
/// struct SomeOptions {
///     foo: i32,
///     bar: f32,
/// }
/// ```
///
/// How can we define some default values? You can use `Default`:
///
/// ```
/// # #[allow(dead_code)]
/// #[derive(Default)]
/// struct SomeOptions {
///     foo: i32,
///     bar: f32,
/// }
///
/// fn main() {
///     let options: SomeOptions = Default::default();
/// }
/// ```
///
/// Now, you get all of the default values. Rust implements `Default` for various primitives types.
///
/// If you want to override a particular option, but still retain the other defaults:
///
/// ```
/// # #[allow(dead_code)]
/// # #[derive(Default)]
/// # struct SomeOptions {
/// #     foo: i32,
/// #     bar: f32,
/// # }
/// fn main() {
///     let options = SomeOptions { foo: 42, ..Default::default() };
/// }
/// ```
///
/// ## Derivable
///
/// This trait can be used with `#[derive]` if all of the type's fields implement
/// `Default`. When `derive`d, it will use the default value for each field's type.
///
/// ## How can I implement `Default`?
///
/// Provide an implementation for the `default()` method that returns the value of
/// your type that should be the default:
///
/// ```
/// # #![allow(dead_code)]
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
///
/// # Examples
///
/// ```
/// # #[allow(dead_code)]
/// #[derive(Default)]
/// struct SomeOptions {
///     foo: i32,
///     bar: f32,
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Default: Sized {
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
    /// # #[allow(dead_code)]
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
    ($t:ty, $v:expr, $doc:expr) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Default for $t {
            #[inline]
            #[doc = $doc]
            fn default() -> $t { $v }
        }
    }
}

default_impl! { (), (), "Returns the default value of `()`" }
default_impl! { bool, false, "Returns the default value of `false`" }
default_impl! { char, '\x00', "Returns the default value of `\\x00`" }

default_impl! { usize, 0, "Returns the default value of `0`" }
default_impl! { u8, 0, "Returns the default value of `0`" }
default_impl! { u16, 0, "Returns the default value of `0`" }
default_impl! { u32, 0, "Returns the default value of `0`" }
default_impl! { u64, 0, "Returns the default value of `0`" }
default_impl! { u128, 0, "Returns the default value of `0`" }

default_impl! { isize, 0, "Returns the default value of `0`" }
default_impl! { i8, 0, "Returns the default value of `0`" }
default_impl! { i16, 0, "Returns the default value of `0`" }
default_impl! { i32, 0, "Returns the default value of `0`" }
default_impl! { i64, 0, "Returns the default value of `0`" }
default_impl! { i128, 0, "Returns the default value of `0`" }

default_impl! { f32, 0.0f32, "Returns the default value of `0.0`" }
default_impl! { f64, 0.0f64, "Returns the default value of `0.0`" }
