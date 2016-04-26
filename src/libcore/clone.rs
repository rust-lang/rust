// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The `Clone` trait for types that cannot be 'implicitly copied'.
//!
//! In Rust, some simple types are "implicitly copyable" and when you
//! assign them or pass them as arguments, the receiver will get a copy,
//! leaving the original value in place. These types do not require
//! allocation to copy and do not have finalizers (i.e. they do not
//! contain owned boxes or implement `Drop`), so the compiler considers
//! them cheap and safe to copy. For other types copies must be made
//! explicitly, by convention implementing the `Clone` trait and calling
//! the `clone` method.
//!
//! Basic usage example:
//!
//! ```
//! let s = String::new(); // String type implements Clone
//! let copy = s.clone(); // so we can clone it
//! ```
//!
//! To easily implement the Clone trait, you can also use
//! `#[derive(Clone)]`. Example:
//!
//! ```
//! #[derive(Clone)] // we add the Clone trait to Morpheus struct
//! struct Morpheus {
//!    blue_pill: f32,
//!    red_pill: i64,
//! }
//!
//! fn main() {
//!    let f = Morpheus { blue_pill: 0.0, red_pill: 0 };
//!    let copy = f.clone(); // and now we can clone it!
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

use marker::Sized;

/// A common trait for cloning an object.
///
/// This trait can be used with `#[derive]`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Clone : Sized {
    /// Returns a copy of the value.
    ///
    /// # Examples
    ///
    /// ```
    /// let hello = "Hello"; // &str implements Clone
    ///
    /// assert_eq!("Hello", hello.clone());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn clone(&self) -> Self;

    /// Performs copy-assignment from `source`.
    ///
    /// `a.clone_from(&b)` is equivalent to `a = b.clone()` in functionality,
    /// but can be overridden to reuse the resources of `a` to avoid unnecessary
    /// allocations.
    #[inline(always)]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn clone_from(&mut self, source: &Self) {
        *self = source.clone()
    }
}

// FIXME(aburka): this method is used solely by #[derive] to
// assert that every component of a type implements Clone.
//
// This should never be called by user code.
#[doc(hidden)]
#[inline(always)]
#[unstable(feature = "derive_clone_copy",
           reason = "deriving hack, should not be public",
           issue = "0")]
pub fn assert_receiver_is_clone<T: Clone + ?Sized>(_: &T) {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> Clone for &'a T {
    /// Returns a shallow copy of the reference.
    #[inline]
    fn clone(&self) -> &'a T { *self }
}

macro_rules! clone_impl {
    ($t:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Clone for $t {
            /// Returns a deep copy of the value.
            #[inline]
            fn clone(&self) -> $t { *self }
        }
    }
}

clone_impl! { isize }
clone_impl! { i8 }
clone_impl! { i16 }
clone_impl! { i32 }
clone_impl! { i64 }

clone_impl! { usize }
clone_impl! { u8 }
clone_impl! { u16 }
clone_impl! { u32 }
clone_impl! { u64 }

clone_impl! { f32 }
clone_impl! { f64 }

clone_impl! { () }
clone_impl! { bool }
clone_impl! { char }
