// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The AST pointer
//!
//! Provides `P<T>`, a frozen owned smart pointer, as a replacement for `@T` in
//! the AST.
//!
//! # Motivations and benefits
//!
//! * **Identity**: sharing AST nodes is problematic for the various analysis
//!   passes (e.g. one may be able to bypass the borrow checker with a shared
//!   `ExprAddrOf` node taking a mutable borrow). The only reason `@T` in the
//!   AST hasn't caused issues is because of inefficient folding passes which
//!   would always deduplicate any such shared nodes. Even if the AST were to
//!   switch to an arena, this would still hold, i.e. it couldn't use `&'a T`,
//!   but rather a wrapper like `P<'a, T>`.
//!
//! * **Immutability**: `P<T>` disallows mutating its inner `T`, unlike `Box<T>`
//!   (unless it contains an `Unsafe` interior, but that may be denied later).
//!   This mainly prevents mistakes, but can also enforces a kind of "purity".
//!
//! * **Efficiency**: folding can reuse allocation space for `P<T>` and `Vec<T>`,
//!   the latter even when the input and output types differ (as it would be the
//!   case with arenas or a GADT AST using type parameters to toggle features).
//!
//! * **Maintainability**: `P<T>` provides a fixed interface - `Deref`,
//!   `and_then` and `map` - which can remain fully functional even if the
//!   implementation changes (using a special thread-local heap, for example).
//!   Moreover, a switch to, e.g. `P<'a, T>` would be easy and mostly automated.

use std::fmt;
use std::iter::FromIterator;
use std::ops::Deref;
use std::ptr;
use std::slice;
use std::vec;
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// An owned smart pointer.
#[derive(PartialEq, Eq, Hash)]
pub struct P<T: ?Sized> {
    ptr: Box<T>
}

// ----------------------------------------------------------------------------
// Common impls

impl<T: ?Sized> Deref for P<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.ptr
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for P<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.ptr, fmt)
    }
}

// ----------------------------------------------------------------------------
// Impls for one boxed element `P<T>`

#[allow(non_snake_case)]
/// Construct a `P<T>` from a `T` value.
pub fn P<T: 'static>(value: T) -> P<T> {
    P {
        ptr: Box::new(value)
    }
}

impl<T: 'static> P<T> {
    /// Move out of the pointer.
    /// Intended for chaining transformations not covered by `map`.
    pub fn and_then<U, F>(self, f: F) -> U where
        F: FnOnce(T) -> U,
    {
        f(*self.ptr)
    }

    /// Transform the inner value, consuming `self` and producing a new `P<T>`.
    pub fn map<F>(mut self, f: F) -> P<T> where
        F: FnOnce(T) -> T,
    {
        unsafe {
            let p = &mut *self.ptr;
            // FIXME(#5016) this shouldn't need to drop-fill to be safe.
            ptr::write(p, f(ptr::read_and_drop(p)));
        }
        self
    }
}

impl<T: 'static + Clone> Clone for P<T> {
    fn clone(&self) -> P<T> {
        P((**self).clone())
    }
}

impl<T: fmt::Display> fmt::Display for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<T: 'static + Decodable> Decodable for P<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<P<T>, D::Error> {
        Decodable::decode(d).map(P)
    }
}

impl<T: Encodable> Encodable for P<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

// ----------------------------------------------------------------------------
// Impls for boxed array `P<[T]>` (ex-`OwnedSlice`)

impl<T> P<[T]> {
    pub fn new() -> Self {
        P { ptr: Box::new([]) as Box<[T]> }
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `P::new` instead")]
    pub fn empty() -> P<[T]> {
        P { ptr: Box::new([]) as Box<[T]> }
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `P::from` instead")]
    pub fn from_vec(v: Vec<T>) -> P<[T]> {
        P { ptr: v.into_boxed_slice() }
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `P::into` instead")]
    pub fn into_vec(self) -> Vec<T> {
        self.ptr.into_vec()
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `&owned_slice[..]` instead")]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        &*self.ptr
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `P::into_iter` instead")]
    pub fn move_iter(self) -> vec::IntoIter<T> {
        self.ptr.into_vec().into_iter()
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `iter().map(f).collect()` instead")]
    pub fn map<U, F: FnMut(&T) -> U>(&self, f: F) -> P<[U]> {
        self.iter().map(f).collect()
    }
}

impl<T: Clone> Clone for P<[T]> {
    fn clone(&self) -> Self {
        P { ptr: self.ptr.clone() }
    }
}

impl<T> From<Vec<T>> for P<[T]> {
    fn from(v: Vec<T>) -> Self {
        P { ptr: v.into_boxed_slice() }
    }
}

impl<T> Into<Vec<T>> for P<[T]> {
    fn into(self) -> Vec<T> {
        self.ptr.into_vec()
    }
}

impl<T> FromIterator<T> for P<[T]> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        P::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<T> IntoIterator for P<[T]> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.ptr.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a P<[T]> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.ptr.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut P<[T]> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.ptr.iter_mut()
    }
}

impl<T: Encodable> Encodable for P<[T]> {
    fn encode<E: Encoder>(&self, s: &mut E) -> Result<(), E::Error> {
        Encodable::encode(&self.ptr, s)
    }
}

impl<T: Decodable> Decodable for P<[T]> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        Ok(P { ptr: try!(Decodable::decode(d)) })
    }
}
