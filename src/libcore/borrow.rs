// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A module for working with borrowed data.
//!
//! # The `BorrowFrom` traits
//!
//! In general, there may be several ways to "borrow" a piece of data.  The
//! typical ways of borrowing a type `T` are `&T` (a shared borrow) and `&mut T`
//! (a mutable borrow). But types like `Vec<T>` provide additional kinds of
//! borrows: the borrowed slices `&[T]` and `&mut [T]`.
//!
//! When writing generic code, it is often desirable to abstract over all ways
//! of borrowing data from a given type. That is the role of the `BorrowFrom`
//! trait: if `T: BorrowFrom<U>`, then `&T` can be borrowed from `&U`.  A given
//! type can be borrowed as multiple different types. In particular, `Vec<T>:
//! BorrowFrom<Vec<T>>` and `[T]: BorrowFrom<Vec<T>>`.
//!
//! # The `ToOwned` trait
//!
//! Some types make it possible to go from borrowed to owned, usually by
//! implementing the `Clone` trait. But `Clone` works only for going from `&T`
//! to `T`. The `ToOwned` trait generalizes `Clone` to construct owned data
//! from any borrow of a given type.
//!
//! # The `Cow` (clone-on-write) type
//!
//! The type `Cow` is a smart pointer providing clone-on-write functionality: it
//! can enclose and provide immutable access to borrowed data, and clone the
//! data lazily when mutation or ownership is required. The type is designed to
//! work with general borrowed data via the `BorrowFrom` trait.
//!
//! `Cow` implements both `Deref` and `DerefMut`, which means that you can call
//! methods directly on the data it encloses. The first time a mutable reference
//! is required, the data will be cloned (via `to_owned`) if it is not
//! already owned.

#![unstable = "recently added as part of collections reform"]

use clone::Clone;
use kinds::Sized;
use ops::Deref;
use self::Cow::*;

/// A trait for borrowing data.
pub trait BorrowFrom<Sized? Owned> for Sized? {
    /// Immutably borrow from an owned value.
    fn borrow_from(owned: &Owned) -> &Self;
}

/// A trait for mutably borrowing data.
pub trait BorrowFromMut<Sized? Owned> for Sized? : BorrowFrom<Owned> {
    /// Mutably borrow from an owned value.
    fn borrow_from_mut(owned: &mut Owned) -> &mut Self;
}

impl<Sized? T> BorrowFrom<T> for T {
    fn borrow_from(owned: &T) -> &T { owned }
}

impl<Sized? T> BorrowFromMut<T> for T {
    fn borrow_from_mut(owned: &mut T) -> &mut T { owned }
}

impl BorrowFrom<&'static str> for str {
    fn borrow_from<'a>(owned: &'a &'static str) -> &'a str { &**owned }
}

/// A generalization of Clone to borrowed data.
pub trait ToOwned<Owned> for Sized?: BorrowFrom<Owned> {
    /// Create owned data from borrowed data, usually by copying.
    fn to_owned(&self) -> Owned;
}

impl<T> ToOwned<T> for T where T: Clone {
    fn to_owned(&self) -> T { self.clone() }
}

/// A clone-on-write smart pointer.
pub enum Cow<'a, T, B: 'a> where B: ToOwned<T> {
    /// Borrowed data.
    Borrowed(&'a B),

    /// Owned data.
    Owned(T)
}

impl<'a, T, B> Cow<'a, T, B> where B: ToOwned<T> {
    /// Acquire a mutable reference to the owned form of the data.
    ///
    /// Copies the data if it is not already owned.
    pub fn to_mut(&mut self) -> &mut T {
        match *self {
            Borrowed(borrowed) => {
                *self = Owned(borrowed.to_owned());
                self.to_mut()
            }
            Owned(ref mut owned) => owned
        }
    }

    /// Extract the owned data.
    ///
    /// Copies the data if it is not already owned.
    pub fn into_owned(self) -> T {
        match self {
            Borrowed(borrowed) => borrowed.to_owned(),
            Owned(owned) => owned
        }
    }
}

impl<'a, T, B> Deref<B> for Cow<'a, T, B> where B: ToOwned<T>  {
    fn deref(&self) -> &B {
        match *self {
            Borrowed(borrowed) => borrowed,
            Owned(ref owned) => BorrowFrom::borrow_from(owned)
        }
    }
}
