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
//! `Cow` implements both `Deref`, which means that you can call
//! non-mutating methods directly on the data it encloses. If mutation
//! is desired, `to_mut` will obtain a mutable references to an owned
//! value, cloning if necessary.

#![unstable = "recently added as part of collections reform"]

use clone::Clone;
use cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use fmt;
use marker::Sized;
use ops::Deref;
use option::Option;
use self::Cow::*;

/// A trait for borrowing data.
#[old_orphan_check]
pub trait BorrowFrom<Owned: ?Sized> {
    /// Immutably borrow from an owned value.
    fn borrow_from(owned: &Owned) -> &Self;
}

/// A trait for mutably borrowing data.
#[old_orphan_check]
pub trait BorrowFromMut<Owned: ?Sized> : BorrowFrom<Owned> {
    /// Mutably borrow from an owned value.
    fn borrow_from_mut(owned: &mut Owned) -> &mut Self;
}

impl<T: ?Sized> BorrowFrom<T> for T {
    fn borrow_from(owned: &T) -> &T { owned }
}

impl<T: ?Sized> BorrowFromMut<T> for T {
    fn borrow_from_mut(owned: &mut T) -> &mut T { owned }
}

impl<'a, T: ?Sized> BorrowFrom<&'a T> for T {
    fn borrow_from<'b>(owned: &'b &'a T) -> &'b T { &**owned }
}

impl<'a, T: ?Sized> BorrowFrom<&'a mut T> for T {
    fn borrow_from<'b>(owned: &'b &'a mut T) -> &'b T { &**owned }
}

impl<'a, T: ?Sized> BorrowFromMut<&'a mut T> for T {
    fn borrow_from_mut<'b>(owned: &'b mut &'a mut T) -> &'b mut T { &mut **owned }
}

impl<'a, T, B: ?Sized> BorrowFrom<Cow<'a, T, B>> for B where B: ToOwned<T> {
    fn borrow_from<'b>(owned: &'b Cow<'a, T, B>) -> &'b B {
        &**owned
    }
}

/// Trait for moving into a `Cow`
#[old_orphan_check]
pub trait IntoCow<'a, T, B: ?Sized> {
    /// Moves `self` into `Cow`
    fn into_cow(self) -> Cow<'a, T, B>;
}

impl<'a, T, B: ?Sized> IntoCow<'a, T, B> for Cow<'a, T, B> where B: ToOwned<T> {
    fn into_cow(self) -> Cow<'a, T, B> {
        self
    }
}

/// A generalization of Clone to borrowed data.
#[old_orphan_check]
pub trait ToOwned<Owned>: BorrowFrom<Owned> {
    /// Create owned data from borrowed data, usually by copying.
    fn to_owned(&self) -> Owned;
}

impl<T> ToOwned<T> for T where T: Clone {
    fn to_owned(&self) -> T { self.clone() }
}

/// A clone-on-write smart pointer.
///
/// # Example
///
/// ```rust
/// use std::borrow::Cow;
///
/// fn abs_all(input: &mut Cow<Vec<int>, [int]>) {
///     for i in range(0, input.len()) {
///         let v = input[i];
///         if v < 0 {
///             // clones into a vector the first time (if not already owned)
///             input.to_mut()[i] = -v;
///         }
///     }
/// }
/// ```
#[derive(Show)]
pub enum Cow<'a, T, B: ?Sized + 'a> where B: ToOwned<T> {
    /// Borrowed data.
    Borrowed(&'a B),

    /// Owned data.
    Owned(T)
}

#[stable]
impl<'a, T, B: ?Sized> Clone for Cow<'a, T, B> where B: ToOwned<T> {
    fn clone(&self) -> Cow<'a, T, B> {
        match *self {
            Borrowed(b) => Borrowed(b),
            Owned(ref o) => {
                let b: &B = BorrowFrom::borrow_from(o);
                Owned(b.to_owned())
            },
        }
    }
}

impl<'a, T, B: ?Sized> Cow<'a, T, B> where B: ToOwned<T> {
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

    /// Returns true if this `Cow` wraps a borrowed value
    pub fn is_borrowed(&self) -> bool {
        match *self {
            Borrowed(_) => true,
            _ => false,
        }
    }

    /// Returns true if this `Cow` wraps an owned value
    pub fn is_owned(&self) -> bool {
        match *self {
            Owned(_) => true,
            _ => false,
        }
    }
}

#[stable]
impl<'a, T, B: ?Sized> Deref for Cow<'a, T, B> where B: ToOwned<T>  {
    type Target = B;

    fn deref(&self) -> &B {
        match *self {
            Borrowed(borrowed) => borrowed,
            Owned(ref owned) => BorrowFrom::borrow_from(owned)
        }
    }
}

#[stable]
impl<'a, T, B: ?Sized> Eq for Cow<'a, T, B> where B: Eq + ToOwned<T> {}

#[stable]
impl<'a, T, B: ?Sized> Ord for Cow<'a, T, B> where B: Ord + ToOwned<T> {
    #[inline]
    fn cmp(&self, other: &Cow<'a, T, B>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

#[stable]
impl<'a, 'b, T, U, B: ?Sized, C: ?Sized> PartialEq<Cow<'b, U, C>> for Cow<'a, T, B> where
    B: PartialEq<C> + ToOwned<T>,
    C: ToOwned<U>,
{
    #[inline]
    fn eq(&self, other: &Cow<'b, U, C>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

#[stable]
impl<'a, T, B: ?Sized> PartialOrd for Cow<'a, T, B> where B: PartialOrd + ToOwned<T> {
    #[inline]
    fn partial_cmp(&self, other: &Cow<'a, T, B>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

#[stable]
impl<'a, T, B: ?Sized> fmt::String for Cow<'a, T, B> where
    B: fmt::String + ToOwned<T>,
    T: fmt::String,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Borrowed(ref b) => fmt::String::fmt(b, f),
            Owned(ref o) => fmt::String::fmt(o, f),
        }
    }
}
