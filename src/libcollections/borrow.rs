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

#![stable(feature = "rust1", since = "1.0.0")]

use core::clone::Clone;
use core::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use core::hash::{Hash, Hasher};
use core::marker::Sized;
use core::ops::Deref;
use core::option::Option;

use fmt;
use alloc::{rc, arc};

use self::Cow::*;

/// A trait for borrowing data.
///
/// In general, there may be several ways to "borrow" a piece of data.  The
/// typical ways of borrowing a type `T` are `&T` (a shared borrow) and `&mut T`
/// (a mutable borrow). But types like `Vec<T>` provide additional kinds of
/// borrows: the borrowed slices `&[T]` and `&mut [T]`.
///
/// When writing generic code, it is often desirable to abstract over all ways
/// of borrowing data from a given type. That is the role of the `Borrow`
/// trait: if `T: Borrow<U>`, then `&U` can be borrowed from `&T`.  A given
/// type can be borrowed as multiple different types. In particular, `Vec<T>:
/// Borrow<Vec<T>>` and `Vec<T>: Borrow<[T]>`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Borrow<Borrowed: ?Sized> {
    /// Immutably borrow from an owned value.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn borrow(&self) -> &Borrowed;
}

/// A trait for mutably borrowing data.
///
/// Similar to `Borrow`, but for mutable borrows.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait BorrowMut<Borrowed: ?Sized> : Borrow<Borrowed> {
    /// Mutably borrow from an owned value.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn borrow_mut(&mut self) -> &mut Borrowed;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Borrow<T> for T {
    fn borrow(&self) -> &T { self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> BorrowMut<T> for T {
    fn borrow_mut(&mut self) -> &mut T { self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> Borrow<T> for &'a T {
    fn borrow(&self) -> &T { &**self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> Borrow<T> for &'a mut T {
    fn borrow(&self) -> &T { &**self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized> BorrowMut<T> for &'a mut T {
    fn borrow_mut(&mut self) -> &mut T { &mut **self }
}

impl<T> Borrow<T> for rc::Rc<T> {
    fn borrow(&self) -> &T { &**self }
}

impl<T> Borrow<T> for arc::Arc<T> {
    fn borrow(&self) -> &T { &**self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Borrow<B> for Cow<'a, B> where B: ToOwned, <B as ToOwned>::Owned: 'a {
    fn borrow(&self) -> &B {
        &**self
    }
}

/// A generalization of Clone to borrowed data.
///
/// Some types make it possible to go from borrowed to owned, usually by
/// implementing the `Clone` trait. But `Clone` works only for going from `&T`
/// to `T`. The `ToOwned` trait generalizes `Clone` to construct owned data
/// from any borrow of a given type.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ToOwned {
    #[stable(feature = "rust1", since = "1.0.0")]
    type Owned: Borrow<Self>;

    /// Create owned data from borrowed data, usually by copying.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_owned(&self) -> Self::Owned;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ToOwned for T where T: Clone {
    type Owned = T;
    fn to_owned(&self) -> T { self.clone() }
}

/// A clone-on-write smart pointer.
///
/// The type `Cow` is a smart pointer providing clone-on-write functionality: it
/// can enclose and provide immutable access to borrowed data, and clone the
/// data lazily when mutation or ownership is required. The type is designed to
/// work with general borrowed data via the `Borrow` trait.
///
/// `Cow` implements both `Deref`, which means that you can call
/// non-mutating methods directly on the data it encloses. If mutation
/// is desired, `to_mut` will obtain a mutable references to an owned
/// value, cloning if necessary.
///
/// # Example
///
/// ```rust
/// use std::borrow::Cow;
///
/// fn abs_all(input: &mut Cow<[int]>) {
///     for i in 0..input.len() {
///         let v = input[i];
///         if v < 0 {
///             // clones into a vector the first time (if not already owned)
///             input.to_mut()[i] = -v;
///         }
///     }
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Cow<'a, B: ?Sized + 'a> where B: ToOwned {
    /// Borrowed data.
    #[stable(feature = "rust1", since = "1.0.0")]
    Borrowed(&'a B),

    /// Owned data.
    #[stable(feature = "rust1", since = "1.0.0")]
    Owned(<B as ToOwned>::Owned)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Clone for Cow<'a, B> where B: ToOwned {
    fn clone(&self) -> Cow<'a, B> {
        match *self {
            Borrowed(b) => Borrowed(b),
            Owned(ref o) => {
                let b: &B = o.borrow();
                Owned(b.to_owned())
            },
        }
    }
}

impl<'a, B: ?Sized> Cow<'a, B> where B: ToOwned {
    /// Acquire a mutable reference to the owned form of the data.
    ///
    /// Copies the data if it is not already owned.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_mut(&mut self) -> &mut <B as ToOwned>::Owned {
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_owned(self) -> <B as ToOwned>::Owned {
        match self {
            Borrowed(borrowed) => borrowed.to_owned(),
            Owned(owned) => owned
        }
    }

    /// Returns true if this `Cow` wraps a borrowed value
    #[deprecated(since = "1.0.0", reason = "match on the enum instead")]
    #[unstable(feature = "std_misc")]
    pub fn is_borrowed(&self) -> bool {
        match *self {
            Borrowed(_) => true,
            _ => false,
        }
    }

    /// Returns true if this `Cow` wraps an owned value
    #[deprecated(since = "1.0.0", reason = "match on the enum instead")]
    #[unstable(feature = "std_misc")]
    pub fn is_owned(&self) -> bool {
        match *self {
            Owned(_) => true,
            _ => false,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Deref for Cow<'a, B> where B: ToOwned {
    type Target = B;

    fn deref(&self) -> &B {
        match *self {
            Borrowed(borrowed) => borrowed,
            Owned(ref owned) => owned.borrow()
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Eq for Cow<'a, B> where B: Eq + ToOwned {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Ord for Cow<'a, B> where B: Ord + ToOwned {
    #[inline]
    fn cmp(&self, other: &Cow<'a, B>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, B: ?Sized, C: ?Sized> PartialEq<Cow<'b, C>> for Cow<'a, B> where
    B: PartialEq<C> + ToOwned, C: ToOwned,
{
    #[inline]
    fn eq(&self, other: &Cow<'b, C>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> PartialOrd for Cow<'a, B> where B: PartialOrd + ToOwned,
{
    #[inline]
    fn partial_cmp(&self, other: &Cow<'a, B>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> fmt::Debug for Cow<'a, B> where
    B: fmt::Debug + ToOwned,
    <B as ToOwned>::Owned: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Borrowed(ref b) => fmt::Debug::fmt(b, f),
            Owned(ref o) => fmt::Debug::fmt(o, f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> fmt::Display for Cow<'a, B> where
    B: fmt::Display + ToOwned,
    <B as ToOwned>::Owned: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Borrowed(ref b) => fmt::Display::fmt(b, f),
            Owned(ref o) => fmt::Display::fmt(o, f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(stage0)]
impl<'a, B: ?Sized, S: Hasher> Hash<S> for Cow<'a, B> where B: Hash<S> + ToOwned
{
    #[inline]
    fn hash(&self, state: &mut S) {
        Hash::hash(&**self, state)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(not(stage0))]
impl<'a, B: ?Sized> Hash for Cow<'a, B> where B: Hash + ToOwned
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

/// Trait for moving into a `Cow`
#[stable(feature = "rust1", since = "1.0.0")]
pub trait IntoCow<'a, B: ?Sized> where B: ToOwned {
    /// Moves `self` into `Cow`
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into_cow(self) -> Cow<'a, B>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a,  B: ?Sized> IntoCow<'a, B> for Cow<'a, B> where B: ToOwned {
    fn into_cow(self) -> Cow<'a, B> {
        self
    }
}
