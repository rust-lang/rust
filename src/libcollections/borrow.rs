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

use core::cmp::Ordering;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Deref};

use fmt;
use string::String;

use self::Cow::*;

#[stable(feature = "rust1", since = "1.0.0")]
pub use core::borrow::{Borrow, BorrowMut};

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Borrow<B> for Cow<'a, B>
    where B: ToOwned,
          <B as ToOwned>::Owned: 'a
{
    fn borrow(&self) -> &B {
        &**self
    }
}

/// A generalization of `Clone` to borrowed data.
///
/// Some types make it possible to go from borrowed to owned, usually by
/// implementing the `Clone` trait. But `Clone` works only for going from `&T`
/// to `T`. The `ToOwned` trait generalizes `Clone` to construct owned data
/// from any borrow of a given type.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait ToOwned {
    #[stable(feature = "rust1", since = "1.0.0")]
    type Owned: Borrow<Self>;

    /// Creates owned data from borrowed data, usually by cloning.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "a"; // &str
    /// let ss = s.to_owned(); // String
    ///
    /// let v = &[1, 2]; // slice
    /// let vv = v.to_owned(); // Vec
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn to_owned(&self) -> Self::Owned;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ToOwned for T
    where T: Clone
{
    type Owned = T;
    fn to_owned(&self) -> T {
        self.clone()
    }
}

/// A clone-on-write smart pointer.
///
/// The type `Cow` is a smart pointer providing clone-on-write functionality: it
/// can enclose and provide immutable access to borrowed data, and clone the
/// data lazily when mutation or ownership is required. The type is designed to
/// work with general borrowed data via the `Borrow` trait.
///
/// `Cow` implements `Deref`, which means that you can call
/// non-mutating methods directly on the data it encloses. If mutation
/// is desired, `to_mut` will obtain a mutable reference to an owned
/// value, cloning if necessary.
///
/// # Examples
///
/// ```
/// use std::borrow::Cow;
///
/// fn abs_all(input: &mut Cow<[i32]>) {
///     for i in 0..input.len() {
///         let v = input[i];
///         if v < 0 {
///             // Clones into a vector if not already owned.
///             input.to_mut()[i] = -v;
///         }
///     }
/// }
///
/// // No clone occurs because `input` doesn't need to be mutated.
/// let slice = [0, 1, 2];
/// let mut input = Cow::from(&slice[..]);
/// abs_all(&mut input);
///
/// // Clone occurs because `input` needs to be mutated.
/// let slice = [-1, 0, 1];
/// let mut input = Cow::from(&slice[..]);
/// abs_all(&mut input);
///
/// // No clone occurs because `input` is already owned.
/// let mut input = Cow::from(vec![-1, 0, 1]);
/// abs_all(&mut input);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Cow<'a, B: ?Sized + 'a>
    where B: ToOwned
{
    /// Borrowed data.
    #[stable(feature = "rust1", since = "1.0.0")]
    Borrowed(#[stable(feature = "rust1", since = "1.0.0")]
             &'a B),

    /// Owned data.
    #[stable(feature = "rust1", since = "1.0.0")]
    Owned(#[stable(feature = "rust1", since = "1.0.0")]
          <B as ToOwned>::Owned),
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Clone for Cow<'a, B>
    where B: ToOwned
{
    fn clone(&self) -> Cow<'a, B> {
        match *self {
            Borrowed(b) => Borrowed(b),
            Owned(ref o) => {
                let b: &B = o.borrow();
                Owned(b.to_owned())
            }
        }
    }
}

impl<'a, B: ?Sized> Cow<'a, B>
    where B: ToOwned
{
    /// Acquires a mutable reference to the owned form of the data.
    ///
    /// Clones the data if it is not already owned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::Cow;
    ///
    /// let mut cow: Cow<[_]> = Cow::Owned(vec![1, 2, 3]);
    ///
    /// let hello = cow.to_mut();
    ///
    /// assert_eq!(hello, &[1, 2, 3]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_mut(&mut self) -> &mut <B as ToOwned>::Owned {
        match *self {
            Borrowed(borrowed) => {
                *self = Owned(borrowed.to_owned());
                match *self {
                    Borrowed(..) => unreachable!(),
                    Owned(ref mut owned) => owned,
                }
            }
            Owned(ref mut owned) => owned,
        }
    }

    /// Extracts the owned data.
    ///
    /// Clones the data if it is not already owned.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::Cow;
    ///
    /// let cow: Cow<[_]> = Cow::Owned(vec![1, 2, 3]);
    ///
    /// let hello = cow.into_owned();
    ///
    /// assert_eq!(vec![1, 2, 3], hello);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_owned(self) -> <B as ToOwned>::Owned {
        match self {
            Borrowed(borrowed) => borrowed.to_owned(),
            Owned(owned) => owned,
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Deref for Cow<'a, B>
    where B: ToOwned
{
    type Target = B;

    fn deref(&self) -> &B {
        match *self {
            Borrowed(borrowed) => borrowed,
            Owned(ref owned) => owned.borrow(),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Eq for Cow<'a, B> where B: Eq + ToOwned {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Ord for Cow<'a, B>
    where B: Ord + ToOwned
{
    #[inline]
    fn cmp(&self, other: &Cow<'a, B>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b, B: ?Sized, C: ?Sized> PartialEq<Cow<'b, C>> for Cow<'a, B>
    where B: PartialEq<C> + ToOwned,
          C: ToOwned
{
    #[inline]
    fn eq(&self, other: &Cow<'b, C>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> PartialOrd for Cow<'a, B>
    where B: PartialOrd + ToOwned
{
    #[inline]
    fn partial_cmp(&self, other: &Cow<'a, B>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> fmt::Debug for Cow<'a, B>
    where B: fmt::Debug + ToOwned,
          <B as ToOwned>::Owned: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Borrowed(ref b) => fmt::Debug::fmt(b, f),
            Owned(ref o) => fmt::Debug::fmt(o, f),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> fmt::Display for Cow<'a, B>
    where B: fmt::Display + ToOwned,
          <B as ToOwned>::Owned: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Borrowed(ref b) => fmt::Display::fmt(b, f),
            Owned(ref o) => fmt::Display::fmt(o, f),
        }
    }
}

#[stable(feature = "default", since = "1.11.0")]
impl<'a, B: ?Sized> Default for Cow<'a, B>
    where B: ToOwned,
          <B as ToOwned>::Owned: Default
{
    /// Creates an owned Cow<'a, B> with the default value for the contained owned value.
    fn default() -> Cow<'a, B> {
        Owned(<B as ToOwned>::Owned::default())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, B: ?Sized> Hash for Cow<'a, B>
    where B: Hash + ToOwned
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl<'a, T: ?Sized + ToOwned> AsRef<T> for Cow<'a, T> {
    fn as_ref(&self) -> &T {
        self
    }
}

#[stable(feature = "cow_add", since = "1.14.0")]
impl<'a> Add<&'a str> for Cow<'a, str> {
    type Output = Cow<'a, str>;

    #[inline]
    fn add(mut self, rhs: &'a str) -> Self::Output {
        self += rhs;
        self
    }
}

#[stable(feature = "cow_add", since = "1.14.0")]
impl<'a> Add<Cow<'a, str>> for Cow<'a, str> {
    type Output = Cow<'a, str>;

    #[inline]
    fn add(mut self, rhs: Cow<'a, str>) -> Self::Output {
        self += rhs;
        self
    }
}

#[stable(feature = "cow_add", since = "1.14.0")]
impl<'a> AddAssign<&'a str> for Cow<'a, str> {
    fn add_assign(&mut self, rhs: &'a str) {
        if self.is_empty() {
            *self = Cow::Borrowed(rhs)
        } else if rhs.is_empty() {
            return;
        } else {
            if let Cow::Borrowed(lhs) = *self {
                let mut s = String::with_capacity(lhs.len() + rhs.len());
                s.push_str(lhs);
                *self = Cow::Owned(s);
            }
            self.to_mut().push_str(rhs);
        }
    }
}

#[stable(feature = "cow_add", since = "1.14.0")]
impl<'a> AddAssign<Cow<'a, str>> for Cow<'a, str> {
    fn add_assign(&mut self, rhs: Cow<'a, str>) {
        if self.is_empty() {
            *self = rhs
        } else if rhs.is_empty() {
            return;
        } else {
            if let Cow::Borrowed(lhs) = *self {
                let mut s = String::with_capacity(lhs.len() + rhs.len());
                s.push_str(lhs);
                *self = Cow::Owned(s);
            }
            self.to_mut().push_str(&rhs);
        }
    }
}
