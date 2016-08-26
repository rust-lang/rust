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
///
/// If you are implementing `Borrow` and both `Self` and `Borrowed` implement
/// `Hash`, `Eq`, and/or `Ord`, they must produce the same result.
///
/// `Borrow` is very similar to, but different than, `AsRef`. See
/// [the book][book] for more.
///
/// [book]: ../../book/borrow-and-asref.html
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Borrow<Borrowed: ?Sized> {
    /// Immutably borrows from an owned value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::Borrow;
    ///
    /// fn check<T: Borrow<str>>(s: T) {
    ///     assert_eq!("Hello", s.borrow());
    /// }
    ///
    /// let s = "Hello".to_string();
    ///
    /// check(s);
    ///
    /// let s = "Hello";
    ///
    /// check(s);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn borrow(&self) -> &Borrowed;
}

/// A trait for mutably borrowing data.
///
/// Similar to `Borrow`, but for mutable borrows.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait BorrowMut<Borrowed: ?Sized> : Borrow<Borrowed> {
    /// Mutably borrows from an owned value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::BorrowMut;
    ///
    /// fn check<T: BorrowMut<[i32]>>(mut v: T) {
    ///     assert_eq!(&mut [1, 2, 3], v.borrow_mut());
    /// }
    ///
    /// let v = vec![1, 2, 3];
    ///
    /// check(v);
    /// ```
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
