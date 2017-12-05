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

// impl Borrow<str> for String
// impl<T> Borrow<T> for Arc<T>
// impl<K> HashSet<K> { fn get<Q>(&self, q: &Q) where K: Borrow<Q> }

/// A trait identifying how borrowed data behaves.
///
/// If a type implements this trait, it signals that a reference to it behaves
/// exactly like a reference to `Borrowed`. As a consequence, if a trait is
/// implemented both by `Self` and `Borrowed`, all trait methods that
/// take a `&self` argument must produce the same result in both
/// implementations.
///
/// As a consequence, this trait should only be implemented for types managing
/// a value of another type without modifying its behavior. Examples are
/// smart pointers such as [`Box`] or [`Rc`] as well the owned version of
/// slices such as [`Vec`].
///
/// A relaxed version that allows providing a reference to some other type
/// without any further promises is available through [`AsRef`].
///
/// When writing generic code, a use of `Borrow` should always be justified
/// by additional trait bounds, making it clear that the two types need to
/// behave identically in a certain context. If the code should merely be
/// able to operate on any type that can produce a reference to a given type,
/// you should use [`AsRef`] instead.
///
/// The companion trait [`BorrowMut`] provides the same guarantees for
/// mutable references.
///
/// [`Box`]: ../boxed/struct.Box.html
/// [`Rc`]: ../rc/struct.Rc.html
/// [`Vec`]: ../vec/struct.Vec.html
/// [`AsRef`]: ../convert/trait.AsRef.html
/// [`BorrowMut`]: trait.BorrowMut.html
///
/// # Examples
///
/// As a data collection, [`HashMap`] owns both keys and values. If the key’s
/// actual data is wrapped in a managing type of some kind, it should,
/// however, still be possible to search for a value using a reference to the
/// key’s data. For instance, if the key is a string, then it is likely
/// stored with the hash map as a [`String`], while it should be possible
/// to search using a [`&str`][`str`]. Thus, `insert` needs to operate on a
/// string while `get` needs to be able to use a `&str`.
///
/// Slightly simplified, the relevant parts of `HashMap` look like this:
///
/// ```
/// use std::borrow::Borrow;
/// use std::hash::Hash;
///
/// pub struct HashMap<K, V> {
///     # marker: ::std::marker::PhantomData<(K, V)>,
///     // fields omitted
/// }
///
/// impl<K, V> HashMap<K, V> {
///     pub fn insert(&self, key: K, value: V) -> Option<V>
///         where K: Hash + Eq
///     {
///         # unimplemented!()
///         // ...
///     }
///
///     pub fn get<Q>(&self, k: &Q) -> Option<&V>
///         where K: Borrow<Q>,
///               Q: Hash + Eq + ?Sized
///     {
///         # unimplemented!()
///         // ...
///     }
/// }
/// ```
///
/// The entire hash map is generic over the stored type for the key, `K`.
/// When inserting a value, the map is given such a `K` and needs to find
/// the correct hash bucket and check if the key is already present based
/// on that `K` value. It therefore requires `K: Hash + Eq`.
///
/// In order to search for a value based on the key’s data, the `get` method
/// is generic over some type `Q`. Technically, it needs to convert that `Q`
/// into a `K` in order to use `K`’s [`Hash`] implementation to be able to
/// arrive at the same hash value as during insertion in order to look into
/// the right hash bucket. Since `K` is some kind of owned value, this likely
/// would involve cloning and isn’t really practical.
///
/// Instead, `get` relies on `Q`’s implementation of `Hash` and uses `Borrow`
/// to indicate that `K`’s implementation of `Hash` must produce the same
/// result as `Q`’s by demanding that `K: Borrow<Q>`.
///
/// As a consequence, the hash map breaks if a `K` wrapping a `Q` value
/// produces a different hash than `Q`. For instance, image you have a
/// type that wraps a string but compares ASCII letters case-insensitive:
///
/// ```
/// use std::ascii::AsciiExt;
///
/// pub struct CIString(String);
///
/// impl PartialEq for CIString {
///     fn eq(&self, other: &Self) -> bool {
///         self.0.eq_ignore_ascii_case(&other.0)
///     }
/// }
///
/// impl Eq for CIString { }
/// ```
///
/// Because two equal values need to produce the same hash value, the
/// implementation of `Hash` need to reflect that, too:
///
/// ```
/// # use std::ascii::AsciiExt;
/// # use std::hash::{Hash, Hasher};
/// # pub struct CIString(String);
/// impl Hash for CIString {
///     fn hash<H: Hasher>(&self, state: &mut H) {
///         for c in self.0.as_bytes() {
///             c.to_ascii_lowercase().hash(state)
///         }
///     }
/// }
/// ```
///
/// Can `CIString` implement `Borrow<str>`? It certainly can provide a
/// reference to a string slice via its contained owned string. But because
/// its `Hash` implementation differs, it cannot fulfill the guarantee for
/// `Borrow` that all common trait implementations must behave the same way
/// and must not, in fact, implement `Borrow<str>`. If it wants to allow
/// others access to the underlying `str`, it can do that via `AsRef<str>`
/// which doesn’t carry any such restrictions.
///
/// [`Hash`]: ../hash/trait.Hash.html
/// [`HashMap`]: ../collections/struct.HashMap.html
/// [`String`]: ../string/struct.String.html
/// [`str`]: ../primitive.str.html
///
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
