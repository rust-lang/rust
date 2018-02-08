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

/// A trait identifying how borrowed data behaves.
///
/// In Rust, it is common to provide different representations of a type for
/// different use cases. For instance, storage location and management for a
/// value can be specifically chosen as appropriate for a particular use via
/// pointer types such as [`Box<T>`] or [`Rc<T>`] or one can opt into
/// concurrency via synchronization types such as [`Mutex<T>`], avoiding the
/// associated cost when in parallel doesn’t happen. Beyond these generic
/// wrappers that can be used with any type, some types provide optional
/// facets providing potentially costly functionality. An example for such a
/// type is [`String`] which adds the ability to extend a string to the basic
/// [`str`]. This requires keeping additional information unnecessary for a
/// simple, imutable string.
///
/// These types signal that they are a specialized representation of a basic
/// type `T` by implementing `Borrow<T>`. The method `borrow` provides a way
/// to convert a reference to the type into a reference to the underlying
/// basic type.
///
/// If a type implementing `Borrow<T>` implements other traits also
/// implemented by `T`, these implementations behave identically if the trait
/// is concerned with the data rather than its representation. For instance,
/// the comparison traits such as `PartialEq` or `PartialOrd` must behave
/// identical for `T` and any type implemeting `Borrow<T>`.
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
/// [`AsRef`]: ../../std/convert/trait.AsRef.html
/// [`BorrowMut`]: trait.BorrowMut.html
/// [`Box<T>`]: ../../std/boxed/struct.Box.html
/// [`Mutex<T>`]: ../../std/sync/struct.Mutex.html
/// [`Rc<T>`]: ../../std/rc/struct.Rc.html
/// [`str`]: ../../std/primitive.str.html
/// [`String`]: ../../std/string/struct.String.html
///
/// # Examples
///
/// As a data collection, [`HashMap<K, V>`] owns both keys and values. If
/// the key’s actual data is wrapped in a managing type of some kind, it
/// should, however, still be possible to search for a value using a
/// reference to the key’s data. For instance, if the key is a string, then
/// it is likely stored with the hash map as a [`String`], while it should
/// be possible to search using a [`&str`][`str`]. Thus, `insert` needs to
/// operate on a `String` while `get` needs to be able to use a `&str`.
///
/// Slightly simplified, the relevant parts of `HashMap<K, V>` look like
/// this:
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
///     where K: Hash + Eq
///     {
///         # unimplemented!()
///         // ...
///     }
///
///     pub fn get<Q>(&self, k: &Q) -> Option<&V>
///     where
///         K: Borrow<Q>,
///         Q: Hash + Eq + ?Sized
///     {
///         # unimplemented!()
///         // ...
///     }
/// }
/// ```
///
/// The entire hash map is generic over a key type `K`. Because these keys
/// are stored with the hash map, this type has to own the key’s data.
/// When inserting a key-value pair, the map is given such a `K` and needs
/// to find the correct hash bucket and check if the key is already present
/// based on that `K`. It therefore requires `K: Hash + Eq`.
///
/// When searching for a value in the map, however, having to provide a
/// reference to a `K` as the key to search for would require to always
/// create such an owned value. For string keys, this would mean a `String`
/// value needs to be created just for the search for cases where only a
/// `str` is available.
///
/// Instead, the `get` method is generic over the type of the underlying key
/// data, called `Q` in the method signature above. It states that `K` is a
/// representation of `Q` by requiring that `K: Borrow<Q>`. By additionally
/// requiring `Q: Hash + Eq`, it demands that `K` and `Q` have
/// implementations of the `Hash` and `Eq` traits that procude identical
/// results.
///
/// The implementation of `get` relies in particular on identical
/// implementations of `Hash` by determining the key’s hash bucket by calling
/// `Hash::hash` on the `Q` value even though it inserted the key based on
/// the hash value calculated from the `K` value.
///
/// As a consequence, the hash map breaks if a `K` wrapping a `Q` value
/// produces a different hash than `Q`. For instance, imagine you have a
/// type that wraps a string but compares ASCII letters ignoring their case:
///
/// ```
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
/// implementation of `Hash` needs to reflect that, too:
///
/// ```
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
/// [`Hash`]: ../../std/hash/trait.Hash.html
/// [`HashMap<K, V>`]: ../../std/collections/struct.HashMap.html
/// [`String`]: ../../std/string/struct.String.html
/// [`str`]: ../../std/primitive.str.html
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
