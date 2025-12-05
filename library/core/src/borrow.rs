//! Utilities for working with borrowed data.

#![stable(feature = "rust1", since = "1.0.0")]

/// A trait for borrowing data.
///
/// In Rust, it is common to provide different representations of a type for
/// different use cases. For instance, storage location and management for a
/// value can be specifically chosen as appropriate for a particular use via
/// pointer types such as [`Box<T>`] or [`Rc<T>`]. Beyond these generic
/// wrappers that can be used with any type, some types provide optional
/// facets providing potentially costly functionality. An example for such a
/// type is [`String`] which adds the ability to extend a string to the basic
/// [`str`]. This requires keeping additional information unnecessary for a
/// simple, immutable string.
///
/// These types provide access to the underlying data through references
/// to the type of that data. They are said to be ‘borrowed as’ that type.
/// For instance, a [`Box<T>`] can be borrowed as `T` while a [`String`]
/// can be borrowed as `str`.
///
/// Types express that they can be borrowed as some type `T` by implementing
/// `Borrow<T>`, providing a reference to a `T` in the trait’s
/// [`borrow`] method. A type is free to borrow as several different types.
/// If it wishes to mutably borrow as the type, allowing the underlying data
/// to be modified, it can additionally implement [`BorrowMut<T>`].
///
/// Further, when providing implementations for additional traits, it needs
/// to be considered whether they should behave identically to those of the
/// underlying type as a consequence of acting as a representation of that
/// underlying type. Generic code typically uses `Borrow<T>` when it relies
/// on the identical behavior of these additional trait implementations.
/// These traits will likely appear as additional trait bounds.
///
/// In particular `Eq`, `Ord` and `Hash` must be equivalent for
/// borrowed and owned values: `x.borrow() == y.borrow()` should give the
/// same result as `x == y`.
///
/// If generic code merely needs to work for all types that can
/// provide a reference to related type `T`, it is often better to use
/// [`AsRef<T>`] as more types can safely implement it.
///
/// [`Box<T>`]: ../../std/boxed/struct.Box.html
/// [`Mutex<T>`]: ../../std/sync/struct.Mutex.html
/// [`Rc<T>`]: ../../std/rc/struct.Rc.html
/// [`String`]: ../../std/string/struct.String.html
/// [`borrow`]: Borrow::borrow
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
/// data, called `Q` in the method signature above. It states that `K`
/// borrows as a `Q` by requiring that `K: Borrow<Q>`. By additionally
/// requiring `Q: Hash + Eq`, it signals the requirement that `K` and `Q`
/// have implementations of the `Hash` and `Eq` traits that produce identical
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
/// pub struct CaseInsensitiveString(String);
///
/// impl PartialEq for CaseInsensitiveString {
///     fn eq(&self, other: &Self) -> bool {
///         self.0.eq_ignore_ascii_case(&other.0)
///     }
/// }
///
/// impl Eq for CaseInsensitiveString { }
/// ```
///
/// Because two equal values need to produce the same hash value, the
/// implementation of `Hash` needs to ignore ASCII case, too:
///
/// ```
/// # use std::hash::{Hash, Hasher};
/// # pub struct CaseInsensitiveString(String);
/// impl Hash for CaseInsensitiveString {
///     fn hash<H: Hasher>(&self, state: &mut H) {
///         for c in self.0.as_bytes() {
///             c.to_ascii_lowercase().hash(state)
///         }
///     }
/// }
/// ```
///
/// Can `CaseInsensitiveString` implement `Borrow<str>`? It certainly can
/// provide a reference to a string slice via its contained owned string.
/// But because its `Hash` implementation differs, it behaves differently
/// from `str` and therefore must not, in fact, implement `Borrow<str>`.
/// If it wants to allow others access to the underlying `str`, it can do
/// that via `AsRef<str>` which doesn’t carry any extra requirements.
///
/// [`Hash`]: crate::hash::Hash
/// [`HashMap<K, V>`]: ../../std/collections/struct.HashMap.html
/// [`String`]: ../../std/string/struct.String.html
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Borrow"]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
pub const trait Borrow<Borrowed: ?Sized> {
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
/// As a companion to [`Borrow<T>`] this trait allows a type to borrow as
/// an underlying type by providing a mutable reference. See [`Borrow<T>`]
/// for more information on borrowing as another type.
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "BorrowMut"]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
pub const trait BorrowMut<Borrowed: ?Sized>: Borrow<Borrowed> {
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
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T: ?Sized> const Borrow<T> for T {
    #[rustc_diagnostic_item = "noop_method_borrow"]
    fn borrow(&self) -> &T {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T: ?Sized> const BorrowMut<T> for T {
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T: ?Sized> const Borrow<T> for &T {
    fn borrow(&self) -> &T {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T: ?Sized> const Borrow<T> for &mut T {
    fn borrow(&self) -> &T {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<T: ?Sized> const BorrowMut<T> for &mut T {
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}
