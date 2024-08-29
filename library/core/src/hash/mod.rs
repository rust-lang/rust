//! Generic hashing support.
//!
//! This module provides a generic way to compute the [hash] of a value.
//! Hashes are most commonly used with [`HashMap`] and [`HashSet`].
//!
//! [hash]: https://en.wikipedia.org/wiki/Hash_function
//! [`HashMap`]: ../../std/collections/struct.HashMap.html
//! [`HashSet`]: ../../std/collections/struct.HashSet.html
//!
//! The simplest way to make a type hashable is to use `#[derive(Hash)]`:
//!
//! # Examples
//!
//! ```rust
//! use std::hash::{DefaultHasher, Hash, Hasher};
//!
//! #[derive(Hash)]
//! struct Person {
//!     id: u32,
//!     name: String,
//!     phone: u64,
//! }
//!
//! let person1 = Person {
//!     id: 5,
//!     name: "Janet".to_string(),
//!     phone: 555_666_7777,
//! };
//! let person2 = Person {
//!     id: 5,
//!     name: "Bob".to_string(),
//!     phone: 555_666_7777,
//! };
//!
//! assert!(calculate_hash(&person1) != calculate_hash(&person2));
//!
//! fn calculate_hash<T: Hash>(t: &T) -> u64 {
//!     let mut s = DefaultHasher::new();
//!     t.hash(&mut s);
//!     s.finish()
//! }
//! ```
//!
//! If you need more control over how a value is hashed, you need to implement
//! the [`Hash`] trait:
//!
//! ```rust
//! use std::hash::{DefaultHasher, Hash, Hasher};
//!
//! struct Person {
//!     id: u32,
//!     # #[allow(dead_code)]
//!     name: String,
//!     phone: u64,
//! }
//!
//! impl Hash for Person {
//!     fn hash<H: Hasher>(&self, state: &mut H) {
//!         self.id.hash(state);
//!         self.phone.hash(state);
//!     }
//! }
//!
//! let person1 = Person {
//!     id: 5,
//!     name: "Janet".to_string(),
//!     phone: 555_666_7777,
//! };
//! let person2 = Person {
//!     id: 5,
//!     name: "Bob".to_string(),
//!     phone: 555_666_7777,
//! };
//!
//! assert_eq!(calculate_hash(&person1), calculate_hash(&person2));
//!
//! fn calculate_hash<T: Hash>(t: &T) -> u64 {
//!     let mut s = DefaultHasher::new();
//!     t.hash(&mut s);
//!     s.finish()
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
pub use self::sip::SipHasher;
#[unstable(feature = "hashmap_internals", issue = "none")]
#[allow(deprecated)]
#[doc(hidden)]
pub use self::sip::SipHasher13;
use crate::{fmt, marker};

mod sip;

/// A hashable type.
///
/// Types implementing `Hash` are able to be [`hash`]ed with an instance of
/// [`Hasher`].
///
/// ## Implementing `Hash`
///
/// You can derive `Hash` with `#[derive(Hash)]` if all fields implement `Hash`.
/// The resulting hash will be the combination of the values from calling
/// [`hash`] on each field.
///
/// ```
/// #[derive(Hash)]
/// struct Rustacean {
///     name: String,
///     country: String,
/// }
/// ```
///
/// If you need more control over how a value is hashed, you can of course
/// implement the `Hash` trait yourself:
///
/// ```
/// use std::hash::{Hash, Hasher};
///
/// struct Person {
///     id: u32,
///     name: String,
///     phone: u64,
/// }
///
/// impl Hash for Person {
///     fn hash<H: Hasher>(&self, state: &mut H) {
///         self.id.hash(state);
///         self.phone.hash(state);
///     }
/// }
/// ```
///
/// ## `Hash` and `Eq`
///
/// When implementing both `Hash` and [`Eq`], it is important that the following
/// property holds:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes must also be equal.
/// [`HashMap`] and [`HashSet`] both rely on this behavior.
///
/// Thankfully, you won't need to worry about upholding this property when
/// deriving both [`Eq`] and `Hash` with `#[derive(PartialEq, Eq, Hash)]`.
///
/// Violating this property is a logic error. The behavior resulting from a logic error is not
/// specified, but users of the trait must ensure that such logic errors do *not* result in
/// undefined behavior. This means that `unsafe` code **must not** rely on the correctness of these
/// methods.
///
/// ## Prefix collisions
///
/// Implementations of `hash` should ensure that the data they
/// pass to the `Hasher` are prefix-free. That is,
/// values which are not equal should cause two different sequences of values to be written,
/// and neither of the two sequences should be a prefix of the other.
///
/// For example, the standard implementation of [`Hash` for `&str`][impl] passes an extra
/// `0xFF` byte to the `Hasher` so that the values `("ab", "c")` and `("a",
/// "bc")` hash differently.
///
/// ## Portability
///
/// Due to differences in endianness and type sizes, data fed by `Hash` to a `Hasher`
/// should not be considered portable across platforms. Additionally the data passed by most
/// standard library types should not be considered stable between compiler versions.
///
/// This means tests shouldn't probe hard-coded hash values or data fed to a `Hasher` and
/// instead should check consistency with `Eq`.
///
/// Serialization formats intended to be portable between platforms or compiler versions should
/// either avoid encoding hashes or only rely on `Hash` and `Hasher` implementations that
/// provide additional guarantees.
///
/// [`HashMap`]: ../../std/collections/struct.HashMap.html
/// [`HashSet`]: ../../std/collections/struct.HashSet.html
/// [`hash`]: Hash::hash
/// [impl]: ../../std/primitive.str.html#impl-Hash-for-str
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Hash"]
pub trait Hash {
    /// Feeds this value into the given [`Hasher`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::hash::{DefaultHasher, Hash, Hasher};
    ///
    /// let mut hasher = DefaultHasher::new();
    /// 7920.hash(&mut hasher);
    /// println!("Hash is {:x}!", hasher.finish());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn hash<H: Hasher>(&self, state: &mut H);

    /// Feeds a slice of this type into the given [`Hasher`].
    ///
    /// This method is meant as a convenience, but its implementation is
    /// also explicitly left unspecified. It isn't guaranteed to be
    /// equivalent to repeated calls of [`hash`] and implementations of
    /// [`Hash`] should keep that in mind and call [`hash`] themselves
    /// if the slice isn't treated as a whole unit in the [`PartialEq`]
    /// implementation.
    ///
    /// For example, a [`VecDeque`] implementation might naïvely call
    /// [`as_slices`] and then [`hash_slice`] on each slice, but this
    /// is wrong since the two slices can change with a call to
    /// [`make_contiguous`] without affecting the [`PartialEq`]
    /// result. Since these slices aren't treated as singular
    /// units, and instead part of a larger deque, this method cannot
    /// be used.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::hash::{DefaultHasher, Hash, Hasher};
    ///
    /// let mut hasher = DefaultHasher::new();
    /// let numbers = [6, 28, 496, 8128];
    /// Hash::hash_slice(&numbers, &mut hasher);
    /// println!("Hash is {:x}!", hasher.finish());
    /// ```
    ///
    /// [`VecDeque`]: ../../std/collections/struct.VecDeque.html
    /// [`as_slices`]: ../../std/collections/struct.VecDeque.html#method.as_slices
    /// [`make_contiguous`]: ../../std/collections/struct.VecDeque.html#method.make_contiguous
    /// [`hash`]: Hash::hash
    /// [`hash_slice`]: Hash::hash_slice
    #[stable(feature = "hash_slice", since = "1.3.0")]
    fn hash_slice<H: Hasher>(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        for piece in data {
            piece.hash(state)
        }
    }
}

// Separate module to reexport the macro `Hash` from prelude without the trait `Hash`.
pub(crate) mod macros {
    /// Derive macro generating an impl of the trait `Hash`.
    #[rustc_builtin_macro]
    #[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
    #[allow_internal_unstable(core_intrinsics)]
    pub macro Hash($item:item) {
        /* compiler built-in */
    }
}
#[stable(feature = "builtin_macro_prelude", since = "1.38.0")]
#[doc(inline)]
pub use macros::Hash;

/// A trait for hashing an arbitrary stream of bytes.
///
/// Instances of `Hasher` usually represent state that is changed while hashing
/// data.
///
/// `Hasher` provides a fairly basic interface for retrieving the generated hash
/// (with [`finish`]), and writing integers as well as slices of bytes into an
/// instance (with [`write`] and [`write_u8`] etc.). Most of the time, `Hasher`
/// instances are used in conjunction with the [`Hash`] trait.
///
/// This trait provides no guarantees about how the various `write_*` methods are
/// defined and implementations of [`Hash`] should not assume that they work one
/// way or another. You cannot assume, for example, that a [`write_u32`] call is
/// equivalent to four calls of [`write_u8`].  Nor can you assume that adjacent
/// `write` calls are merged, so it's possible, for example, that
/// ```
/// # fn foo(hasher: &mut impl std::hash::Hasher) {
/// hasher.write(&[1, 2]);
/// hasher.write(&[3, 4, 5, 6]);
/// # }
/// ```
/// and
/// ```
/// # fn foo(hasher: &mut impl std::hash::Hasher) {
/// hasher.write(&[1, 2, 3, 4]);
/// hasher.write(&[5, 6]);
/// # }
/// ```
/// end up producing different hashes.
///
/// Thus to produce the same hash value, [`Hash`] implementations must ensure
/// for equivalent items that exactly the same sequence of calls is made -- the
/// same methods with the same parameters in the same order.
///
/// # Examples
///
/// ```
/// use std::hash::{DefaultHasher, Hasher};
///
/// let mut hasher = DefaultHasher::new();
///
/// hasher.write_u32(1989);
/// hasher.write_u8(11);
/// hasher.write_u8(9);
/// hasher.write(b"Huh?");
///
/// println!("Hash is {:x}!", hasher.finish());
/// ```
///
/// [`finish`]: Hasher::finish
/// [`write`]: Hasher::write
/// [`write_u8`]: Hasher::write_u8
/// [`write_u32`]: Hasher::write_u32
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Hasher {
    /// Returns the hash value for the values written so far.
    ///
    /// Despite its name, the method does not reset the hasher’s internal
    /// state. Additional [`write`]s will continue from the current value.
    /// If you need to start a fresh hash value, you will have to create
    /// a new hasher.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::hash::{DefaultHasher, Hasher};
    ///
    /// let mut hasher = DefaultHasher::new();
    /// hasher.write(b"Cool!");
    ///
    /// println!("Hash is {:x}!", hasher.finish());
    /// ```
    ///
    /// [`write`]: Hasher::write
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    fn finish(&self) -> u64;

    /// Writes some data into this `Hasher`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::hash::{DefaultHasher, Hasher};
    ///
    /// let mut hasher = DefaultHasher::new();
    /// let data = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
    ///
    /// hasher.write(&data);
    ///
    /// println!("Hash is {:x}!", hasher.finish());
    /// ```
    ///
    /// # Note to Implementers
    ///
    /// You generally should not do length-prefixing as part of implementing
    /// this method.  It's up to the [`Hash`] implementation to call
    /// [`Hasher::write_length_prefix`] before sequences that need it.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn write(&mut self, bytes: &[u8]);

    /// Writes a single `u8` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u8(&mut self, i: u8) {
        self.write(&[i])
    }
    /// Writes a single `u16` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u16(&mut self, i: u16) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `u32` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u32(&mut self, i: u32) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `u64` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u64(&mut self, i: u64) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `u128` into this hasher.
    #[inline]
    #[stable(feature = "i128", since = "1.26.0")]
    fn write_u128(&mut self, i: u128) {
        self.write(&i.to_ne_bytes())
    }
    /// Writes a single `usize` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_usize(&mut self, i: usize) {
        self.write(&i.to_ne_bytes())
    }

    /// Writes a single `i8` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_i8(&mut self, i: i8) {
        self.write_u8(i as u8)
    }
    /// Writes a single `i16` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_i16(&mut self, i: i16) {
        self.write_u16(i as u16)
    }
    /// Writes a single `i32` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_i32(&mut self, i: i32) {
        self.write_u32(i as u32)
    }
    /// Writes a single `i64` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64)
    }
    /// Writes a single `i128` into this hasher.
    #[inline]
    #[stable(feature = "i128", since = "1.26.0")]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128)
    }
    /// Writes a single `isize` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_isize(&mut self, i: isize) {
        self.write_usize(i as usize)
    }

    /// Writes a length prefix into this hasher, as part of being prefix-free.
    ///
    /// If you're implementing [`Hash`] for a custom collection, call this before
    /// writing its contents to this `Hasher`.  That way
    /// `(collection![1, 2, 3], collection![4, 5])` and
    /// `(collection![1, 2], collection![3, 4, 5])` will provide different
    /// sequences of values to the `Hasher`
    ///
    /// The `impl<T> Hash for [T]` includes a call to this method, so if you're
    /// hashing a slice (or array or vector) via its `Hash::hash` method,
    /// you should **not** call this yourself.
    ///
    /// This method is only for providing domain separation.  If you want to
    /// hash a `usize` that represents part of the *data*, then it's important
    /// that you pass it to [`Hasher::write_usize`] instead of to this method.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(hasher_prefixfree_extras)]
    /// # // Stubs to make the `impl` below pass the compiler
    /// # #![allow(non_local_definitions)]
    /// # struct MyCollection<T>(Option<T>);
    /// # impl<T> MyCollection<T> {
    /// #     fn len(&self) -> usize { todo!() }
    /// # }
    /// # impl<'a, T> IntoIterator for &'a MyCollection<T> {
    /// #     type Item = T;
    /// #     type IntoIter = std::iter::Empty<T>;
    /// #     fn into_iter(self) -> Self::IntoIter { todo!() }
    /// # }
    ///
    /// use std::hash::{Hash, Hasher};
    /// impl<T: Hash> Hash for MyCollection<T> {
    ///     fn hash<H: Hasher>(&self, state: &mut H) {
    ///         state.write_length_prefix(self.len());
    ///         for elt in self {
    ///             elt.hash(state);
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Note to Implementers
    ///
    /// If you've decided that your `Hasher` is willing to be susceptible to
    /// Hash-DoS attacks, then you might consider skipping hashing some or all
    /// of the `len` provided in the name of increased performance.
    #[inline]
    #[unstable(feature = "hasher_prefixfree_extras", issue = "96762")]
    fn write_length_prefix(&mut self, len: usize) {
        self.write_usize(len);
    }

    /// Writes a single `str` into this hasher.
    ///
    /// If you're implementing [`Hash`], you generally do not need to call this,
    /// as the `impl Hash for str` does, so you should prefer that instead.
    ///
    /// This includes the domain separator for prefix-freedom, so you should
    /// **not** call `Self::write_length_prefix` before calling this.
    ///
    /// # Note to Implementers
    ///
    /// There are at least two reasonable default ways to implement this.
    /// Which one will be the default is not yet decided, so for now
    /// you probably want to override it specifically.
    ///
    /// ## The general answer
    ///
    /// It's always correct to implement this with a length prefix:
    ///
    /// ```
    /// # #![feature(hasher_prefixfree_extras)]
    /// # struct Foo;
    /// # impl std::hash::Hasher for Foo {
    /// # fn finish(&self) -> u64 { unimplemented!() }
    /// # fn write(&mut self, _bytes: &[u8]) { unimplemented!() }
    /// fn write_str(&mut self, s: &str) {
    ///     self.write_length_prefix(s.len());
    ///     self.write(s.as_bytes());
    /// }
    /// # }
    /// ```
    ///
    /// And, if your `Hasher` works in `usize` chunks, this is likely a very
    /// efficient way to do it, as anything more complicated may well end up
    /// slower than just running the round with the length.
    ///
    /// ## If your `Hasher` works byte-wise
    ///
    /// One nice thing about `str` being UTF-8 is that the `b'\xFF'` byte
    /// never happens.  That means that you can append that to the byte stream
    /// being hashed and maintain prefix-freedom:
    ///
    /// ```
    /// # #![feature(hasher_prefixfree_extras)]
    /// # struct Foo;
    /// # impl std::hash::Hasher for Foo {
    /// # fn finish(&self) -> u64 { unimplemented!() }
    /// # fn write(&mut self, _bytes: &[u8]) { unimplemented!() }
    /// fn write_str(&mut self, s: &str) {
    ///     self.write(s.as_bytes());
    ///     self.write_u8(0xff);
    /// }
    /// # }
    /// ```
    ///
    /// This does require that your implementation not add extra padding, and
    /// thus generally requires that you maintain a buffer, running a round
    /// only once that buffer is full (or `finish` is called).
    ///
    /// That's because if `write` pads data out to a fixed chunk size, it's
    /// likely that it does it in such a way that `"a"` and `"a\x00"` would
    /// end up hashing the same sequence of things, introducing conflicts.
    #[inline]
    #[unstable(feature = "hasher_prefixfree_extras", issue = "96762")]
    fn write_str(&mut self, s: &str) {
        self.write(s.as_bytes());
        self.write_u8(0xff);
    }
}

#[stable(feature = "indirect_hasher_impl", since = "1.22.0")]
impl<H: Hasher + ?Sized> Hasher for &mut H {
    fn finish(&self) -> u64 {
        (**self).finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        (**self).write(bytes)
    }
    fn write_u8(&mut self, i: u8) {
        (**self).write_u8(i)
    }
    fn write_u16(&mut self, i: u16) {
        (**self).write_u16(i)
    }
    fn write_u32(&mut self, i: u32) {
        (**self).write_u32(i)
    }
    fn write_u64(&mut self, i: u64) {
        (**self).write_u64(i)
    }
    fn write_u128(&mut self, i: u128) {
        (**self).write_u128(i)
    }
    fn write_usize(&mut self, i: usize) {
        (**self).write_usize(i)
    }
    fn write_i8(&mut self, i: i8) {
        (**self).write_i8(i)
    }
    fn write_i16(&mut self, i: i16) {
        (**self).write_i16(i)
    }
    fn write_i32(&mut self, i: i32) {
        (**self).write_i32(i)
    }
    fn write_i64(&mut self, i: i64) {
        (**self).write_i64(i)
    }
    fn write_i128(&mut self, i: i128) {
        (**self).write_i128(i)
    }
    fn write_isize(&mut self, i: isize) {
        (**self).write_isize(i)
    }
    fn write_length_prefix(&mut self, len: usize) {
        (**self).write_length_prefix(len)
    }
    fn write_str(&mut self, s: &str) {
        (**self).write_str(s)
    }
}

/// A trait for creating instances of [`Hasher`].
///
/// A `BuildHasher` is typically used (e.g., by [`HashMap`]) to create
/// [`Hasher`]s for each key such that they are hashed independently of one
/// another, since [`Hasher`]s contain state.
///
/// For each instance of `BuildHasher`, the [`Hasher`]s created by
/// [`build_hasher`] should be identical. That is, if the same stream of bytes
/// is fed into each hasher, the same output will also be generated.
///
/// # Examples
///
/// ```
/// use std::hash::{BuildHasher, Hasher, RandomState};
///
/// let s = RandomState::new();
/// let mut hasher_1 = s.build_hasher();
/// let mut hasher_2 = s.build_hasher();
///
/// hasher_1.write_u32(8128);
/// hasher_2.write_u32(8128);
///
/// assert_eq!(hasher_1.finish(), hasher_2.finish());
/// ```
///
/// [`build_hasher`]: BuildHasher::build_hasher
/// [`HashMap`]: ../../std/collections/struct.HashMap.html
#[stable(since = "1.7.0", feature = "build_hasher")]
pub trait BuildHasher {
    /// Type of the hasher that will be created.
    #[stable(since = "1.7.0", feature = "build_hasher")]
    type Hasher: Hasher;

    /// Creates a new hasher.
    ///
    /// Each call to `build_hasher` on the same instance should produce identical
    /// [`Hasher`]s.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::hash::{BuildHasher, RandomState};
    ///
    /// let s = RandomState::new();
    /// let new_s = s.build_hasher();
    /// ```
    #[stable(since = "1.7.0", feature = "build_hasher")]
    fn build_hasher(&self) -> Self::Hasher;

    /// Calculates the hash of a single value.
    ///
    /// This is intended as a convenience for code which *consumes* hashes, such
    /// as the implementation of a hash table or in unit tests that check
    /// whether a custom [`Hash`] implementation behaves as expected.
    ///
    /// This must not be used in any code which *creates* hashes, such as in an
    /// implementation of [`Hash`].  The way to create a combined hash of
    /// multiple values is to call [`Hash::hash`] multiple times using the same
    /// [`Hasher`], not to call this method repeatedly and combine the results.
    ///
    /// # Example
    ///
    /// ```
    /// use std::cmp::{max, min};
    /// use std::hash::{BuildHasher, Hash, Hasher};
    /// struct OrderAmbivalentPair<T: Ord>(T, T);
    /// impl<T: Ord + Hash> Hash for OrderAmbivalentPair<T> {
    ///     fn hash<H: Hasher>(&self, hasher: &mut H) {
    ///         min(&self.0, &self.1).hash(hasher);
    ///         max(&self.0, &self.1).hash(hasher);
    ///     }
    /// }
    ///
    /// // Then later, in a `#[test]` for the type...
    /// let bh = std::hash::RandomState::new();
    /// assert_eq!(
    ///     bh.hash_one(OrderAmbivalentPair(1, 2)),
    ///     bh.hash_one(OrderAmbivalentPair(2, 1))
    /// );
    /// assert_eq!(
    ///     bh.hash_one(OrderAmbivalentPair(10, 2)),
    ///     bh.hash_one(&OrderAmbivalentPair(2, 10))
    /// );
    /// ```
    #[stable(feature = "build_hasher_simple_hash_one", since = "1.71.0")]
    fn hash_one<T: Hash>(&self, x: T) -> u64
    where
        Self: Sized,
        Self::Hasher: Hasher,
    {
        let mut hasher = self.build_hasher();
        x.hash(&mut hasher);
        hasher.finish()
    }
}

/// Used to create a default [`BuildHasher`] instance for types that implement
/// [`Hasher`] and [`Default`].
///
/// `BuildHasherDefault<H>` can be used when a type `H` implements [`Hasher`] and
/// [`Default`], and you need a corresponding [`BuildHasher`] instance, but none is
/// defined.
///
/// Any `BuildHasherDefault` is [zero-sized]. It can be created with
/// [`default`][method.default]. When using `BuildHasherDefault` with [`HashMap`] or
/// [`HashSet`], this doesn't need to be done, since they implement appropriate
/// [`Default`] instances themselves.
///
/// # Examples
///
/// Using `BuildHasherDefault` to specify a custom [`BuildHasher`] for
/// [`HashMap`]:
///
/// ```
/// use std::collections::HashMap;
/// use std::hash::{BuildHasherDefault, Hasher};
///
/// #[derive(Default)]
/// struct MyHasher;
///
/// impl Hasher for MyHasher {
///     fn write(&mut self, bytes: &[u8]) {
///         // Your hashing algorithm goes here!
///        unimplemented!()
///     }
///
///     fn finish(&self) -> u64 {
///         // Your hashing algorithm goes here!
///         unimplemented!()
///     }
/// }
///
/// type MyBuildHasher = BuildHasherDefault<MyHasher>;
///
/// let hash_map = HashMap::<u32, u32, MyBuildHasher>::default();
/// ```
///
/// [method.default]: BuildHasherDefault::default
/// [`HashMap`]: ../../std/collections/struct.HashMap.html
/// [`HashSet`]: ../../std/collections/struct.HashSet.html
/// [zero-sized]: https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts
#[stable(since = "1.7.0", feature = "build_hasher")]
pub struct BuildHasherDefault<H>(marker::PhantomData<fn() -> H>);

impl<H> BuildHasherDefault<H> {
    /// Creates a new BuildHasherDefault for Hasher `H`.
    #[unstable(
        feature = "build_hasher_default_const_new",
        issue = "123197",
        reason = "recently added"
    )]
    pub const fn new() -> Self {
        BuildHasherDefault(marker::PhantomData)
    }
}

#[stable(since = "1.9.0", feature = "core_impl_debug")]
impl<H> fmt::Debug for BuildHasherDefault<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuildHasherDefault").finish()
    }
}

#[stable(since = "1.7.0", feature = "build_hasher")]
impl<H: Default + Hasher> BuildHasher for BuildHasherDefault<H> {
    type Hasher = H;

    fn build_hasher(&self) -> H {
        H::default()
    }
}

#[stable(since = "1.7.0", feature = "build_hasher")]
impl<H> Clone for BuildHasherDefault<H> {
    fn clone(&self) -> BuildHasherDefault<H> {
        BuildHasherDefault(marker::PhantomData)
    }
}

#[stable(since = "1.7.0", feature = "build_hasher")]
impl<H> Default for BuildHasherDefault<H> {
    fn default() -> BuildHasherDefault<H> {
        Self::new()
    }
}

#[stable(since = "1.29.0", feature = "build_hasher_eq")]
impl<H> PartialEq for BuildHasherDefault<H> {
    fn eq(&self, _other: &BuildHasherDefault<H>) -> bool {
        true
    }
}

#[stable(since = "1.29.0", feature = "build_hasher_eq")]
impl<H> Eq for BuildHasherDefault<H> {}

mod impls {
    use super::*;
    use crate::{mem, slice};

    macro_rules! impl_write {
        ($(($ty:ident, $meth:ident),)*) => {$(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Hash for $ty {
                #[inline]
                fn hash<H: Hasher>(&self, state: &mut H) {
                    state.$meth(*self)
                }

                #[inline]
                fn hash_slice<H: Hasher>(data: &[$ty], state: &mut H) {
                    let newlen = mem::size_of_val(data);
                    let ptr = data.as_ptr() as *const u8;
                    // SAFETY: `ptr` is valid and aligned, as this macro is only used
                    // for numeric primitives which have no padding. The new slice only
                    // spans across `data` and is never mutated, and its total size is the
                    // same as the original `data` so it can't be over `isize::MAX`.
                    state.write(unsafe { slice::from_raw_parts(ptr, newlen) })
                }
            }
        )*}
    }

    impl_write! {
        (u8, write_u8),
        (u16, write_u16),
        (u32, write_u32),
        (u64, write_u64),
        (usize, write_usize),
        (i8, write_i8),
        (i16, write_i16),
        (i32, write_i32),
        (i64, write_i64),
        (isize, write_isize),
        (u128, write_u128),
        (i128, write_i128),
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Hash for bool {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u8(*self as u8)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Hash for char {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u32(*self as u32)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Hash for str {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_str(self);
        }
    }

    #[stable(feature = "never_hash", since = "1.29.0")]
    impl Hash for ! {
        #[inline]
        fn hash<H: Hasher>(&self, _: &mut H) {
            *self
        }
    }

    macro_rules! impl_hash_tuple {
        () => (
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Hash for () {
                #[inline]
                fn hash<H: Hasher>(&self, _state: &mut H) {}
            }
        );

        ( $($name:ident)+) => (
            maybe_tuple_doc! {
                $($name)+ @
                #[stable(feature = "rust1", since = "1.0.0")]
                impl<$($name: Hash),+> Hash for ($($name,)+) where last_type!($($name,)+): ?Sized {
                    #[allow(non_snake_case)]
                    #[inline]
                    fn hash<S: Hasher>(&self, state: &mut S) {
                        let ($(ref $name,)+) = *self;
                        $($name.hash(state);)+
                    }
                }
            }
        );
    }

    macro_rules! maybe_tuple_doc {
        ($a:ident @ #[$meta:meta] $item:item) => {
            #[doc(fake_variadic)]
            #[doc = "This trait is implemented for tuples up to twelve items long."]
            #[$meta]
            $item
        };
        ($a:ident $($rest_a:ident)+ @ #[$meta:meta] $item:item) => {
            #[doc(hidden)]
            #[$meta]
            $item
        };
    }

    macro_rules! last_type {
        ($a:ident,) => { $a };
        ($a:ident, $($rest_a:ident,)+) => { last_type!($($rest_a,)+) };
    }

    impl_hash_tuple! {}
    impl_hash_tuple! { T }
    impl_hash_tuple! { T B }
    impl_hash_tuple! { T B C }
    impl_hash_tuple! { T B C D }
    impl_hash_tuple! { T B C D E }
    impl_hash_tuple! { T B C D E F }
    impl_hash_tuple! { T B C D E F G }
    impl_hash_tuple! { T B C D E F G H }
    impl_hash_tuple! { T B C D E F G H I }
    impl_hash_tuple! { T B C D E F G H I J }
    impl_hash_tuple! { T B C D E F G H I J K }
    impl_hash_tuple! { T B C D E F G H I J K L }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: Hash> Hash for [T] {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_length_prefix(self.len());
            Hash::hash_slice(self, state)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized + Hash> Hash for &T {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            (**self).hash(state);
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized + Hash> Hash for &mut T {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            (**self).hash(state);
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized> Hash for *const T {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            let (address, metadata) = self.to_raw_parts();
            state.write_usize(address.addr());
            metadata.hash(state);
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: ?Sized> Hash for *mut T {
        #[inline]
        fn hash<H: Hasher>(&self, state: &mut H) {
            let (address, metadata) = self.to_raw_parts();
            state.write_usize(address.addr());
            metadata.hash(state);
        }
    }
}
