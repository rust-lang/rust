// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generic hashing support.
//!
//! This module provides a generic way to compute the hash of a value. The
//! simplest way to make a type hashable is to use `#[derive(Hash)]`:
//!
//! # Examples
//!
//! ```rust
//! use std::collections::hash_map::DefaultHasher;
//! use std::hash::{Hash, Hasher};
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
//! [`Hash`]: trait.Hash.html
//!
//! ```rust
//! use std::collections::hash_map::DefaultHasher;
//! use std::hash::{Hash, Hasher};
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

use fmt;
use marker;
use mem;

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
pub use self::sip::SipHasher;

#[unstable(feature = "sip_hash_13", issue = "29754")]
#[allow(deprecated)]
pub use self::sip::{SipHasher13, SipHasher24};

mod sip;

/// A hashable type.
///
/// The `H` type parameter is an abstract hash state that is used by the `Hash`
/// to compute the hash.
///
/// If you are also implementing [`Eq`], there is an additional property that
/// is important:
///
/// ```text
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// In other words, if two keys are equal, their hashes should also be equal.
/// [`HashMap`] and [`HashSet`] both rely on this behavior.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]` if all fields implement `Hash`.
/// When `derive`d, the resulting hash will be the combination of the values
/// from calling [`.hash`] on each field.
///
/// ## How can I implement `Hash`?
///
/// If you need more control over how a value is hashed, you need to implement
/// the `Hash` trait:
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
/// [`Eq`]: ../../std/cmp/trait.Eq.html
/// [`HashMap`]: ../../std/collections/struct.HashMap.html
/// [`HashSet`]: ../../std/collections/struct.HashSet.html
/// [`.hash`]: #tymethod.hash
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Hash {
    /// Feeds this value into the state given, updating the hasher as necessary.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn hash<H: Hasher>(&self, state: &mut H);

    /// Feeds a slice of this type into the state provided.
    #[stable(feature = "hash_slice", since = "1.3.0")]
    fn hash_slice<H: Hasher>(data: &[Self], state: &mut H)
        where Self: Sized
    {
        for piece in data {
            piece.hash(state);
        }
    }
}

/// A trait which represents the ability to hash an arbitrary stream of bytes.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Hasher {
    /// Completes a round of hashing, producing the output hash generated.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn finish(&self) -> u64;

    /// Writes some data into this `Hasher`.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn write(&mut self, bytes: &[u8]);

    /// Write a single `u8` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u8(&mut self, i: u8) {
        self.write(&[i])
    }
    /// Writes a single `u16` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u16(&mut self, i: u16) {
        self.write(&unsafe { mem::transmute::<_, [u8; 2]>(i) })
    }
    /// Writes a single `u32` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u32(&mut self, i: u32) {
        self.write(&unsafe { mem::transmute::<_, [u8; 4]>(i) })
    }
    /// Writes a single `u64` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_u64(&mut self, i: u64) {
        self.write(&unsafe { mem::transmute::<_, [u8; 8]>(i) })
    }
    /// Writes a single `u128` into this hasher.
    #[inline]
    #[unstable(feature = "i128", issue = "35118")]
    fn write_u128(&mut self, i: u128) {
        self.write(&unsafe { mem::transmute::<_, [u8; 16]>(i) })
    }
    /// Writes a single `usize` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_usize(&mut self, i: usize) {
        let bytes = unsafe {
            ::slice::from_raw_parts(&i as *const usize as *const u8, mem::size_of::<usize>())
        };
        self.write(bytes);
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
    #[unstable(feature = "i128", issue = "35118")]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128)
    }
    /// Writes a single `isize` into this hasher.
    #[inline]
    #[stable(feature = "hasher_write", since = "1.3.0")]
    fn write_isize(&mut self, i: isize) {
        self.write_usize(i as usize)
    }
}

/// A `BuildHasher` is typically used as a factory for instances of `Hasher`
/// which a `HashMap` can then use to hash keys independently.
///
/// Note that for each instance of `BuildHasher`, the created hashers should be
/// identical. That is, if the same stream of bytes is fed into each hasher, the
/// same output will also be generated.
#[stable(since = "1.7.0", feature = "build_hasher")]
pub trait BuildHasher {
    /// Type of the hasher that will be created.
    #[stable(since = "1.7.0", feature = "build_hasher")]
    type Hasher: Hasher;

    /// Creates a new hasher.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::hash_map::RandomState;
    /// use std::hash::BuildHasher;
    ///
    /// let s = RandomState::new();
    /// let new_s = s.build_hasher();
    /// ```
    #[stable(since = "1.7.0", feature = "build_hasher")]
    fn build_hasher(&self) -> Self::Hasher;
}

/// Used to create a default [`BuildHasher`] instance for types that implement
/// [`Hasher`] and [`Default`].
///
/// `BuildHasherDefault<H>` can be used when a type `H` implements [`Hasher`] and
/// [`Default`], and you need a corresponding [`BuildHasher`] instance, but none is
/// defined.
///
/// Any `BuildHasherDefault` is [zero-sized]. It can be created with
/// [`default`][method.Default]. When using `BuildHasherDefault` with [`HashMap`] or
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
/// [`BuildHasher`]: trait.BuildHasher.html
/// [`Default`]: ../default/trait.Default.html
/// [method.default]: #method.default
/// [`Hasher`]: trait.Hasher.html
/// [`HashMap`]: ../../std/collections/struct.HashMap.html
/// [`HashSet`]: ../../std/collections/struct.HashSet.html
/// [zero-sized]: https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts
#[stable(since = "1.7.0", feature = "build_hasher")]
pub struct BuildHasherDefault<H>(marker::PhantomData<H>);

#[stable(since = "1.9.0", feature = "core_impl_debug")]
impl<H> fmt::Debug for BuildHasherDefault<H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("BuildHasherDefault")
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
        BuildHasherDefault(marker::PhantomData)
    }
}

//////////////////////////////////////////////////////////////////////////////

mod impls {
    use mem;
    use slice;
    use super::*;

    macro_rules! impl_write {
        ($(($ty:ident, $meth:ident),)*) => {$(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Hash for $ty {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    state.$meth(*self)
                }

                fn hash_slice<H: Hasher>(data: &[$ty], state: &mut H) {
                    let newlen = data.len() * mem::size_of::<$ty>();
                    let ptr = data.as_ptr() as *const u8;
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
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u8(*self as u8)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Hash for char {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_u32(*self as u32)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Hash for str {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write(self.as_bytes());
            state.write_u8(0xff)
        }
    }

    macro_rules! impl_hash_tuple {
        () => (
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Hash for () {
                fn hash<H: Hasher>(&self, _state: &mut H) {}
            }
        );

        ( $($name:ident)+) => (
            #[stable(feature = "rust1", since = "1.0.0")]
            impl<$($name: Hash),*> Hash for ($($name,)*) {
                #[allow(non_snake_case)]
                fn hash<S: Hasher>(&self, state: &mut S) {
                    let ($(ref $name,)*) = *self;
                    $($name.hash(state);)*
                }
            }
        );
    }

    impl_hash_tuple! {}
    impl_hash_tuple! { A }
    impl_hash_tuple! { A B }
    impl_hash_tuple! { A B C }
    impl_hash_tuple! { A B C D }
    impl_hash_tuple! { A B C D E }
    impl_hash_tuple! { A B C D E F }
    impl_hash_tuple! { A B C D E F G }
    impl_hash_tuple! { A B C D E F G H }
    impl_hash_tuple! { A B C D E F G H I }
    impl_hash_tuple! { A B C D E F G H I J }
    impl_hash_tuple! { A B C D E F G H I J K }
    impl_hash_tuple! { A B C D E F G H I J K L }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T: Hash> Hash for [T] {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.len().hash(state);
            Hash::hash_slice(self, state)
        }
    }


    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, T: ?Sized + Hash> Hash for &'a T {
        fn hash<H: Hasher>(&self, state: &mut H) {
            (**self).hash(state);
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<'a, T: ?Sized + Hash> Hash for &'a mut T {
        fn hash<H: Hasher>(&self, state: &mut H) {
            (**self).hash(state);
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T> Hash for *const T {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_usize(*self as usize)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<T> Hash for *mut T {
        fn hash<H: Hasher>(&self, state: &mut H) {
            state.write_usize(*self as usize)
        }
    }
}
