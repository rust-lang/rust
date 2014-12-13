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
//! simplest way to make a type hashable is to use `#[deriving(Hash)]`:
//!
//! # Examples
//!
//! ```rust
//! use std::hash;
//! use std::hash::Hash;
//!
//! #[deriving(Hash)]
//! struct Person {
//!     id: uint,
//!     name: String,
//!     phone: u64,
//! }
//!
//! let person1 = Person { id: 5, name: "Janet".to_string(), phone: 555_666_7777 };
//! let person2 = Person { id: 5, name: "Bob".to_string(), phone: 555_666_7777 };
//!
//! assert!(hash::hash(&person1) != hash::hash(&person2));
//! ```
//!
//! If you need more control over how a value is hashed, you need to implement
//! the trait `Hash`:
//!
//! ```rust
//! use std::hash;
//! use std::hash::Hash;
//! use std::hash::sip::SipState;
//!
//! struct Person {
//!     id: uint,
//!     name: String,
//!     phone: u64,
//! }
//!
//! impl Hash for Person {
//!     fn hash(&self, state: &mut SipState) {
//!         self.id.hash(state);
//!         self.phone.hash(state);
//!     }
//! }
//!
//! let person1 = Person { id: 5, name: "Janet".to_string(), phone: 555_666_7777 };
//! let person2 = Person { id: 5, name: "Bob".to_string(), phone: 555_666_7777 };
//!
//! assert!(hash::hash(&person1) == hash::hash(&person2));
//! ```

#![allow(unused_must_use)]

use prelude::*;

use borrow::{Cow, ToOwned};
use intrinsics::TypeId;
use mem;
use num::Int;

/// Reexport the `sip::hash` function as our default hasher.
pub use self::sip::hash as hash;

pub mod sip;

/// A hashable type. The `S` type parameter is an abstract hash state that is
/// used by the `Hash` to compute the hash. It defaults to
/// `std::hash::sip::SipState`.
pub trait Hash<S = sip::SipState> for Sized? {
    /// Computes the hash of a value.
    fn hash(&self, state: &mut S);
}

/// A trait that computes a hash for a value. The main users of this trait are
/// containers like `HashMap`, which need a generic way hash multiple types.
pub trait Hasher<S> {
    /// Compute the hash of a value.
    fn hash<Sized? T: Hash<S>>(&self, value: &T) -> u64;
}

#[allow(missing_docs)]
pub trait Writer {
    fn write(&mut self, bytes: &[u8]);
}

//////////////////////////////////////////////////////////////////////////////

macro_rules! impl_hash {
    ($ty:ident, $uty:ident) => {
        impl<S: Writer> Hash<S> for $ty {
            #[inline]
            fn hash(&self, state: &mut S) {
                let a: [u8, ..::$ty::BYTES] = unsafe {
                    mem::transmute((*self as $uty).to_le() as $ty)
                };
                state.write(a.as_slice())
            }
        }
    }
}

impl_hash!(u8, u8)
impl_hash!(u16, u16)
impl_hash!(u32, u32)
impl_hash!(u64, u64)
impl_hash!(uint, uint)
impl_hash!(i8, u8)
impl_hash!(i16, u16)
impl_hash!(i32, u32)
impl_hash!(i64, u64)
impl_hash!(int, uint)

impl<S: Writer> Hash<S> for bool {
    #[inline]
    fn hash(&self, state: &mut S) {
        (*self as u8).hash(state);
    }
}

impl<S: Writer> Hash<S> for char {
    #[inline]
    fn hash(&self, state: &mut S) {
        (*self as u32).hash(state);
    }
}

impl<S: Writer> Hash<S> for str {
    #[inline]
    fn hash(&self, state: &mut S) {
        state.write(self.as_bytes());
        0xffu8.hash(state)
    }
}

macro_rules! impl_hash_tuple(
    () => (
        impl<S: Writer> Hash<S> for () {
            #[inline]
            fn hash(&self, state: &mut S) {
                state.write(&[]);
            }
        }
    );

    ( $($name:ident)+) => (
        impl<S: Writer, $($name: Hash<S>),*> Hash<S> for ($($name,)*) {
            #[inline]
            #[allow(non_snake_case)]
            fn hash(&self, state: &mut S) {
                match *self {
                    ($(ref $name,)*) => {
                        $(
                            $name.hash(state);
                        )*
                    }
                }
            }
        }
    );
)

impl_hash_tuple!()
impl_hash_tuple!(A)
impl_hash_tuple!(A B)
impl_hash_tuple!(A B C)
impl_hash_tuple!(A B C D)
impl_hash_tuple!(A B C D E)
impl_hash_tuple!(A B C D E F)
impl_hash_tuple!(A B C D E F G)
impl_hash_tuple!(A B C D E F G H)
impl_hash_tuple!(A B C D E F G H I)
impl_hash_tuple!(A B C D E F G H I J)
impl_hash_tuple!(A B C D E F G H I J K)
impl_hash_tuple!(A B C D E F G H I J K L)

impl<S: Writer, T: Hash<S>> Hash<S> for [T] {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.len().hash(state);
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}


impl<'a, S: Writer, Sized? T: Hash<S>> Hash<S> for &'a T {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<'a, S: Writer, Sized? T: Hash<S>> Hash<S> for &'a mut T {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<S: Writer, T> Hash<S> for *const T {
    #[inline]
    fn hash(&self, state: &mut S) {
        // NB: raw-pointer Hash does _not_ dereference
        // to the target; it just gives you the pointer-bytes.
        (*self as uint).hash(state);
    }
}

impl<S: Writer, T> Hash<S> for *mut T {
    #[inline]
    fn hash(&self, state: &mut S) {
        // NB: raw-pointer Hash does _not_ dereference
        // to the target; it just gives you the pointer-bytes.
        (*self as uint).hash(state);
    }
}

impl<S: Writer> Hash<S> for TypeId {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.hash().hash(state)
    }
}

impl<'a, T, Sized? B, S> Hash<S> for Cow<'a, T, B> where B: Hash<S> + ToOwned<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        Hash::hash(&**self, state)
    }
}
