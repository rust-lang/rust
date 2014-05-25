// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Generic hashing support.
 *
 * This module provides a generic way to compute the hash of a value. The
 * simplest way to make a type hashable is to use `#[deriving(Hash)]`:
 *
 * # Example
 *
 * ```rust
 * use std::hash;
 * use std::hash::Hash;
 *
 * #[deriving(Hash)]
 * struct Person {
 *     id: uint,
 *     name: String,
 *     phone: u64,
 * }
 *
 * let person1 = Person { id: 5, name: "Janet".to_string(), phone: 555_666_7777 };
 * let person2 = Person { id: 5, name: "Bob".to_string(), phone: 555_666_7777 };
 *
 * assert!(hash::hash(&person1) != hash::hash(&person2));
 * ```
 *
 * If you need more control over how a value is hashed, you need to implement
 * the trait `Hash`:
 *
 * ```rust
 * use std::hash;
 * use std::hash::Hash;
 * use std::hash::sip::SipState;
 *
 * struct Person {
 *     id: uint,
 *     name: String,
 *     phone: u64,
 * }
 *
 * impl Hash for Person {
 *     fn hash(&self, state: &mut SipState) {
 *         self.id.hash(state);
 *         self.phone.hash(state);
 *     }
 * }
 *
 * let person1 = Person { id: 5, name: "Janet".to_string(), phone: 555_666_7777 };
 * let person2 = Person { id: 5, name: "Bob".to_string(), phone: 555_666_7777 };
 *
 * assert!(hash::hash(&person1) == hash::hash(&person2));
 * ```
 */

#![allow(unused_must_use)]

use container::Container;
use intrinsics::TypeId;
use iter::Iterator;
use option::{Option, Some, None};
use owned::Box;
use rc::Rc;
use result::{Result, Ok, Err};
use slice::{Vector, ImmutableVector};
use str::{Str, StrSlice};
use vec::Vec;

/// Reexport the `sip::hash` function as our default hasher.
pub use hash = self::sip::hash;

pub use Writer = io::Writer;

pub mod sip;

/// A trait that represents a hashable type. The `S` type parameter is an
/// abstract hash state that is used by the `Hash` to compute the hash.
/// It defaults to `std::hash::sip::SipState`.
pub trait Hash<S = sip::SipState> {
    /// Compute a hash of the value.
    fn hash(&self, state: &mut S);
}

/// A trait that computes a hash for a value. The main users of this trait are
/// containers like `HashMap`, which need a generic way hash multiple types.
pub trait Hasher<S> {
    /// Compute a hash of the value.
    fn hash<T: Hash<S>>(&self, value: &T) -> u64;
}

//////////////////////////////////////////////////////////////////////////////

macro_rules! impl_hash(
    ( $( $ty:ty => $method:ident;)* ) => (
        $(
            impl<S: Writer> Hash<S> for $ty {
                #[inline]
                fn hash(&self, state: &mut S) {
                    state.$method(*self);
                }
            }
        )*
    )
)

impl_hash!(
    u8 => write_u8;
    u16 => write_le_u16;
    u32 => write_le_u32;
    u64 => write_le_u64;
    uint => write_le_uint;
    i8 => write_i8;
    i16 => write_le_i16;
    i32 => write_le_i32;
    i64 => write_le_i64;
    int => write_le_int;
)

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

impl<'a, S: Writer> Hash<S> for &'a str {
    #[inline]
    fn hash(&self, state: &mut S) {
        state.write(self.as_bytes());
        state.write_u8(0xFF);
    }
}

macro_rules! impl_hash_tuple(
    () => (
        impl<S: Writer> Hash<S> for () {
            #[inline]
            fn hash(&self, state: &mut S) {
                state.write([]);
            }
        }
    );

    ($A:ident $($B:ident)*) => (
        impl<
            S: Writer,
            $A: Hash<S> $(, $B: Hash<S>)*
        > Hash<S> for ($A, $($B),*) {
            #[inline]
            fn hash(&self, state: &mut S) {
                match *self {
                    (ref $A, $(ref $B),*) => {
                        $A.hash(state);
                        $(
                            $B.hash(state);
                        )*
                    }
                }
            }
        }

        impl_hash_tuple!($($B)*)
    );
)

impl_hash_tuple!(a0 a1 a2 a3 a4 a5 a6 a7)

impl<'a, S: Writer, T: Hash<S>> Hash<S> for &'a [T] {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.len().hash(state);
        for elt in self.iter() {
            elt.hash(state);
        }
    }
}


impl<'a, S: Writer, T: Hash<S>> Hash<S> for &'a mut [T] {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.as_slice().hash(state);
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for ~[T] {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.as_slice().hash(state);
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for Vec<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.as_slice().hash(state);
    }
}

impl<'a, S: Writer, T: Hash<S>> Hash<S> for &'a T {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<'a, S: Writer, T: Hash<S>> Hash<S> for &'a mut T {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for Box<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for @T {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for Rc<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for Option<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        match *self {
            Some(ref x) => {
                0u8.hash(state);
                x.hash(state);
            }
            None => {
                1u8.hash(state);
            }
        }
    }
}

impl<S: Writer, T> Hash<S> for *T {
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

impl<S: Writer, T: Hash<S>, U: Hash<S>> Hash<S> for Result<T, U> {
    #[inline]
    fn hash(&self, state: &mut S) {
        match *self {
            Ok(ref t) => { 1u.hash(state); t.hash(state); }
            Err(ref t) => { 2u.hash(state); t.hash(state); }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use mem;
    use io::{IoResult, Writer};
    use iter::{Iterator};
    use option::{Some, None};
    use result::Ok;
    use slice::ImmutableVector;

    use super::{Hash, Hasher};

    struct MyWriterHasher;

    impl Hasher<MyWriter> for MyWriterHasher {
        fn hash<T: Hash<MyWriter>>(&self, value: &T) -> u64 {
            let mut state = MyWriter { hash: 0 };
            value.hash(&mut state);
            state.hash
        }
    }

    struct MyWriter {
        hash: u64,
    }

    impl Writer for MyWriter {
        // Most things we'll just add up the bytes.
        fn write(&mut self, buf: &[u8]) -> IoResult<()> {
            for byte in buf.iter() {
                self.hash += *byte as u64;
            }
            Ok(())
        }
    }

    #[test]
    fn test_writer_hasher() {
        let hasher = MyWriterHasher;

        assert_eq!(hasher.hash(&()), 0);

        assert_eq!(hasher.hash(&5u8), 5);
        assert_eq!(hasher.hash(&5u16), 5);
        assert_eq!(hasher.hash(&5u32), 5);
        assert_eq!(hasher.hash(&5u64), 5);
        assert_eq!(hasher.hash(&5u), 5);

        assert_eq!(hasher.hash(&5i8), 5);
        assert_eq!(hasher.hash(&5i16), 5);
        assert_eq!(hasher.hash(&5i32), 5);
        assert_eq!(hasher.hash(&5i64), 5);
        assert_eq!(hasher.hash(&5i), 5);

        assert_eq!(hasher.hash(&false), 0);
        assert_eq!(hasher.hash(&true), 1);

        assert_eq!(hasher.hash(&'a'), 97);

        assert_eq!(hasher.hash(&("a")), 97 + 0xFF);
        assert_eq!(hasher.hash(& &[1u8, 2u8, 3u8]), 9);

        unsafe {
            let ptr: *int = mem::transmute(5);
            assert_eq!(hasher.hash(&ptr), 5);
        }

        unsafe {
            let ptr: *mut int = mem::transmute(5);
            assert_eq!(hasher.hash(&ptr), 5);
        }
    }

    struct Custom {
        hash: u64
    }

    impl Hash<u64> for Custom {
        fn hash(&self, state: &mut u64) {
            *state = self.hash;
        }
    }

    #[test]
    fn test_custom_state() {
        let custom = Custom { hash: 5 };
        let mut state = 0;
        custom.hash(&mut state);
        assert_eq!(state, 5);
    }
}
