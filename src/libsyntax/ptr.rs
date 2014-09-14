// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The AST pointer
//!
//! Provides `P<T>`, a frozen owned smart pointer, as a replacement for `@T` in the AST.
//!
//! # Motivations and benefits
//!
//! * **Identity**: sharing AST nodes is problematic for the various analysis passes
//!   (e.g. one may be able to bypass the borrow checker with a shared `ExprAddrOf`
//!   node taking a mutable borrow). The only reason `@T` in the AST hasn't caused
//!   issues is because of inefficient folding passes which would always deduplicate
//!   any such shared nodes. Even if the AST were to switch to an arena, this would
//!   still hold, i.e. it couldn't use `&'a T`, but rather a wrapper like `P<'a, T>`.
//!
//! * **Immutability**: `P<T>` disallows mutating its inner `T`, unlike `Box<T>`
//!   (unless it contains an `Unsafe` interior, but that may be denied later).
//!   This mainly prevents mistakes, but can also enforces a kind of "purity".
//!
//! * **Efficiency**: folding can reuse allocation space for `P<T>` and `Vec<T>`,
//!   the latter even when the input and output types differ (as it would be the
//!   case with arenas or a GADT AST using type parameters to toggle features).
//!
//! * **Maintainability**: `P<T>` provides a fixed interface - `Deref`,
//!   `and_then` and `map` - which can remain fully functional even if the
//!   implementation changes (using a special thread-local heap, for example).
//!   Moreover, a switch to, e.g. `P<'a, T>` would be easy and mostly automated.

use std::fmt;
use std::fmt::Show;
use std::hash::Hash;
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// An owned smart pointer.
pub struct P<T> {
    ptr: Box<T>
}

#[allow(non_snake_case)]
/// Construct a `P<T>` from a `T` value.
pub fn P<T: 'static>(value: T) -> P<T> {
    P {
        ptr: box value
    }
}

impl<T: 'static> P<T> {
    /// Move out of the pointer.
    /// Intended for chaining transformations not covered by `map`.
    pub fn and_then<U>(self, f: |T| -> U) -> U {
        f(*self.ptr)
    }

    /// Transform the inner value, consuming `self` and producing a new `P<T>`.
    pub fn map(mut self, f: |T| -> T) -> P<T> {
        use std::{mem, ptr};
        unsafe {
            let p = &mut *self.ptr;
            // FIXME(#5016) this shouldn't need to zero to be safe.
            mem::move_val_init(p, f(ptr::read_and_zero(p)));
        }
        self
    }
}

impl<T> Deref<T> for P<T> {
    fn deref<'a>(&'a self) -> &'a T {
        &*self.ptr
    }
}

impl<T: 'static + Clone> Clone for P<T> {
    fn clone(&self) -> P<T> {
        P((**self).clone())
    }
}

impl<T: PartialEq> PartialEq for P<T> {
    fn eq(&self, other: &P<T>) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for P<T> {}

impl<T: Show> Show for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<S, T: Hash<S>> Hash<S> for P<T> {
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

impl<E, D: Decoder<E>, T: 'static + Decodable<D, E>> Decodable<D, E> for P<T> {
    fn decode(d: &mut D) -> Result<P<T>, E> {
        Decodable::decode(d).map(P)
    }
}

impl<E, S: Encoder<E>, T: Encodable<S, E>> Encodable<S, E> for P<T> {
    fn encode(&self, s: &mut S) -> Result<(), E> {
        (**self).encode(s)
    }
}
