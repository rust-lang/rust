// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::fmt::Show;
use std::hash::Hash;
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// An owned smart pointer.
pub struct P<T> {
    ptr: Box<T>
}

#[allow(non_snake_case)]
/// Construct a P<T> from a T value.
pub fn P<T: 'static>(value: T) -> P<T> {
    P {
        ptr: box value
    }
}

impl<T: 'static> P<T> {
    pub fn and_then<U>(self, f: |T| -> U) -> U {
        f(*self.ptr)
    }

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
