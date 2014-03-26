// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::default::Default;
use std::hash::Hash;
use std::{cast, mem, raw, ptr, slice};
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// A non-growable owned slice. This would preferably become `~[T]`
/// under DST.
#[unsafe_no_drop_flag] // data is set to null on destruction
pub struct OwnedSlice<T> {
    /// null iff len == 0
    priv data: *mut T,
    priv len: uint,
}

#[unsafe_destructor]
impl<T> Drop for OwnedSlice<T> {
    fn drop(&mut self) {
        if self.data.is_null() { return }

        // extract the vector
        let v = mem::replace(self, OwnedSlice::empty());
        // free via the Vec destructor
        v.into_vec();
    }
}

impl<T> OwnedSlice<T> {
    pub fn empty() -> OwnedSlice<T> {
        OwnedSlice  { data: ptr::mut_null(), len: 0 }
    }

    #[inline(never)]
    pub fn from_vec(mut v: Vec<T>) -> OwnedSlice<T> {
        let len = v.len();

        if len == 0 {
            OwnedSlice::empty()
        } else {
            let p = v.as_mut_ptr();
            // we own the allocation now
            unsafe {cast::forget(v)}

            OwnedSlice { data: p, len: len }
        }
    }

    #[inline(never)]
    pub fn into_vec(self) -> Vec<T> {
        // null is ok, because len == 0 in that case, as required by Vec.
        unsafe {
            let ret = Vec::from_raw_parts(self.len, self.len, self.data);
            // the vector owns the allocation now
            cast::forget(self);
            ret
        }
    }

    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        static PTR_MARKER: u8 = 0;
        let ptr = if self.data.is_null() {
            // length zero, i.e. this will never be read as a T.
            &PTR_MARKER as *u8 as *T
        } else {
            self.data as *T
        };

        let slice: &[T] = unsafe {cast::transmute(raw::Slice {
            data: ptr,
            len: self.len
        })};

        slice
    }

    pub fn get<'a>(&'a self, i: uint) -> &'a T {
        self.as_slice().get(i).expect("OwnedSlice: index out of bounds")
    }

    pub fn iter<'r>(&'r self) -> slice::Items<'r, T> {
        self.as_slice().iter()
    }

    pub fn map<U>(&self, f: |&T| -> U) -> OwnedSlice<U> {
        self.iter().map(f).collect()
    }
}

impl<T> Default for OwnedSlice<T> {
    fn default() -> OwnedSlice<T> {
        OwnedSlice::empty()
    }
}

impl<T: Clone> Clone for OwnedSlice<T> {
    fn clone(&self) -> OwnedSlice<T> {
        OwnedSlice::from_vec(Vec::from_slice(self.as_slice()))
    }
}

impl<S: Writer, T: Hash<S>> Hash<S> for OwnedSlice<T> {
    fn hash(&self, state: &mut S) {
        self.as_slice().hash(state)
    }
}

impl<T: Eq> Eq for OwnedSlice<T> {
    fn eq(&self, other: &OwnedSlice<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: TotalEq> TotalEq for OwnedSlice<T> {}

impl<T> Container for OwnedSlice<T> {
    fn len(&self) -> uint { self.len }
}

impl<T> FromIterator<T> for OwnedSlice<T> {
    fn from_iterator<I: Iterator<T>>(mut iter: I) -> OwnedSlice<T> {
        OwnedSlice::from_vec(iter.collect())
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for OwnedSlice<T> {
    fn encode(&self, s: &mut S) {
       self.as_slice().encode(s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for OwnedSlice<T> {
    fn decode(d: &mut D) -> OwnedSlice<T> {
        OwnedSlice::from_vec(Decodable::decode(d))
    }
}
