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
use std::default::Default;
use std::hash;
use std::{mem, raw, ptr, slice, vec};
use std::rt::heap::EMPTY;
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// A non-growable owned slice. This would preferably become `~[T]`
/// under DST.
#[unsafe_no_drop_flag] // data is set to null on destruction
pub struct OwnedSlice<T> {
    /// null iff len == 0
    data: *mut T,
    len: uint,
}

impl<T:fmt::Show> fmt::Show for OwnedSlice<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        try!("OwnedSlice {{".fmt(fmt));
        for i in self.iter() {
            try!(i.fmt(fmt));
        }
        try!("}}".fmt(fmt));
        Ok(())
    }
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
        OwnedSlice  { data: ptr::null_mut(), len: 0 }
    }

    #[inline(never)]
    pub fn from_vec(mut v: Vec<T>) -> OwnedSlice<T> {
        let len = v.len();

        if len == 0 {
            OwnedSlice::empty()
        } else {
            // drop excess capacity to avoid breaking sized deallocation
            v.shrink_to_fit();

            let p = v.as_mut_ptr();
            // we own the allocation now
            unsafe { mem::forget(v) }

            OwnedSlice { data: p, len: len }
        }
    }

    #[inline(never)]
    pub fn into_vec(self) -> Vec<T> {
        // null is ok, because len == 0 in that case, as required by Vec.
        unsafe {
            let ret = Vec::from_raw_parts(self.len, self.len, self.data);
            // the vector owns the allocation now
            mem::forget(self);
            ret
        }
    }

    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        let ptr = if self.data.is_null() {
            // length zero, i.e. this will never be read as a T.
            EMPTY as *const T
        } else {
            self.data as *const T
        };

        let slice: &[T] = unsafe {mem::transmute(raw::Slice {
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

    pub fn move_iter(self) -> vec::MoveItems<T> {
        self.into_vec().into_iter()
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
        OwnedSlice::from_vec(self.as_slice().to_vec())
    }
}

impl<S: hash::Writer, T: hash::Hash<S>> hash::Hash<S> for OwnedSlice<T> {
    fn hash(&self, state: &mut S) {
        self.as_slice().hash(state)
    }
}

impl<T: PartialEq> PartialEq for OwnedSlice<T> {
    fn eq(&self, other: &OwnedSlice<T>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for OwnedSlice<T> {}

impl<T> Collection for OwnedSlice<T> {
    fn len(&self) -> uint { self.len }
}

impl<T> FromIterator<T> for OwnedSlice<T> {
    fn from_iter<I: Iterator<T>>(mut iter: I) -> OwnedSlice<T> {
        OwnedSlice::from_vec(iter.collect())
    }
}

impl<S: Encoder<E>, T: Encodable<S, E>, E> Encodable<S, E> for OwnedSlice<T> {
    fn encode(&self, s: &mut S) -> Result<(), E> {
       self.as_slice().encode(s)
    }
}

impl<D: Decoder<E>, T: Decodable<D, E>, E> Decodable<D, E> for OwnedSlice<T> {
    fn decode(d: &mut D) -> Result<OwnedSlice<T>, E> {
        Ok(OwnedSlice::from_vec(match Decodable::decode(d) {
            Ok(t) => t,
            Err(e) => return Err(e)
        }))
    }
}
