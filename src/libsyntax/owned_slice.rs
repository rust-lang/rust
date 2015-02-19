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
use std::fmt;
use std::iter::{IntoIterator, FromIterator};
use std::ops::Deref;
use std::vec;
use serialize::{Encodable, Decodable, Encoder, Decoder};

/// A non-growable owned slice. This is a separate type to allow the
/// representation to change.
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct OwnedSlice<T> {
    data: Box<[T]>
}

impl<T:fmt::Debug> fmt::Debug for OwnedSlice<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.data.fmt(fmt)
    }
}

impl<T> OwnedSlice<T> {
    pub fn empty() -> OwnedSlice<T> {
        OwnedSlice  { data: box [] }
    }

    #[inline(never)]
    pub fn from_vec(v: Vec<T>) -> OwnedSlice<T> {
        OwnedSlice { data: v.into_boxed_slice() }
    }

    #[inline(never)]
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
    }

    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        &*self.data
    }

    pub fn move_iter(self) -> vec::IntoIter<T> {
        self.into_vec().into_iter()
    }

    pub fn map<U, F: FnMut(&T) -> U>(&self, f: F) -> OwnedSlice<U> {
        self.iter().map(f).collect()
    }
}

impl<T> Deref for OwnedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> Default for OwnedSlice<T> {
    fn default() -> OwnedSlice<T> {
        OwnedSlice::empty()
    }
}

impl<T: Clone> Clone for OwnedSlice<T> {
    fn clone(&self) -> OwnedSlice<T> {
        OwnedSlice::from_vec(self.to_vec())
    }
}

impl<T> FromIterator<T> for OwnedSlice<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> OwnedSlice<T> {
        OwnedSlice::from_vec(iter.into_iter().collect())
    }
}

impl<T: Encodable> Encodable for OwnedSlice<T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&**self, s)
    }
}

impl<T: Decodable> Decodable for OwnedSlice<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<OwnedSlice<T>, D::Error> {
        Ok(OwnedSlice::from_vec(match Decodable::decode(d) {
            Ok(t) => t,
            Err(e) => return Err(e)
        }))
    }
}
