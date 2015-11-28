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
use std::iter::FromIterator;
use std::ops::Deref;
use std::ptr;
use std::slice;
use std::vec;

use util::MoveMap;

/// A non-growable owned slice. This is a separate type to allow the
/// representation to change.
#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct OwnedSlice<T> {
    data: Box<[T]>
}

impl<T> OwnedSlice<T> {
    pub fn new() -> OwnedSlice<T> {
        OwnedSlice  { data: Box::new([]) }
    }
}

impl<T> Deref for OwnedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.data
    }
}

impl<T> From<Vec<T>> for OwnedSlice<T> {
    fn from(v: Vec<T>) -> Self {
        OwnedSlice { data: v.into_boxed_slice() }
    }
}

impl<T> Into<Vec<T>> for OwnedSlice<T> {
    fn into(self) -> Vec<T> {
        self.data.into_vec()
    }
}

impl<T: fmt::Debug> fmt::Debug for OwnedSlice<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.data.fmt(fmt)
    }
}

impl<T> FromIterator<T> for OwnedSlice<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> OwnedSlice<T> {
        OwnedSlice::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<T> IntoIterator for OwnedSlice<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a OwnedSlice<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut OwnedSlice<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T> MoveMap for OwnedSlice<T> {
    type Item = T;
    fn move_map<F>(mut self, mut f: F) -> OwnedSlice<T> where F: FnMut(T) -> T {
        for p in &mut self {
            unsafe {
                // FIXME(#5016) this shouldn't need to zero to be safe.
                ptr::write(p, f(ptr::read_and_drop(p)));
            }
        }
        self
    }
}
