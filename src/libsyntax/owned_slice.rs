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
use std::slice;
use std::vec;

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

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `OwnedSlice::new` instead")]
    pub fn empty() -> OwnedSlice<T> {
        OwnedSlice  { data: Box::new([]) }
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `OwnedSlice::from` instead")]
    pub fn from_vec(v: Vec<T>) -> OwnedSlice<T> {
        OwnedSlice { data: v.into_boxed_slice() }
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `OwnedSlice::into` instead")]
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_vec()
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `&owned_slice[..]` instead")]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        &*self.data
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `OwnedSlice::into_iter` instead")]
    pub fn move_iter(self) -> vec::IntoIter<T> {
        self.data.into_vec().into_iter()
    }

    #[unstable(feature = "rustc_private", issue = "0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use `iter().map(f).collect()` instead")]
    pub fn map<U, F: FnMut(&T) -> U>(&self, f: F) -> OwnedSlice<U> {
        self.iter().map(f).collect()
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
