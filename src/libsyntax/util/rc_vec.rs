// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::ops::{Deref, Range};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher, StableHasherResult};
use rustc_data_structures::sync::Lrc;

#[derive(Clone)]
pub struct RcVec<T> {
    data: Lrc<Vec<T>>,
    offset: u32,
    len: u32,
}

impl<T> RcVec<T> {
    pub fn new(mut vec: Vec<T>) -> Self {
        // By default, constructing RcVec from Vec gives it just enough capacity
        // to hold the initial elements. Callers that anticipate needing to
        // extend the vector may prefer RcVec::new_preserving_capacity.
        vec.shrink_to_fit();
        Self::new_preserving_capacity(vec)
    }

    pub fn new_preserving_capacity(vec: Vec<T>) -> Self {
        RcVec {
            offset: 0,
            len: vec.len() as u32,
            data: Lrc::new(vec),
        }
    }

    pub fn sub_slice(&self, range: Range<usize>) -> Self {
        RcVec {
            data: self.data.clone(),
            offset: self.offset + range.start as u32,
            len: (range.end - range.start) as u32,
        }
    }

    /// If this RcVec has exactly one strong reference, returns ownership of the
    /// underlying vector. Otherwise returns self unmodified.
    pub fn try_unwrap(self) -> Result<Vec<T>, Self> {
        match Lrc::try_unwrap(self.data) {
            // If no other RcVec shares ownership of this data.
            Ok(mut vec) => {
                // Drop any elements after our view of the data.
                vec.truncate(self.offset as usize + self.len as usize);
                // Drop any elements before our view of the data. Do this after
                // the `truncate` so that elements past the end of our view do
                // not need to be copied around.
                vec.drain(..self.offset as usize);
                Ok(vec)
            }

            // If the data is shared.
            Err(data) => Err(RcVec { data, ..self }),
        }
    }
}

impl<T> Deref for RcVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.data[self.offset as usize..(self.offset + self.len) as usize]
    }
}

impl<T: fmt::Debug> fmt::Debug for RcVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.deref(), f)
    }
}

impl<CTX, T> HashStable<CTX> for RcVec<T>
where
    T: HashStable<CTX>,
{
    fn hash_stable<W: StableHasherResult>(&self, hcx: &mut CTX, hasher: &mut StableHasher<W>) {
        (**self).hash_stable(hcx, hasher);
    }
}
