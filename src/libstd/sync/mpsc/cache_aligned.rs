// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ops::{Deref, DerefMut};

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(align(64))]
pub(super) struct Aligner;

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CacheAligned<T>(pub T, pub Aligner);

impl<T> Deref for CacheAligned<T> {
     type Target = T;
     fn deref(&self) -> &Self::Target {
         &self.0
     }
}

impl<T> DerefMut for CacheAligned<T> {
     fn deref_mut(&mut self) -> &mut Self::Target {
         &mut self.0
     }
}

impl<T> CacheAligned<T> {
    pub(super) fn new(t: T) -> Self {
        CacheAligned(t, Aligner)
    }
}
