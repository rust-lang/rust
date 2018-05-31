// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::num::NonZeroU32;
use std::u32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeIndex {
    index: NonZeroU32,
}

impl NodeIndex {
    #[inline]
    pub fn new(value: usize) -> NodeIndex {
        assert!(value < (u32::MAX as usize));
        NodeIndex { index: NonZeroU32::new((value as u32) + 1).unwrap() }
    }

    #[inline]
    pub fn get(self) -> usize {
        (self.index.get() - 1) as usize
    }
}
