// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::u32;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TreeIndex {
    index: u32,
}

impl TreeIndex {
    pub fn new(value: usize) -> TreeIndex {
        assert!(value < (u32::MAX as usize));
        TreeIndex { index: value as u32 }
    }

    pub fn get(self) -> usize {
        self.index as usize
    }
}
