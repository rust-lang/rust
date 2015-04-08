// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter;

/// A very simple BitVector type.
pub struct BitVector {
    data: Vec<u64>
}

impl BitVector {
    pub fn new(num_bits: usize) -> BitVector {
        let num_words = (num_bits + 63) / 64;
        BitVector { data: iter::repeat(0).take(num_words).collect() }
    }

    fn word_mask(&self, bit: usize) -> (usize, u64) {
        let word = bit / 64;
        let mask = 1 << (bit % 64);
        (word, mask)
    }

    pub fn contains(&self, bit: usize) -> bool {
        let (word, mask) = self.word_mask(bit);
        (self.data[word] & mask) != 0
    }

    pub fn insert(&mut self, bit: usize) -> bool {
        let (word, mask) = self.word_mask(bit);
        let data = &mut self.data[word];
        let value = *data;
        *data = value | mask;
        (value | mask) != value
    }
}
