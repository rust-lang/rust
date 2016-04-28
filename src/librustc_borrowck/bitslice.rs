// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

/// `BitSlice` provides helper methods for treating a `[usize]`
/// as a bitvector.
pub trait BitSlice {
    fn clear_bit(&mut self, idx: usize) -> bool;
    fn set_bit(&mut self, idx: usize) -> bool;
    fn get_bit(&self, idx: usize) -> bool;
}

impl BitSlice for [usize] {
    /// Clears bit at `idx` to 0; returns true iff this changed `self.`
    fn clear_bit(&mut self, idx: usize) -> bool {
        let words = self;
        debug!("clear_bit: words={} idx={}",
               bits_to_string(words, words.len() * mem::size_of::<usize>()), bit_str(idx));
        let BitLookup { word, bit_in_word, bit_mask } = bit_lookup(idx);
        debug!("word={} bit_in_word={} bit_mask={}", word, bit_in_word, bit_mask);
        let oldv = words[word];
        let newv = oldv & !bit_mask;
        words[word] = newv;
        oldv != newv
    }

    /// Sets bit at `idx` to 1; returns true iff this changed `self.`
    fn set_bit(&mut self, idx: usize) -> bool {
        let words = self;
        debug!("set_bit: words={} idx={}",
               bits_to_string(words, words.len() * mem::size_of::<usize>()), bit_str(idx));
        let BitLookup { word, bit_in_word, bit_mask } = bit_lookup(idx);
        debug!("word={} bit_in_word={} bit_mask={}", word, bit_in_word, bit_mask);
        let oldv = words[word];
        let newv = oldv | bit_mask;
        words[word] = newv;
        oldv != newv
    }

    /// Extracts value of bit at `idx` in `self`.
    fn get_bit(&self, idx: usize) -> bool {
        let words = self;
        let BitLookup { word, bit_mask, .. } = bit_lookup(idx);
        (words[word] & bit_mask) != 0
    }
}

struct BitLookup {
    /// An index of the word holding the bit in original `[usize]` of query.
    word: usize,
    /// Index of the particular bit within the word holding the bit.
    bit_in_word: usize,
    /// Word with single 1-bit set corresponding to where the bit is located.
    bit_mask: usize,
}

#[inline]
fn bit_lookup(bit: usize) -> BitLookup {
    let usize_bits = mem::size_of::<usize>() * 8;
    let word = bit / usize_bits;
    let bit_in_word = bit % usize_bits;
    let bit_mask = 1 << bit_in_word;
    BitLookup { word: word, bit_in_word: bit_in_word, bit_mask: bit_mask }
}


fn bit_str(bit: usize) -> String {
    let byte = bit >> 3;
    let lobits = 1 << (bit & 0b111);
    format!("[{}:{}-{:02x}]", bit, byte, lobits)
}

pub fn bits_to_string(words: &[usize], bits: usize) -> String {
    let mut result = String::new();
    let mut sep = '[';

    // Note: this is a little endian printout of bytes.

    // i tracks how many bits we have printed so far.
    let mut i = 0;
    for &word in words.iter() {
        let mut v = word;
        loop { // for each byte in `v`:
            let remain = bits - i;
            // If less than a byte remains, then mask just that many bits.
            let mask = if remain <= 8 { (1 << remain) - 1 } else { 0xFF };
            assert!(mask <= 0xFF);
            let byte = v & mask;

            result.push(sep);
            result.push_str(&format!("{:02x}", byte));

            if remain <= 8 { break; }
            v >>= 8;
            i += 8;
            sep = '-';
        }
    }
    result.push(']');
    return result
}
