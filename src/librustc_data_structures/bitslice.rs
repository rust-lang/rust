// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: merge with `bitvec`

use std::mem;

pub type Word = usize;

/// `BitSlice` provides helper methods for treating a `[Word]`
/// as a bitvector.
pub trait BitSlice {
    fn clear_bit(&mut self, idx: usize) -> bool;
    fn set_bit(&mut self, idx: usize) -> bool;
    fn get_bit(&self, idx: usize) -> bool;
}

impl BitSlice for [Word] {
    /// Clears bit at `idx` to 0; returns true iff this changed `self.`
    fn clear_bit(&mut self, idx: usize) -> bool {
        let words = self;
        debug!("clear_bit: words={} idx={}",
               bits_to_string(words, words.len() * mem::size_of::<Word>()), bit_str(idx));
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
               bits_to_string(words, words.len() * mem::size_of::<Word>()), bit_str(idx));
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
    /// An index of the word holding the bit in original `[Word]` of query.
    word: usize,
    /// Index of the particular bit within the word holding the bit.
    bit_in_word: usize,
    /// Word with single 1-bit set corresponding to where the bit is located.
    bit_mask: Word,
}

#[inline]
fn bit_lookup(bit: usize) -> BitLookup {
    let word_bits = mem::size_of::<Word>() * 8;
    let word = bit / word_bits;
    let bit_in_word = bit % word_bits;
    let bit_mask = 1 << bit_in_word;
    BitLookup { word: word, bit_in_word: bit_in_word, bit_mask: bit_mask }
}


fn bit_str(bit: Word) -> String {
    let byte = bit >> 3;
    let lobits = 1 << (bit & 0b111);
    format!("[{}:{}-{:02x}]", bit, byte, lobits)
}

pub fn bits_to_string(words: &[Word], bits: usize) -> String {
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

#[inline]
pub fn bitwise<Op:BitwiseOperator>(out_vec: &mut [usize],
                                   in_vec: &[usize],
                                   op: &Op) -> bool {
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for (out_elt, in_elt) in out_vec.iter_mut().zip(in_vec) {
        let old_val = *out_elt;
        let new_val = op.join(old_val, *in_elt);
        *out_elt = new_val;
        changed |= old_val != new_val;
    }
    changed
}

pub trait BitwiseOperator {
    /// Applies some bit-operation pointwise to each of the bits in the two inputs.
    fn join(&self, pred1: usize, pred2: usize) -> usize;
}

pub struct Union;
impl BitwiseOperator for Union {
    fn join(&self, a: usize, b: usize) -> usize { a | b }
}
pub struct Subtract;
impl BitwiseOperator for Subtract {
    fn join(&self, a: usize, b: usize) -> usize { a & !b }
}
