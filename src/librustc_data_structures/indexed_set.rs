// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::{Borrow, BorrowMut, ToOwned};
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut, Range};
use std::slice;
use bitslice::{BitSlice, Word};
use bitslice::{bitwise, Union, Subtract, Intersect};
use indexed_vec::Idx;
use rustc_serialize;

/// Represents a set (or packed family of sets), of some element type
/// E, where each E is identified by some unique index type `T`.
///
/// In other words, `T` is the type used to index into the bitvector
/// this type uses to represent the set of object it holds.
#[derive(Eq, PartialEq)]
pub struct IdxSetBuf<T: Idx> {
    _pd: PhantomData<fn(&T)>,
    bits: Vec<Word>,
}

impl<T: Idx> Clone for IdxSetBuf<T> {
    fn clone(&self) -> Self {
        IdxSetBuf { _pd: PhantomData, bits: self.bits.clone() }
    }
}

impl<T: Idx> rustc_serialize::Encodable for IdxSetBuf<T> {
    fn encode<E: rustc_serialize::Encoder>(&self,
                                     encoder: &mut E)
                                     -> Result<(), E::Error> {
        self.bits.encode(encoder)
    }
}

impl<T: Idx> rustc_serialize::Decodable for IdxSetBuf<T> {
    fn decode<D: rustc_serialize::Decoder>(d: &mut D) -> Result<IdxSetBuf<T>, D::Error> {
        let words: Vec<Word> = rustc_serialize::Decodable::decode(d)?;

        Ok(IdxSetBuf {
            _pd: PhantomData,
            bits: words,
        })
    }
}


// pnkfelix wants to have this be `IdxSet<T>([Word]) and then pass
// around `&mut IdxSet<T>` or `&IdxSet<T>`.
//
// WARNING: Mapping a `&IdxSetBuf<T>` to `&IdxSet<T>` (at least today)
// requires a transmute relying on representation guarantees that may
// not hold in the future.

/// Represents a set (or packed family of sets), of some element type
/// E, where each E is identified by some unique index type `T`.
///
/// In other words, `T` is the type used to index into the bitslice
/// this type uses to represent the set of object it holds.
pub struct IdxSet<T: Idx> {
    _pd: PhantomData<fn(&T)>,
    bits: [Word],
}

impl<T: Idx> Borrow<IdxSet<T>> for IdxSetBuf<T> {
    fn borrow(&self) -> &IdxSet<T> {
        &*self
    }
}

impl<T: Idx> BorrowMut<IdxSet<T>> for IdxSetBuf<T> {
    fn borrow_mut(&mut self) -> &mut IdxSet<T> {
        &mut *self
    }
}

impl<T: Idx> ToOwned for IdxSet<T> {
    type Owned = IdxSetBuf<T>;
    fn to_owned(&self) -> Self::Owned {
        IdxSet::to_owned(self)
    }
}

impl<T: Idx> fmt::Debug for IdxSetBuf<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        w.debug_list()
         .entries(self.iter())
         .finish()
    }
}

impl<T: Idx> fmt::Debug for IdxSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        w.debug_list()
         .entries(self.iter())
         .finish()
    }
}

impl<T: Idx> IdxSetBuf<T> {
    fn new(init: Word, universe_size: usize) -> Self {
        let bits_per_word = mem::size_of::<Word>() * 8;
        let num_words = (universe_size + (bits_per_word - 1)) / bits_per_word;
        IdxSetBuf {
            _pd: Default::default(),
            bits: vec![init; num_words],
        }
    }

    /// Creates set holding every element whose index falls in range 0..universe_size.
    pub fn new_filled(universe_size: usize) -> Self {
        Self::new(!0, universe_size)
    }

    /// Creates set holding no elements.
    pub fn new_empty(universe_size: usize) -> Self {
        Self::new(0, universe_size)
    }
}

impl<T: Idx> IdxSet<T> {
    unsafe fn from_slice(s: &[Word]) -> &Self {
        mem::transmute(s) // (see above WARNING)
    }

    unsafe fn from_slice_mut(s: &mut [Word]) -> &mut Self {
        mem::transmute(s) // (see above WARNING)
    }
}

impl<T: Idx> Deref for IdxSetBuf<T> {
    type Target = IdxSet<T>;
    fn deref(&self) -> &IdxSet<T> {
        unsafe { IdxSet::from_slice(&self.bits) }
    }
}

impl<T: Idx> DerefMut for IdxSetBuf<T> {
    fn deref_mut(&mut self) -> &mut IdxSet<T> {
        unsafe { IdxSet::from_slice_mut(&mut self.bits) }
    }
}

impl<T: Idx> IdxSet<T> {
    pub fn to_owned(&self) -> IdxSetBuf<T> {
        IdxSetBuf {
            _pd: Default::default(),
            bits: self.bits.to_owned(),
        }
    }

    /// Removes all elements
    pub fn clear(&mut self) {
        for b in &mut self.bits {
            *b = 0;
        }
    }

    /// Removes `elem` from the set `self`; returns true iff this changed `self`.
    pub fn remove(&mut self, elem: &T) -> bool {
        self.bits.clear_bit(elem.index())
    }

    /// Adds `elem` to the set `self`; returns true iff this changed `self`.
    pub fn add(&mut self, elem: &T) -> bool {
        self.bits.set_bit(elem.index())
    }

    pub fn range(&self, elems: &Range<T>) -> &Self {
        let elems = elems.start.index()..elems.end.index();
        unsafe { Self::from_slice(&self.bits[elems]) }
    }

    pub fn range_mut(&mut self, elems: &Range<T>) -> &mut Self {
        let elems = elems.start.index()..elems.end.index();
        unsafe { Self::from_slice_mut(&mut self.bits[elems]) }
    }

    /// Returns true iff set `self` contains `elem`.
    pub fn contains(&self, elem: &T) -> bool {
        self.bits.get_bit(elem.index())
    }

    pub fn words(&self) -> &[Word] {
        &self.bits
    }

    pub fn words_mut(&mut self) -> &mut [Word] {
        &mut self.bits
    }

    pub fn clone_from(&mut self, other: &IdxSet<T>) {
        self.words_mut().clone_from_slice(other.words());
    }

    pub fn union(&mut self, other: &IdxSet<T>) -> bool {
        bitwise(self.words_mut(), other.words(), &Union)
    }

    pub fn subtract(&mut self, other: &IdxSet<T>) -> bool {
        bitwise(self.words_mut(), other.words(), &Subtract)
    }

    pub fn intersect(&mut self, other: &IdxSet<T>) -> bool {
        bitwise(self.words_mut(), other.words(), &Intersect)
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            cur: None,
            iter: self.words().iter().enumerate(),
            _pd: PhantomData,
        }
    }

    /// Calls `f` on each index value held in this set, up to the
    /// bound `max_bits` on the size of universe of indexes.
    pub fn each_bit<F>(&self, max_bits: usize, f: F) where F: FnMut(T) {
        each_bit(self, max_bits, f)
    }

    /// Removes all elements from this set.
    pub fn reset_to_empty(&mut self) {
        for word in self.words_mut() { *word = 0; }
    }

    pub fn elems(&self, universe_size: usize) -> Elems<T> {
        Elems { i: 0, set: self, universe_size: universe_size }
    }
}

pub struct Elems<'a, T: Idx> { i: usize, set: &'a IdxSet<T>, universe_size: usize }

impl<'a, T: Idx> Iterator for Elems<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.i >= self.universe_size { return None; }
        let mut i = self.i;
        loop {
            if i >= self.universe_size {
                self.i = i; // (mark iteration as complete.)
                return None;
            }
            if self.set.contains(&T::new(i)) {
                self.i = i + 1; // (next element to start at.)
                return Some(T::new(i));
            }
            i = i + 1;
        }
    }
}

fn each_bit<T: Idx, F>(words: &IdxSet<T>, max_bits: usize, mut f: F) where F: FnMut(T) {
    let usize_bits: usize = mem::size_of::<usize>() * 8;

    for (word_index, &word) in words.words().iter().enumerate() {
        if word != 0 {
            let base_index = word_index * usize_bits;
            for offset in 0..usize_bits {
                let bit = 1 << offset;
                if (word & bit) != 0 {
                    // NB: we round up the total number of bits
                    // that we store in any given bit set so that
                    // it is an even multiple of usize::BITS. This
                    // means that there may be some stray bits at
                    // the end that do not correspond to any
                    // actual value; that's why we first check
                    // that we are in range of bits_per_block.
                    let bit_index = base_index + offset as usize;
                    if bit_index >= max_bits {
                        return;
                    } else {
                        f(Idx::new(bit_index));
                    }
                }
            }
        }
    }
}

pub struct Iter<'a, T: Idx> {
    cur: Option<(Word, usize)>,
    iter: iter::Enumerate<slice::Iter<'a, Word>>,
    _pd: PhantomData<fn(&T)>,
}

impl<'a, T: Idx> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let word_bits = mem::size_of::<Word>() * 8;
        loop {
            if let Some((ref mut word, offset)) = self.cur {
                let bit_pos = word.trailing_zeros() as usize;
                if bit_pos != word_bits {
                    let bit = 1 << bit_pos;
                    *word ^= bit;
                    return Some(T::new(bit_pos + offset))
                }
            }

            let (i, word) = self.iter.next()?;
            self.cur = Some((*word, word_bits * i));
        }
    }
}
