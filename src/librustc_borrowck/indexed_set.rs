// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::marker::PhantomData;
use std::mem;
use bitslice::{BitSlice, Word};

pub trait Indexed {
    type Idx: Idx;
}

pub trait Idx {
    fn idx(&self) -> usize;
}

pub struct OwnIdxSet<T: Idx> {
    _pd: PhantomData<fn(&[T], usize) -> &T>,
    bits: Vec<Word>,
}

// pnkfelix wants to have this be `IdxSet<T>([Word]) and then pass
// around `&mut IdxSet<T>` or `&IdxSet<T>`.
//
// Mmapping a `&OwnIdxSet<T>` to `&IdxSet<T>` (at least today)
// requires a transmute relying on representation guarantees that may
// not hold in the future.

pub struct IdxSet<T: Idx> {
    _pd: PhantomData<fn(&[T], usize) -> &T>,
    bits: [Word],
}

impl<T: Idx> fmt::Debug for OwnIdxSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result { self.bits.fmt(w) }
}

impl<T: Idx> fmt::Debug for IdxSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result { self.bits.fmt(w) }
}

impl<T: Idx> OwnIdxSet<T> {
    fn new(init: Word, universe_size: usize) -> Self {
        let bits_per_word = mem::size_of::<Word>();
        let num_words = (universe_size + (bits_per_word - 1)) / bits_per_word;
        OwnIdxSet {
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

    /// Removes `elem` from the set `self`; returns true iff this changed `self`.
    pub fn clear(&mut self, elem: &T) -> bool {
        self.bits.clear_bit(elem.idx())
    }

    /// Adds `elem` to the set `self`; returns true iff this changed `self`.
    pub fn add(&mut self, elem: &T) -> bool {
        self.bits.set_bit(elem.idx())
    }

    /// Returns true iff set `self` contains `elem`.
    pub fn contains(&self, elem: &T) -> bool {
        self.bits.get_bit(elem.idx())
    }

    pub fn bits(&self) -> &[Word] {
        &self.bits[..]
    }
}
