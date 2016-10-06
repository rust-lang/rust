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
use std::ops::{Deref, DerefMut, Range};
use bitslice::{BitSlice, Word};
use bitslice::{bitwise, Union, Subtract};
use indexed_vec::Idx;

/// Represents a set (or packed family of sets), of some element type
/// E, where each E is identified by some unique index type `T`.
///
/// In other words, `T` is the type used to index into the bitvector
/// this type uses to represent the set of object it holds.
pub struct IdxSetBuf<T: Idx> {
    _pd: PhantomData<fn(&T)>,
    bits: Vec<Word>,
}

impl<T: Idx> Clone for IdxSetBuf<T> {
    fn clone(&self) -> Self {
        IdxSetBuf { _pd: PhantomData, bits: self.bits.clone() }
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

impl<T: Idx> fmt::Debug for IdxSetBuf<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result { self.bits.fmt(w) }
}

impl<T: Idx> fmt::Debug for IdxSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result { self.bits.fmt(w) }
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
        unsafe { IdxSet::from_slice(&self.bits[..]) }
    }
}

impl<T: Idx> DerefMut for IdxSetBuf<T> {
    fn deref_mut(&mut self) -> &mut IdxSet<T> {
        unsafe { IdxSet::from_slice_mut(&mut self.bits[..]) }
    }
}

impl<T: Idx> IdxSet<T> {
    pub fn to_owned(&self) -> IdxSetBuf<T> {
        IdxSetBuf {
            _pd: Default::default(),
            bits: self.bits.to_owned(),
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
        &self.bits[..]
    }

    pub fn words_mut(&mut self) -> &mut [Word] {
        &mut self.bits[..]
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
}
