// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use array_vec::ArrayVec;
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
///
/// The representation is dense, using one bit per possible element.
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

const BITS_PER_WORD: usize = mem::size_of::<Word>() * 8;

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
        let num_words = (universe_size + (BITS_PER_WORD - 1)) / BITS_PER_WORD;
        IdxSetBuf {
            _pd: Default::default(),
            bits: vec![init; num_words],
        }
    }

    /// Creates set holding every element whose index falls in range 0..universe_size.
    pub fn new_filled(universe_size: usize) -> Self {
        let mut result = Self::new(!0, universe_size);
        result.trim_to(universe_size);
        result
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

    /// Duplicates as a hybrid set.
    pub fn to_hybrid(&self) -> HybridIdxSetBuf<T> {
        // This universe_size may be slightly larger than the one specified
        // upon creation, due to rounding up to a whole word. That's ok.
        let universe_size = self.bits.len() * BITS_PER_WORD;

        // Note: we currently don't bother trying to make a Sparse set.
        HybridIdxSetBuf::Dense(self.to_owned(), universe_size)
    }

    /// Removes all elements
    pub fn clear(&mut self) {
        for b in &mut self.bits {
            *b = 0;
        }
    }

    /// Sets all elements up to `universe_size`
    pub fn set_up_to(&mut self, universe_size: usize) {
        for b in &mut self.bits {
            *b = !0;
        }
        self.trim_to(universe_size);
    }

    /// Clear all elements above `universe_size`.
    fn trim_to(&mut self, universe_size: usize) {
        // `trim_block` is the first block where some bits have
        // to be cleared.
        let trim_block = universe_size / BITS_PER_WORD;

        // all the blocks above it have to be completely cleared.
        if trim_block < self.bits.len() {
            for b in &mut self.bits[trim_block+1..] {
                *b = 0;
            }

            // at that block, the `universe_size % BITS_PER_WORD` lsbs
            // should remain.
            let remaining_bits = universe_size % BITS_PER_WORD;
            let mask = (1<<remaining_bits)-1;
            self.bits[trim_block] &= mask;
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

    /// Efficiently overwrite `self` with `other`. Panics if `self` and `other`
    /// don't have the same length.
    pub fn overwrite(&mut self, other: &IdxSet<T>) {
        self.words_mut().clone_from_slice(other.words());
    }

    /// Set `self = self | other` and return true if `self` changed
    /// (i.e., if new bits were added).
    pub fn union(&mut self, other: &IdxSet<T>) -> bool {
        bitwise(self.words_mut(), other.words(), &Union)
    }

    /// Like `union()`, but takes a `SparseIdxSetBuf` argument.
    fn union_sparse(&mut self, other: &SparseIdxSetBuf<T>) -> bool {
        let mut changed = false;
        for elem in other.iter() {
            changed |= self.add(&elem);
        }
        changed
    }

    /// Like `union()`, but takes a `HybridIdxSetBuf` argument.
    pub fn union_hybrid(&mut self, other: &HybridIdxSetBuf<T>) -> bool {
        match other {
            HybridIdxSetBuf::Sparse(sparse, _) => self.union_sparse(sparse),
            HybridIdxSetBuf::Dense(dense, _) => self.union(dense),
        }
    }

    /// Set `self = self - other` and return true if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn subtract(&mut self, other: &IdxSet<T>) -> bool {
        bitwise(self.words_mut(), other.words(), &Subtract)
    }

    /// Like `subtract()`, but takes a `SparseIdxSetBuf` argument.
    fn subtract_sparse(&mut self, other: &SparseIdxSetBuf<T>) -> bool {
        let mut changed = false;
        for elem in other.iter() {
            changed |= self.remove(&elem);
        }
        changed
    }

    /// Like `subtract()`, but takes a `HybridIdxSetBuf` argument.
    pub fn subtract_hybrid(&mut self, other: &HybridIdxSetBuf<T>) -> bool {
        match other {
            HybridIdxSetBuf::Sparse(sparse, _) => self.subtract_sparse(sparse),
            HybridIdxSetBuf::Dense(dense, _) => self.subtract(dense),
        }
    }

    /// Set `self = self & other` and return true if `self` changed.
    /// (i.e., if any bits were removed).
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
}

pub struct Iter<'a, T: Idx> {
    cur: Option<(Word, usize)>,
    iter: iter::Enumerate<slice::Iter<'a, Word>>,
    _pd: PhantomData<fn(&T)>,
}

impl<'a, T: Idx> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            if let Some((ref mut word, offset)) = self.cur {
                let bit_pos = word.trailing_zeros() as usize;
                if bit_pos != BITS_PER_WORD {
                    let bit = 1 << bit_pos;
                    *word ^= bit;
                    return Some(T::new(bit_pos + offset))
                }
            }

            let (i, word) = self.iter.next()?;
            self.cur = Some((*word, BITS_PER_WORD * i));
        }
    }
}

const SPARSE_MAX: usize = 8;

/// A sparse index set with a maximum of SPARSE_MAX elements. Used by
/// HybridIdxSetBuf; do not use directly.
///
/// The elements are stored as an unsorted vector with no duplicates.
#[derive(Clone, Debug)]
pub struct SparseIdxSetBuf<T: Idx>(ArrayVec<[T; SPARSE_MAX]>);

impl<T: Idx> SparseIdxSetBuf<T> {
    fn new() -> Self {
        SparseIdxSetBuf(ArrayVec::new())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn contains(&self, elem: &T) -> bool {
        self.0.contains(elem)
    }

    fn add(&mut self, elem: &T) -> bool {
        // Ensure there are no duplicates.
        if self.0.contains(elem) {
            false
        } else {
            self.0.push(*elem);
            true
        }
    }

    fn remove(&mut self, elem: &T) -> bool {
        if let Some(i) = self.0.iter().position(|e| e == elem) {
            // Swap the found element to the end, then pop it.
            let len = self.0.len();
            self.0.swap(i, len - 1);
            self.0.pop();
            true
        } else {
            false
        }
    }

    fn to_dense(&self, universe_size: usize) -> IdxSetBuf<T> {
        let mut dense = IdxSetBuf::new_empty(universe_size);
        for elem in self.0.iter() {
            dense.add(elem);
        }
        dense
    }

    fn iter(&self) -> SparseIter<T> {
        SparseIter {
            iter: self.0.iter(),
        }
    }
}

pub struct SparseIter<'a, T: Idx> {
    iter: slice::Iter<'a, T>,
}

impl<'a, T: Idx> Iterator for SparseIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|e| *e)
    }
}

/// Like IdxSetBuf, but with a hybrid representation: sparse when there are few
/// elements in the set, but dense when there are many. It's especially
/// efficient for sets that typically have a small number of elements, but a
/// large `universe_size`, and are cleared frequently.
#[derive(Clone, Debug)]
pub enum HybridIdxSetBuf<T: Idx> {
    Sparse(SparseIdxSetBuf<T>, usize),
    Dense(IdxSetBuf<T>, usize),
}

impl<T: Idx> HybridIdxSetBuf<T> {
    pub fn new_empty(universe_size: usize) -> Self {
        HybridIdxSetBuf::Sparse(SparseIdxSetBuf::new(), universe_size)
    }

    fn universe_size(&mut self) -> usize {
        match *self {
            HybridIdxSetBuf::Sparse(_, size) => size,
            HybridIdxSetBuf::Dense(_, size) => size,
        }
    }

    pub fn clear(&mut self) {
        let universe_size = self.universe_size();
        *self = HybridIdxSetBuf::new_empty(universe_size);
    }

    /// Returns true iff set `self` contains `elem`.
    pub fn contains(&self, elem: &T) -> bool {
        match self {
            HybridIdxSetBuf::Sparse(sparse, _) => sparse.contains(elem),
            HybridIdxSetBuf::Dense(dense, _) => dense.contains(elem),
        }
    }

    /// Adds `elem` to the set `self`.
    pub fn add(&mut self, elem: &T) -> bool {
        match self {
            HybridIdxSetBuf::Sparse(sparse, _) if sparse.len() < SPARSE_MAX => {
                // The set is sparse and has space for `elem`.
                sparse.add(elem)
            }
            HybridIdxSetBuf::Sparse(sparse, _) if sparse.contains(elem) => {
                // The set is sparse and does not have space for `elem`, but
                // that doesn't matter because `elem` is already present.
                false
            }
            HybridIdxSetBuf::Sparse(_, _) => {
                // The set is sparse and full. Convert to a dense set.
                //
                // FIXME: This code is awful, but I can't work out how else to
                //        appease the borrow checker.
                let dummy = HybridIdxSetBuf::Sparse(SparseIdxSetBuf::new(), 0);
                match mem::replace(self, dummy) {
                    HybridIdxSetBuf::Sparse(sparse, universe_size) => {
                        let mut dense = sparse.to_dense(universe_size);
                        let changed = dense.add(elem);
                        assert!(changed);
                        mem::replace(self, HybridIdxSetBuf::Dense(dense, universe_size));
                        changed
                    }
                    _ => panic!("impossible"),
                }
            }

            HybridIdxSetBuf::Dense(dense, _) => dense.add(elem),
        }
    }

    /// Removes `elem` from the set `self`.
    pub fn remove(&mut self, elem: &T) -> bool {
        // Note: we currently don't bother going from Dense back to Sparse.
        match self {
            HybridIdxSetBuf::Sparse(sparse, _) => sparse.remove(elem),
            HybridIdxSetBuf::Dense(dense, _) => dense.remove(elem),
        }
    }

    /// Converts to a dense set, consuming itself in the process.
    pub fn to_dense(self) -> IdxSetBuf<T> {
        match self {
            HybridIdxSetBuf::Sparse(sparse, universe_size) => sparse.to_dense(universe_size),
            HybridIdxSetBuf::Dense(dense, _) => dense,
        }
    }

    /// Iteration order is unspecified.
    pub fn iter(&self) -> HybridIter<T> {
        match self {
            HybridIdxSetBuf::Sparse(sparse, _) => HybridIter::Sparse(sparse.iter()),
            HybridIdxSetBuf::Dense(dense, _) => HybridIter::Dense(dense.iter()),
        }
    }
}

pub enum HybridIter<'a, T: Idx> {
    Sparse(SparseIter<'a, T>),
    Dense(Iter<'a, T>),
}

impl<'a, T: Idx> Iterator for HybridIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            HybridIter::Sparse(sparse) => sparse.next(),
            HybridIter::Dense(dense) => dense.next(),
        }
    }
}

#[test]
fn test_trim_to() {
    use std::cmp;

    for i in 0..256 {
        let mut idx_buf: IdxSetBuf<usize> = IdxSetBuf::new_filled(128);
        idx_buf.trim_to(i);

        let elems: Vec<usize> = idx_buf.iter().collect();
        let expected: Vec<usize> = (0..cmp::min(i, 128)).collect();
        assert_eq!(elems, expected);
    }
}

#[test]
fn test_set_up_to() {
    for i in 0..128 {
        for mut idx_buf in
            vec![IdxSetBuf::new_empty(128), IdxSetBuf::new_filled(128)]
            .into_iter()
        {
            idx_buf.set_up_to(i);

            let elems: Vec<usize> = idx_buf.iter().collect();
            let expected: Vec<usize> = (0..i).collect();
            assert_eq!(elems, expected);
        }
    }
}

#[test]
fn test_new_filled() {
    for i in 0..128 {
        let mut idx_buf = IdxSetBuf::new_filled(i);
        let elems: Vec<usize> = idx_buf.iter().collect();
        let expected: Vec<usize> = (0..i).collect();
        assert_eq!(elems, expected);
    }
}
