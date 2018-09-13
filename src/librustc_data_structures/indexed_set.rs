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
use std::fmt;
use std::mem;
use std::slice;
use bitvec::{bitwise, BitArray, BitIter, Intersect, Subtract, Union, Word, WORD_BITS};
use indexed_vec::Idx;
use rustc_serialize;

/// This is implemented by all the index sets so that IdxSet::union() can be
/// passed any type of index set.
pub trait UnionIntoIdxSet<T: Idx> {
    // Performs `other = other | self`.
    fn union_into(&self, other: &mut IdxSet<T>) -> bool;
}

/// This is implemented by all the index sets so that IdxSet::subtract() can be
/// passed any type of index set.
pub trait SubtractFromIdxSet<T: Idx> {
    // Performs `other = other - self`.
    fn subtract_from(&self, other: &mut IdxSet<T>) -> bool;
}

/// Represents a set of some element type E, where each E is identified by some
/// unique index type `T`.
///
/// In other words, `T` is the type used to index into the bitvector
/// this type uses to represent the set of object it holds.
///
/// The representation is dense, using one bit per possible element.
#[derive(Clone, Eq, PartialEq)]
pub struct IdxSet<T: Idx>(BitArray<T>);

impl<T: Idx> rustc_serialize::Encodable for IdxSet<T> {
    fn encode<E: rustc_serialize::Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
        self.0.encode(encoder)
    }
}

impl<T: Idx> rustc_serialize::Decodable for IdxSet<T> {
    fn decode<D: rustc_serialize::Decoder>(d: &mut D) -> Result<IdxSet<T>, D::Error> {
        Ok(IdxSet(rustc_serialize::Decodable::decode(d)?))
    }
}

impl<T: Idx> fmt::Debug for IdxSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        w.debug_list()
         .entries(self.iter())
         .finish()
    }
}

impl<T: Idx> IdxSet<T> {
    /// Creates set holding no elements.
    pub fn new_empty(domain_size: usize) -> Self {
        IdxSet(BitArray::new_empty(domain_size))
    }

    /// Creates set holding every element whose index falls in range 0..domain_size.
    pub fn new_filled(domain_size: usize) -> Self {
        IdxSet(BitArray::new_filled(domain_size))
    }

    /// Duplicates as a hybrid set.
    pub fn to_hybrid(&self) -> HybridIdxSet<T> {
        // This domain_size may be slightly larger than the one specified
        // upon creation, due to rounding up to a whole word. That's ok.
        let domain_size = self.words().len() * WORD_BITS;

        // Note: we currently don't bother trying to make a Sparse set.
        HybridIdxSet::Dense(self.to_owned(), domain_size)
    }

    /// Removes all elements
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Sets all elements up to `domain_size`
    pub fn set_up_to(&mut self, domain_size: usize) {
        self.0.set_up_to(domain_size);
    }

    /// Removes `elem` from the set `self`; returns true iff this changed `self`.
    pub fn remove(&mut self, elem: &T) -> bool {
        self.0.remove(*elem)
    }

    /// Adds `elem` to the set `self`; returns true iff this changed `self`.
    pub fn add(&mut self, elem: &T) -> bool {
        self.0.insert(*elem)
    }

    /// Returns true iff set `self` contains `elem`.
    pub fn contains(&self, elem: &T) -> bool {
        self.0.contains(*elem)
    }

    pub fn words(&self) -> &[Word] {
        self.0.words()
    }

    pub fn words_mut(&mut self) -> &mut [Word] {
        self.0.words_mut()
    }

    /// Efficiently overwrite `self` with `other`. Panics if `self` and `other`
    /// don't have the same length.
    pub fn overwrite(&mut self, other: &IdxSet<T>) {
        self.words_mut().clone_from_slice(other.words());
    }

    /// Set `self = self | other` and return true if `self` changed
    /// (i.e., if new bits were added).
    pub fn union(&mut self, other: &impl UnionIntoIdxSet<T>) -> bool {
        other.union_into(self)
    }

    /// Set `self = self - other` and return true if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn subtract(&mut self, other: &impl SubtractFromIdxSet<T>) -> bool {
        other.subtract_from(self)
    }

    /// Set `self = self & other` and return true if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn intersect(&mut self, other: &IdxSet<T>) -> bool {
        bitwise(self.words_mut(), other.words(), &Intersect)
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            iter: self.0.iter()
        }
    }
}

impl<T: Idx> UnionIntoIdxSet<T> for IdxSet<T> {
    fn union_into(&self, other: &mut IdxSet<T>) -> bool {
        bitwise(other.words_mut(), self.words(), &Union)
    }
}

impl<T: Idx> SubtractFromIdxSet<T> for IdxSet<T> {
    fn subtract_from(&self, other: &mut IdxSet<T>) -> bool {
        bitwise(other.words_mut(), self.words(), &Subtract)
    }
}

pub struct Iter<'a, T: Idx> {
    iter: BitIter<'a, T>
}

impl<'a, T: Idx> Iterator for Iter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.iter.next()
    }
}

const SPARSE_MAX: usize = 8;

/// A sparse index set with a maximum of SPARSE_MAX elements. Used by
/// HybridIdxSet; do not use directly.
///
/// The elements are stored as an unsorted vector with no duplicates.
#[derive(Clone, Debug)]
pub struct SparseIdxSet<T: Idx>(ArrayVec<[T; SPARSE_MAX]>);

impl<T: Idx> SparseIdxSet<T> {
    fn new() -> Self {
        SparseIdxSet(ArrayVec::new())
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

    fn to_dense(&self, domain_size: usize) -> IdxSet<T> {
        let mut dense = IdxSet::new_empty(domain_size);
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

impl<T: Idx> UnionIntoIdxSet<T> for SparseIdxSet<T> {
    fn union_into(&self, other: &mut IdxSet<T>) -> bool {
        let mut changed = false;
        for elem in self.iter() {
            changed |= other.add(&elem);
        }
        changed
    }
}

impl<T: Idx> SubtractFromIdxSet<T> for SparseIdxSet<T> {
    fn subtract_from(&self, other: &mut IdxSet<T>) -> bool {
        let mut changed = false;
        for elem in self.iter() {
            changed |= other.remove(&elem);
        }
        changed
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

/// Like IdxSet, but with a hybrid representation: sparse when there are few
/// elements in the set, but dense when there are many. It's especially
/// efficient for sets that typically have a small number of elements, but a
/// large `domain_size`, and are cleared frequently.
#[derive(Clone, Debug)]
pub enum HybridIdxSet<T: Idx> {
    Sparse(SparseIdxSet<T>, usize),
    Dense(IdxSet<T>, usize),
}

impl<T: Idx> HybridIdxSet<T> {
    pub fn new_empty(domain_size: usize) -> Self {
        HybridIdxSet::Sparse(SparseIdxSet::new(), domain_size)
    }

    pub fn clear(&mut self) {
        let domain_size = match *self {
            HybridIdxSet::Sparse(_, size) => size,
            HybridIdxSet::Dense(_, size) => size,
        };
        *self = HybridIdxSet::new_empty(domain_size);
    }

    /// Returns true iff set `self` contains `elem`.
    pub fn contains(&self, elem: &T) -> bool {
        match self {
            HybridIdxSet::Sparse(sparse, _) => sparse.contains(elem),
            HybridIdxSet::Dense(dense, _) => dense.contains(elem),
        }
    }

    /// Adds `elem` to the set `self`.
    pub fn add(&mut self, elem: &T) -> bool {
        match self {
            HybridIdxSet::Sparse(sparse, _) if sparse.len() < SPARSE_MAX => {
                // The set is sparse and has space for `elem`.
                sparse.add(elem)
            }
            HybridIdxSet::Sparse(sparse, _) if sparse.contains(elem) => {
                // The set is sparse and does not have space for `elem`, but
                // that doesn't matter because `elem` is already present.
                false
            }
            HybridIdxSet::Sparse(_, _) => {
                // The set is sparse and full. Convert to a dense set.
                //
                // FIXME: This code is awful, but I can't work out how else to
                //        appease the borrow checker.
                let dummy = HybridIdxSet::Sparse(SparseIdxSet::new(), 0);
                match mem::replace(self, dummy) {
                    HybridIdxSet::Sparse(sparse, domain_size) => {
                        let mut dense = sparse.to_dense(domain_size);
                        let changed = dense.add(elem);
                        assert!(changed);
                        mem::replace(self, HybridIdxSet::Dense(dense, domain_size));
                        changed
                    }
                    _ => panic!("impossible"),
                }
            }

            HybridIdxSet::Dense(dense, _) => dense.add(elem),
        }
    }

    /// Removes `elem` from the set `self`.
    pub fn remove(&mut self, elem: &T) -> bool {
        // Note: we currently don't bother going from Dense back to Sparse.
        match self {
            HybridIdxSet::Sparse(sparse, _) => sparse.remove(elem),
            HybridIdxSet::Dense(dense, _) => dense.remove(elem),
        }
    }

    /// Converts to a dense set, consuming itself in the process.
    pub fn to_dense(self) -> IdxSet<T> {
        match self {
            HybridIdxSet::Sparse(sparse, domain_size) => sparse.to_dense(domain_size),
            HybridIdxSet::Dense(dense, _) => dense,
        }
    }

    /// Iteration order is unspecified.
    pub fn iter(&self) -> HybridIter<T> {
        match self {
            HybridIdxSet::Sparse(sparse, _) => HybridIter::Sparse(sparse.iter()),
            HybridIdxSet::Dense(dense, _) => HybridIter::Dense(dense.iter()),
        }
    }
}

impl<T: Idx> UnionIntoIdxSet<T> for HybridIdxSet<T> {
    fn union_into(&self, other: &mut IdxSet<T>) -> bool {
        match self {
            HybridIdxSet::Sparse(sparse, _) => sparse.union_into(other),
            HybridIdxSet::Dense(dense, _) => dense.union_into(other),
        }
    }
}

impl<T: Idx> SubtractFromIdxSet<T> for HybridIdxSet<T> {
    fn subtract_from(&self, other: &mut IdxSet<T>) -> bool {
        match self {
            HybridIdxSet::Sparse(sparse, _) => sparse.subtract_from(other),
            HybridIdxSet::Dense(dense, _) => dense.subtract_from(other),
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
