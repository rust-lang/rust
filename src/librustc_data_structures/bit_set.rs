// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use indexed_vec::{Idx, IndexVec};
use rustc_serialize;
use smallvec::SmallVec;
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::slice;

pub type Word = u64;
pub const WORD_BYTES: usize = mem::size_of::<Word>();
pub const WORD_BITS: usize = WORD_BYTES * 8;

/// A fixed-size bitset type with a dense representation. It does not support
/// resizing after creation; use `GrowableBitSet` for that.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
#[derive(Clone, Eq, PartialEq)]
pub struct BitSet<T: Idx> {
    words: Vec<Word>,
    marker: PhantomData<T>,
}

impl<T: Idx> BitSet<T> {
    #[inline]
    pub fn new_empty(domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        BitSet {
            words: vec![0; num_words],
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn new_filled(domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        let mut result = BitSet {
            words: vec![!0; num_words],
            marker: PhantomData,
        };
        result.clear_above(domain_size);
        result
    }

    #[inline]
    pub fn clear(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
    }

    /// Sets all elements up to and including `size`.
    pub fn set_up_to(&mut self, elem: usize) {
        for word in &mut self.words {
            *word = !0;
        }
        self.clear_above(elem);
    }

    /// Clear all elements above `elem`.
    fn clear_above(&mut self, elem: usize) {
        let first_clear_block = elem / WORD_BITS;

        if first_clear_block < self.words.len() {
            // Within `first_clear_block`, the `elem % WORD_BITS` LSBs should
            // remain.
            let mask = (1 << (elem % WORD_BITS)) - 1;
            self.words[first_clear_block] &= mask;

            // All the blocks above `first_clear_block` are fully cleared.
            for word in &mut self.words[first_clear_block + 1..] {
                *word = 0;
            }
        }
    }

    /// Efficiently overwrite `self` with `other`. Panics if `self` and `other`
    /// don't have the same length.
    pub fn overwrite(&mut self, other: &BitSet<T>) {
        self.words.clone_from_slice(&other.words);
    }

    /// Count the number of set bits in the set.
    pub fn count(&self) -> usize {
        self.words.iter().map(|e| e.count_ones() as usize).sum()
    }

    /// True if `self` contains `elem`.
    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        let (word_index, mask) = word_index_and_mask(elem);
        (self.words[word_index] & mask) != 0
    }

    /// True if `self` is a (non-strict) superset of `other`.
    ///
    /// The two sets must have the same domain_size.
    #[inline]
    pub fn superset(&self, other: &BitSet<T>) -> bool {
        assert_eq!(self.words.len(), other.words.len());
        self.words.iter().zip(&other.words).all(|(a, b)| (a & b) == *b)
    }

    /// Is the set empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|a| *a == 0)
    }

    /// Insert `elem`. Returns true if the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        let (word_index, mask) = word_index_and_mask(elem);
        let word_ref = &mut self.words[word_index];
        let word = *word_ref;
        let new_word = word | mask;
        *word_ref = new_word;
        new_word != word
    }

    /// Sets all bits to true.
    pub fn insert_all(&mut self) {
        for word in &mut self.words {
            *word = !0;
        }
    }

    /// Returns true if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        let (word_index, mask) = word_index_and_mask(elem);
        let word_ref = &mut self.words[word_index];
        let word = *word_ref;
        let new_word = word & !mask;
        *word_ref = new_word;
        new_word != word
    }

    /// Set `self = self | other` and return true if `self` changed
    /// (i.e., if new bits were added).
    pub fn union(&mut self, other: &impl UnionIntoBitSet<T>) -> bool {
        other.union_into(self)
    }

    /// Set `self = self - other` and return true if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn subtract(&mut self, other: &impl SubtractFromBitSet<T>) -> bool {
        other.subtract_from(self)
    }

    /// Set `self = self & other` and return true if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn intersect(&mut self, other: &BitSet<T>) -> bool {
        bitwise(&mut self.words, &other.words, |a, b| { a & b })
    }

    /// Get a slice of the underlying words.
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter<'a>(&'a self) -> BitIter<'a, T> {
        BitIter {
            cur: None,
            iter: self.words.iter().enumerate(),
            marker: PhantomData,
        }
    }

    /// Duplicates the set as a hybrid set.
    pub fn to_hybrid(&self) -> HybridBitSet<T> {
        // This domain_size may be slightly larger than the one specified
        // upon creation, due to rounding up to a whole word. That's ok.
        let domain_size = self.words.len() * WORD_BITS;

        // Note: we currently don't bother trying to make a Sparse set.
        HybridBitSet::Dense(self.to_owned(), domain_size)
    }

    pub fn to_string(&self, bits: usize) -> String {
        let mut result = String::new();
        let mut sep = '[';

        // Note: this is a little endian printout of bytes.

        // i tracks how many bits we have printed so far.
        let mut i = 0;
        for word in &self.words {
            let mut word = *word;
            for _ in 0..WORD_BYTES { // for each byte in `word`:
                let remain = bits - i;
                // If less than a byte remains, then mask just that many bits.
                let mask = if remain <= 8 { (1 << remain) - 1 } else { 0xFF };
                assert!(mask <= 0xFF);
                let byte = word & mask;

                result.push_str(&format!("{}{:02x}", sep, byte));

                if remain <= 8 { break; }
                word >>= 8;
                i += 8;
                sep = '-';
            }
            sep = '|';
        }
        result.push(']');

        result
    }
}

/// This is implemented by all the bitsets so that BitSet::union() can be
/// passed any type of bitset.
pub trait UnionIntoBitSet<T: Idx> {
    // Performs `other = other | self`.
    fn union_into(&self, other: &mut BitSet<T>) -> bool;
}

/// This is implemented by all the bitsets so that BitSet::subtract() can be
/// passed any type of bitset.
pub trait SubtractFromBitSet<T: Idx> {
    // Performs `other = other - self`.
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool;
}

impl<T: Idx> UnionIntoBitSet<T> for BitSet<T> {
    fn union_into(&self, other: &mut BitSet<T>) -> bool {
        bitwise(&mut other.words, &self.words, |a, b| { a | b })
    }
}

impl<T: Idx> SubtractFromBitSet<T> for BitSet<T> {
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool {
        bitwise(&mut other.words, &self.words, |a, b| { a & !b })
    }
}

impl<T: Idx> fmt::Debug for BitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        w.debug_list()
         .entries(self.iter())
         .finish()
    }
}

impl<T: Idx> rustc_serialize::Encodable for BitSet<T> {
    fn encode<E: rustc_serialize::Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
        self.words.encode(encoder)
    }
}

impl<T: Idx> rustc_serialize::Decodable for BitSet<T> {
    fn decode<D: rustc_serialize::Decoder>(d: &mut D) -> Result<BitSet<T>, D::Error> {
        let words: Vec<Word> = rustc_serialize::Decodable::decode(d)?;
        Ok(BitSet {
            words,
            marker: PhantomData,
        })
    }
}

pub struct BitIter<'a, T: Idx> {
    cur: Option<(Word, usize)>,
    iter: iter::Enumerate<slice::Iter<'a, Word>>,
    marker: PhantomData<T>
}

impl<'a, T: Idx> Iterator for BitIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        loop {
            if let Some((ref mut word, offset)) = self.cur {
                let bit_pos = word.trailing_zeros() as usize;
                if bit_pos != WORD_BITS {
                    let bit = 1 << bit_pos;
                    *word ^= bit;
                    return Some(T::new(bit_pos + offset))
                }
            }

            let (i, word) = self.iter.next()?;
            self.cur = Some((*word, WORD_BITS * i));
        }
    }
}

pub trait BitSetOperator {
    /// Combine one bitset into another.
    fn join<T: Idx>(&self, inout_set: &mut BitSet<T>, in_set: &BitSet<T>) -> bool;
}

#[inline]
fn bitwise<Op>(out_vec: &mut [Word], in_vec: &[Word], op: Op) -> bool
    where Op: Fn(Word, Word) -> Word
{
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for (out_elem, in_elem) in out_vec.iter_mut().zip(in_vec.iter()) {
        let old_val = *out_elem;
        let new_val = op(old_val, *in_elem);
        *out_elem = new_val;
        changed |= old_val != new_val;
    }
    changed
}

const SPARSE_MAX: usize = 8;

/// A fixed-size bitset type with a sparse representation and a maximum of
/// `SPARSE_MAX` elements. The elements are stored as a sorted `SmallVec` with
/// no duplicates; although `SmallVec` can spill its elements to the heap, that
/// never happens within this type because of the `SPARSE_MAX` limit.
///
/// This type is used by `HybridBitSet`; do not use directly.
#[derive(Clone, Debug)]
pub struct SparseBitSet<T: Idx>(SmallVec<[T; SPARSE_MAX]>);

impl<T: Idx> SparseBitSet<T> {
    fn new_empty() -> Self {
        SparseBitSet(SmallVec::new())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    fn contains(&self, elem: T) -> bool {
        self.0.contains(&elem)
    }

    fn insert(&mut self, elem: T) -> bool {
        assert!(self.len() < SPARSE_MAX);
        if let Some(i) = self.0.iter().position(|&e| e >= elem) {
            if self.0[i] == elem {
                // `elem` is already in the set.
                false
            } else {
                // `elem` is smaller than one or more existing elements.
                self.0.insert(i, elem);
                true
            }
        } else {
            // `elem` is larger than all existing elements.
            self.0.push(elem);
            true
        }
    }

    fn remove(&mut self, elem: T) -> bool {
        if let Some(i) = self.0.iter().position(|&e| e == elem) {
            self.0.remove(i);
            true
        } else {
            false
        }
    }

    fn to_dense(&self, domain_size: usize) -> BitSet<T> {
        let mut dense = BitSet::new_empty(domain_size);
        for elem in self.0.iter() {
            dense.insert(*elem);
        }
        dense
    }

    fn iter(&self) -> slice::Iter<T> {
        self.0.iter()
    }
}

impl<T: Idx> UnionIntoBitSet<T> for SparseBitSet<T> {
    fn union_into(&self, other: &mut BitSet<T>) -> bool {
        let mut changed = false;
        for elem in self.iter() {
            changed |= other.insert(*elem);
        }
        changed
    }
}

impl<T: Idx> SubtractFromBitSet<T> for SparseBitSet<T> {
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool {
        let mut changed = false;
        for elem in self.iter() {
            changed |= other.remove(*elem);
        }
        changed
    }
}

/// A fixed-size bitset type with a hybrid representation: sparse when there
/// are up to a `SPARSE_MAX` elements in the set, but dense when there are more
/// than `SPARSE_MAX`.
///
/// This type is especially efficient for sets that typically have a small
/// number of elements, but a large `domain_size`, and are cleared frequently.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
#[derive(Clone, Debug)]
pub enum HybridBitSet<T: Idx> {
    Sparse(SparseBitSet<T>, usize),
    Dense(BitSet<T>, usize),
}

impl<T: Idx> HybridBitSet<T> {
    // FIXME: This function is used in conjunction with `mem::replace()` in
    // several pieces of awful code below. I can't work out how else to appease
    // the borrow checker.
    fn dummy() -> Self {
        // The cheapest HybridBitSet to construct, which is only used to get
        // around the borrow checker.
        HybridBitSet::Sparse(SparseBitSet::new_empty(), 0)
    }

    pub fn new_empty(domain_size: usize) -> Self {
        HybridBitSet::Sparse(SparseBitSet::new_empty(), domain_size)
    }

    pub fn domain_size(&self) -> usize {
        match *self {
            HybridBitSet::Sparse(_, size) => size,
            HybridBitSet::Dense(_, size) => size,
        }
    }

    pub fn clear(&mut self) {
        let domain_size = self.domain_size();
        *self = HybridBitSet::new_empty(domain_size);
    }

    pub fn contains(&self, elem: T) -> bool {
        match self {
            HybridBitSet::Sparse(sparse, _) => sparse.contains(elem),
            HybridBitSet::Dense(dense, _) => dense.contains(elem),
        }
    }

    pub fn superset(&self, other: &HybridBitSet<T>) -> bool {
        match (self, other) {
            (HybridBitSet::Dense(self_dense, _), HybridBitSet::Dense(other_dense, _)) => {
                self_dense.superset(other_dense)
            }
            _ => other.iter().all(|elem| self.contains(elem)),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            HybridBitSet::Sparse(sparse, _) => sparse.is_empty(),
            HybridBitSet::Dense(dense, _) => dense.is_empty(),
        }
    }

    pub fn insert(&mut self, elem: T) -> bool {
        match self {
            HybridBitSet::Sparse(sparse, _) if sparse.len() < SPARSE_MAX => {
                // The set is sparse and has space for `elem`.
                sparse.insert(elem)
            }
            HybridBitSet::Sparse(sparse, _) if sparse.contains(elem) => {
                // The set is sparse and does not have space for `elem`, but
                // that doesn't matter because `elem` is already present.
                false
            }
            HybridBitSet::Sparse(_, _) => {
                // The set is sparse and full. Convert to a dense set.
                match mem::replace(self, HybridBitSet::dummy()) {
                    HybridBitSet::Sparse(sparse, domain_size) => {
                        let mut dense = sparse.to_dense(domain_size);
                        let changed = dense.insert(elem);
                        assert!(changed);
                        *self = HybridBitSet::Dense(dense, domain_size);
                        changed
                    }
                    _ => unreachable!()
                }
            }

            HybridBitSet::Dense(dense, _) => dense.insert(elem),
        }
    }

    pub fn insert_all(&mut self) {
        let domain_size = self.domain_size();
        match self {
            HybridBitSet::Sparse(_, _) => {
                let dense = BitSet::new_filled(domain_size);
                *self = HybridBitSet::Dense(dense, domain_size);
            }
            HybridBitSet::Dense(dense, _) => dense.insert_all(),
        }
    }

    pub fn remove(&mut self, elem: T) -> bool {
        // Note: we currently don't bother going from Dense back to Sparse.
        match self {
            HybridBitSet::Sparse(sparse, _) => sparse.remove(elem),
            HybridBitSet::Dense(dense, _) => dense.remove(elem),
        }
    }

    pub fn union(&mut self, other: &HybridBitSet<T>) -> bool {
        match self {
            HybridBitSet::Sparse(_, _) => {
                match other {
                    HybridBitSet::Sparse(other_sparse, _) => {
                        // Both sets are sparse. Add the elements in
                        // `other_sparse` to `self_hybrid` one at a time. This
                        // may or may not cause `self_hybrid` to be densified.
                        let mut self_hybrid = mem::replace(self, HybridBitSet::dummy());
                        let mut changed = false;
                        for elem in other_sparse.iter() {
                            changed |= self_hybrid.insert(*elem);
                        }
                        *self = self_hybrid;
                        changed
                    }
                    HybridBitSet::Dense(other_dense, _) => {
                        // `self` is sparse and `other` is dense. Densify
                        // `self` and then do the bitwise union.
                        match mem::replace(self, HybridBitSet::dummy()) {
                            HybridBitSet::Sparse(self_sparse, self_domain_size) => {
                                let mut new_dense = self_sparse.to_dense(self_domain_size);
                                let changed = new_dense.union(other_dense);
                                *self = HybridBitSet::Dense(new_dense, self_domain_size);
                                changed
                            }
                            _ => unreachable!()
                        }
                    }
                }
            }

            HybridBitSet::Dense(self_dense, _) => self_dense.union(other),
        }
    }

    /// Converts to a dense set, consuming itself in the process.
    pub fn to_dense(self) -> BitSet<T> {
        match self {
            HybridBitSet::Sparse(sparse, domain_size) => sparse.to_dense(domain_size),
            HybridBitSet::Dense(dense, _) => dense,
        }
    }

    pub fn iter(&self) -> HybridIter<T> {
        match self {
            HybridBitSet::Sparse(sparse, _) => HybridIter::Sparse(sparse.iter()),
            HybridBitSet::Dense(dense, _) => HybridIter::Dense(dense.iter()),
        }
    }
}

impl<T: Idx> UnionIntoBitSet<T> for HybridBitSet<T> {
    fn union_into(&self, other: &mut BitSet<T>) -> bool {
        match self {
            HybridBitSet::Sparse(sparse, _) => sparse.union_into(other),
            HybridBitSet::Dense(dense, _) => dense.union_into(other),
        }
    }
}

impl<T: Idx> SubtractFromBitSet<T> for HybridBitSet<T> {
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool {
        match self {
            HybridBitSet::Sparse(sparse, _) => sparse.subtract_from(other),
            HybridBitSet::Dense(dense, _) => dense.subtract_from(other),
        }
    }
}

pub enum HybridIter<'a, T: Idx> {
    Sparse(slice::Iter<'a, T>),
    Dense(BitIter<'a, T>),
}

impl<'a, T: Idx> Iterator for HybridIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            HybridIter::Sparse(sparse) => sparse.next().map(|e| *e),
            HybridIter::Dense(dense) => dense.next(),
        }
    }
}

/// A resizable bitset type with a dense representation.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
#[derive(Clone, Debug, PartialEq)]
pub struct GrowableBitSet<T: Idx> {
    bit_set: BitSet<T>,
}

impl<T: Idx> GrowableBitSet<T> {
    pub fn grow(&mut self, domain_size: T) {
        let num_words = num_words(domain_size);
        if self.bit_set.words.len() <= num_words {
            self.bit_set.words.resize(num_words + 1, 0)
        }
    }

    pub fn new_empty() -> GrowableBitSet<T> {
        GrowableBitSet { bit_set: BitSet::new_empty(0) }
    }

    pub fn with_capacity(bits: usize) -> GrowableBitSet<T> {
        GrowableBitSet { bit_set: BitSet::new_empty(bits) }
    }

    /// Returns true if the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        self.grow(elem);
        self.bit_set.insert(elem)
    }

    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        let (word_index, mask) = word_index_and_mask(elem);
        if let Some(word) = self.bit_set.words.get(word_index) {
            (word & mask) != 0
        } else {
            false
        }
    }
}

/// A fixed-size 2D bit matrix type with a dense representation.
///
/// `R` and `C` are index types used to identify rows and columns respectively;
/// typically newtyped `usize` wrappers, but they can also just be `usize`.
///
#[derive(Clone, Debug)]
pub struct BitMatrix<R: Idx, C: Idx> {
    columns: usize,
    words: Vec<Word>,
    marker: PhantomData<(R, C)>,
}

impl<R: Idx, C: Idx> BitMatrix<R, C> {
    /// Create a new `rows x columns` matrix, initially empty.
    pub fn new(rows: usize, columns: usize) -> BitMatrix<R, C> {
        // For every element, we need one bit for every other
        // element. Round up to an even number of words.
        let words_per_row = num_words(columns);
        BitMatrix {
            columns,
            words: vec![0; rows * words_per_row],
            marker: PhantomData,
        }
    }

    /// The range of bits for a given row.
    fn range(&self, row: R) -> (usize, usize) {
        let row = row.index();
        let words_per_row = num_words(self.columns);
        let start = row * words_per_row;
        (start, start + words_per_row)
    }

    /// Sets the cell at `(row, column)` to true. Put another way, insert
    /// `column` to the bitset for `row`.
    ///
    /// Returns true if this changed the matrix, and false otherwise.
    pub fn insert(&mut self, row: R, column: R) -> bool {
        let (start, _) = self.range(row);
        let (word_index, mask) = word_index_and_mask(column);
        let words = &mut self.words[..];
        let word = words[start + word_index];
        let new_word = word | mask;
        words[start + word_index] = new_word;
        word != new_word
    }

    /// Do the bits from `row` contain `column`? Put another way, is
    /// the matrix cell at `(row, column)` true?  Put yet another way,
    /// if the matrix represents (transitive) reachability, can
    /// `row` reach `column`?
    pub fn contains(&self, row: R, column: R) -> bool {
        let (start, _) = self.range(row);
        let (word_index, mask) = word_index_and_mask(column);
        (self.words[start + word_index] & mask) != 0
    }

    /// Returns those indices that are true in rows `a` and `b`.  This
    /// is an O(n) operation where `n` is the number of elements
    /// (somewhat independent from the actual size of the
    /// intersection, in particular).
    pub fn intersect_rows(&self, a: R, b: R) -> Vec<C> {
        let (a_start, a_end) = self.range(a);
        let (b_start, b_end) = self.range(b);
        let mut result = Vec::with_capacity(self.columns);
        for (base, (i, j)) in (a_start..a_end).zip(b_start..b_end).enumerate() {
            let mut v = self.words[i] & self.words[j];
            for bit in 0..WORD_BITS {
                if v == 0 {
                    break;
                }
                if v & 0x1 != 0 {
                    result.push(C::new(base * WORD_BITS + bit));
                }
                v >>= 1;
            }
        }
        result
    }

    /// Add the bits from row `read` to the bits from row `write`,
    /// return true if anything changed.
    ///
    /// This is used when computing transitive reachability because if
    /// you have an edge `write -> read`, because in that case
    /// `write` can reach everything that `read` can (and
    /// potentially more).
    pub fn union_rows(&mut self, read: R, write: R) -> bool {
        let (read_start, read_end) = self.range(read);
        let (write_start, write_end) = self.range(write);
        let words = &mut self.words[..];
        let mut changed = false;
        for (read_index, write_index) in (read_start..read_end).zip(write_start..write_end) {
            let word = words[write_index];
            let new_word = word | words[read_index];
            words[write_index] = new_word;
            changed |= word != new_word;
        }
        changed
    }

    /// Iterates through all the columns set to true in a given row of
    /// the matrix.
    pub fn iter<'a>(&'a self, row: R) -> BitIter<'a, C> {
        let (start, end) = self.range(row);
        BitIter {
            cur: None,
            iter: self.words[start..end].iter().enumerate(),
            marker: PhantomData,
        }
    }
}

/// A fixed-column-size, variable-row-size 2D bit matrix with a moderately
/// sparse representation.
///
/// Initially, every row has no explicit representation. If any bit within a
/// row is set, the entire row is instantiated as `Some(<HybridBitSet>)`.
/// Furthermore, any previously uninstantiated rows prior to it will be
/// instantiated as `None`. Those prior rows may themselves become fully
/// instantiated later on if any of their bits are set.
///
/// `R` and `C` are index types used to identify rows and columns respectively;
/// typically newtyped `usize` wrappers, but they can also just be `usize`.
#[derive(Clone, Debug)]
pub struct SparseBitMatrix<R, C>
where
    R: Idx,
    C: Idx,
{
    num_columns: usize,
    rows: IndexVec<R, Option<HybridBitSet<C>>>,
}

impl<R: Idx, C: Idx> SparseBitMatrix<R, C> {
    /// Create a new empty sparse bit matrix with no rows or columns.
    pub fn new(num_columns: usize) -> Self {
        Self {
            num_columns,
            rows: IndexVec::new(),
        }
    }

    fn ensure_row(&mut self, row: R) -> &mut HybridBitSet<C> {
        // Instantiate any missing rows up to and including row `row` with an
        // empty HybridBitSet.
        self.rows.ensure_contains_elem(row, || None);

        // Then replace row `row` with a full HybridBitSet if necessary.
        let num_columns = self.num_columns;
        self.rows[row].get_or_insert_with(|| HybridBitSet::new_empty(num_columns))
    }

    /// Sets the cell at `(row, column)` to true. Put another way, insert
    /// `column` to the bitset for `row`.
    ///
    /// Returns true if this changed the matrix, and false otherwise.
    pub fn insert(&mut self, row: R, column: C) -> bool {
        self.ensure_row(row).insert(column)
    }

    /// Do the bits from `row` contain `column`? Put another way, is
    /// the matrix cell at `(row, column)` true?  Put yet another way,
    /// if the matrix represents (transitive) reachability, can
    /// `row` reach `column`?
    pub fn contains(&self, row: R, column: C) -> bool {
        self.row(row).map_or(false, |r| r.contains(column))
    }

    /// Add the bits from row `read` to the bits from row `write`,
    /// return true if anything changed.
    ///
    /// This is used when computing transitive reachability because if
    /// you have an edge `write -> read`, because in that case
    /// `write` can reach everything that `read` can (and
    /// potentially more).
    pub fn union_rows(&mut self, read: R, write: R) -> bool {
        if read == write || self.row(read).is_none() {
            return false;
        }

        self.ensure_row(write);
        if let (Some(read_row), Some(write_row)) = self.rows.pick2_mut(read, write) {
            write_row.union(read_row)
        } else {
            unreachable!()
        }
    }

    /// Union a row, `from`, into the `into` row.
    pub fn union_into_row(&mut self, into: R, from: &HybridBitSet<C>) -> bool {
        self.ensure_row(into).union(from)
    }

    /// Insert all bits in the given row.
    pub fn insert_all_into_row(&mut self, row: R) {
        self.ensure_row(row).insert_all();
    }

    pub fn rows(&self) -> impl Iterator<Item = R> {
        self.rows.indices()
    }

    /// Iterates through all the columns set to true in a given row of
    /// the matrix.
    pub fn iter<'a>(&'a self, row: R) -> impl Iterator<Item = C> + 'a {
        self.row(row).into_iter().flat_map(|r| r.iter())
    }

    pub fn row(&self, row: R) -> Option<&HybridBitSet<C>> {
        if let Some(Some(row)) = self.rows.get(row) {
            Some(row)
        } else {
            None
        }
    }
}

#[inline]
fn num_words<T: Idx>(elements: T) -> usize {
    (elements.index() + WORD_BITS - 1) / WORD_BITS
}

#[inline]
fn word_index_and_mask<T: Idx>(index: T) -> (usize, Word) {
    let index = index.index();
    let word_index = index / WORD_BITS;
    let mask = 1 << (index % WORD_BITS);
    (word_index, mask)
}

#[test]
fn test_clear_above() {
    use std::cmp;

    for i in 0..256 {
        let mut idx_buf: BitSet<usize> = BitSet::new_filled(128);
        idx_buf.clear_above(i);

        let elems: Vec<usize> = idx_buf.iter().collect();
        let expected: Vec<usize> = (0..cmp::min(i, 128)).collect();
        assert_eq!(elems, expected);
    }
}

#[test]
fn test_set_up_to() {
    for i in 0..128 {
        for mut idx_buf in
            vec![BitSet::new_empty(128), BitSet::new_filled(128)].into_iter()
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
        let idx_buf = BitSet::new_filled(i);
        let elems: Vec<usize> = idx_buf.iter().collect();
        let expected: Vec<usize> = (0..i).collect();
        assert_eq!(elems, expected);
    }
}

#[test]
fn bitset_iter_works() {
    let mut bitset: BitSet<usize> = BitSet::new_empty(100);
    bitset.insert(1);
    bitset.insert(10);
    bitset.insert(19);
    bitset.insert(62);
    bitset.insert(63);
    bitset.insert(64);
    bitset.insert(65);
    bitset.insert(66);
    bitset.insert(99);
    assert_eq!(
        bitset.iter().collect::<Vec<_>>(),
        [1, 10, 19, 62, 63, 64, 65, 66, 99]
    );
}

#[test]
fn bitset_iter_works_2() {
    let mut bitset: BitSet<usize> = BitSet::new_empty(319);
    bitset.insert(0);
    bitset.insert(127);
    bitset.insert(191);
    bitset.insert(255);
    bitset.insert(319);
    assert_eq!(bitset.iter().collect::<Vec<_>>(), [0, 127, 191, 255, 319]);
}

#[test]
fn union_two_sets() {
    let mut set1: BitSet<usize> = BitSet::new_empty(65);
    let mut set2: BitSet<usize> = BitSet::new_empty(65);
    assert!(set1.insert(3));
    assert!(!set1.insert(3));
    assert!(set2.insert(5));
    assert!(set2.insert(64));
    assert!(set1.union(&set2));
    assert!(!set1.union(&set2));
    assert!(set1.contains(3));
    assert!(!set1.contains(4));
    assert!(set1.contains(5));
    assert!(!set1.contains(63));
    assert!(set1.contains(64));
}

#[test]
fn hybrid_bitset() {
    let mut sparse038: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    assert!(sparse038.is_empty());
    assert!(sparse038.insert(0));
    assert!(sparse038.insert(1));
    assert!(sparse038.insert(8));
    assert!(sparse038.insert(3));
    assert!(!sparse038.insert(3));
    assert!(sparse038.remove(1));
    assert!(!sparse038.is_empty());
    assert_eq!(sparse038.iter().collect::<Vec<_>>(), [0, 3, 8]);

    for i in 0..256 {
        if i == 0 || i == 3 || i == 8 {
            assert!(sparse038.contains(i));
        } else {
            assert!(!sparse038.contains(i));
        }
    }

    let mut sparse01358 = sparse038.clone();
    assert!(sparse01358.insert(1));
    assert!(sparse01358.insert(5));
    assert_eq!(sparse01358.iter().collect::<Vec<_>>(), [0, 1, 3, 5, 8]);

    let mut dense10 = HybridBitSet::new_empty(256);
    for i in 0..10 {
        assert!(dense10.insert(i));
    }
    assert!(!dense10.is_empty());
    assert_eq!(dense10.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut dense256 = HybridBitSet::new_empty(256);
    assert!(dense256.is_empty());
    dense256.insert_all();
    assert!(!dense256.is_empty());
    for i in 0..256 {
        assert!(dense256.contains(i));
    }

    assert!(sparse038.superset(&sparse038));    // sparse + sparse (self)
    assert!(sparse01358.superset(&sparse038));  // sparse + sparse
    assert!(dense10.superset(&sparse038));      // dense + sparse
    assert!(dense10.superset(&dense10));        // dense + dense (self)
    assert!(dense256.superset(&dense10));       // dense + dense

    let mut hybrid = sparse038;
    assert!(!sparse01358.union(&hybrid));       // no change
    assert!(hybrid.union(&sparse01358));
    assert!(hybrid.superset(&sparse01358) && sparse01358.superset(&hybrid));
    assert!(!dense10.union(&sparse01358));
    assert!(!dense256.union(&dense10));
    let mut dense = dense10;
    assert!(dense.union(&dense256));
    assert!(dense.superset(&dense256) && dense256.superset(&dense));
    assert!(hybrid.union(&dense256));
    assert!(hybrid.superset(&dense256) && dense256.superset(&hybrid));

    assert_eq!(dense256.iter().count(), 256);
    let mut dense0 = dense256;
    for i in 0..256 {
        assert!(dense0.remove(i));
    }
    assert!(!dense0.remove(0));
    assert!(dense0.is_empty());
}

#[test]
fn grow() {
    let mut set: GrowableBitSet<usize> = GrowableBitSet::with_capacity(65);
    for index in 0..65 {
        assert!(set.insert(index));
        assert!(!set.insert(index));
    }
    set.grow(128);

    // Check if the bits set before growing are still set
    for index in 0..65 {
        assert!(set.contains(index));
    }

    // Check if the new bits are all un-set
    for index in 65..128 {
        assert!(!set.contains(index));
    }

    // Check that we can set all new bits without running out of bounds
    for index in 65..128 {
        assert!(set.insert(index));
        assert!(!set.insert(index));
    }
}

#[test]
fn matrix_intersection() {
    let mut matrix: BitMatrix<usize, usize> = BitMatrix::new(200, 200);

    // (*) Elements reachable from both 2 and 65.

    matrix.insert(2, 3);
    matrix.insert(2, 6);
    matrix.insert(2, 10); // (*)
    matrix.insert(2, 64); // (*)
    matrix.insert(2, 65);
    matrix.insert(2, 130);
    matrix.insert(2, 160); // (*)

    matrix.insert(64, 133);

    matrix.insert(65, 2);
    matrix.insert(65, 8);
    matrix.insert(65, 10); // (*)
    matrix.insert(65, 64); // (*)
    matrix.insert(65, 68);
    matrix.insert(65, 133);
    matrix.insert(65, 160); // (*)

    let intersection = matrix.intersect_rows(2, 64);
    assert!(intersection.is_empty());

    let intersection = matrix.intersect_rows(2, 65);
    assert_eq!(intersection, &[10, 64, 160]);
}

#[test]
fn matrix_iter() {
    let mut matrix: BitMatrix<usize, usize> = BitMatrix::new(64, 100);
    matrix.insert(3, 22);
    matrix.insert(3, 75);
    matrix.insert(2, 99);
    matrix.insert(4, 0);
    matrix.union_rows(3, 5);

    let expected = [99];
    let mut iter = expected.iter();
    for i in matrix.iter(2) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    for i in matrix.iter(3) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [0];
    let mut iter = expected.iter();
    for i in matrix.iter(4) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    for i in matrix.iter(5) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());
}

#[test]
fn sparse_matrix_iter() {
    let mut matrix: SparseBitMatrix<usize, usize> = SparseBitMatrix::new(100);
    matrix.insert(3, 22);
    matrix.insert(3, 75);
    matrix.insert(2, 99);
    matrix.insert(4, 0);
    matrix.union_rows(3, 5);

    let expected = [99];
    let mut iter = expected.iter();
    for i in matrix.iter(2) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    for i in matrix.iter(3) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [0];
    let mut iter = expected.iter();
    for i in matrix.iter(4) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    for i in matrix.iter(5) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());
}
