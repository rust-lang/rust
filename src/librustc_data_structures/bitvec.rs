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
use std::marker::PhantomData;

type Word = u128;
const WORD_BITS: usize = 128;

/// A very simple BitArray type.
///
/// It does not support resizing after creation; use `BitVector` for that.
#[derive(Clone, Debug, PartialEq)]
pub struct BitArray<C: Idx> {
    data: Vec<Word>,
    marker: PhantomData<C>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BitVector<C: Idx> {
    data: BitArray<C>,
}

impl<C: Idx> BitVector<C> {
    pub fn grow(&mut self, num_bits: C) {
        self.data.grow(num_bits)
    }

    pub fn new() -> BitVector<C> {
        BitVector {
            data: BitArray::new(0),
        }
    }

    pub fn with_capacity(bits: usize) -> BitVector<C> {
        BitVector {
            data: BitArray::new(bits),
        }
    }

    /// Returns true if the bit has changed.
    #[inline]
    pub fn insert(&mut self, bit: C) -> bool {
        self.grow(bit);
        self.data.insert(bit)
    }

    #[inline]
    pub fn contains(&self, bit: C) -> bool {
        let (word, mask) = word_mask(bit);
        if let Some(word) = self.data.data.get(word) {
            (word & mask) != 0
        } else {
            false
        }
    }
}

impl<C: Idx> BitArray<C> {
    // Do not make this method public, instead switch your use case to BitVector.
    #[inline]
    fn grow(&mut self, num_bits: C) {
        let num_words = words(num_bits);
        if self.data.len() <= num_words {
            self.data.resize(num_words + 1, 0)
        }
    }

    #[inline]
    pub fn new(num_bits: usize) -> BitArray<C> {
        let num_words = words(num_bits);
        BitArray {
            data: vec![0; num_words],
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        for p in &mut self.data {
            *p = 0;
        }
    }

    pub fn count(&self) -> usize {
        self.data.iter().map(|e| e.count_ones() as usize).sum()
    }

    /// True if `self` contains the bit `bit`.
    #[inline]
    pub fn contains(&self, bit: C) -> bool {
        let (word, mask) = word_mask(bit);
        (self.data[word] & mask) != 0
    }

    /// True if `self` contains all the bits in `other`.
    ///
    /// The two vectors must have the same length.
    #[inline]
    pub fn contains_all(&self, other: &BitArray<C>) -> bool {
        assert_eq!(self.data.len(), other.data.len());
        self.data.iter().zip(&other.data).all(|(a, b)| (a & b) == *b)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|a| *a == 0)
    }

    /// Returns true if the bit has changed.
    #[inline]
    pub fn insert(&mut self, bit: C) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &mut self.data[word];
        let value = *data;
        let new_value = value | mask;
        *data = new_value;
        new_value != value
    }

    /// Sets all bits to true.
    pub fn insert_all(&mut self) {
        for data in &mut self.data {
            *data = u128::max_value();
        }
    }

    /// Returns true if the bit has changed.
    #[inline]
    pub fn remove(&mut self, bit: C) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &mut self.data[word];
        let value = *data;
        let new_value = value & !mask;
        *data = new_value;
        new_value != value
    }

    #[inline]
    pub fn merge(&mut self, all: &BitArray<C>) -> bool {
        assert!(self.data.len() == all.data.len());
        let mut changed = false;
        for (i, j) in self.data.iter_mut().zip(&all.data) {
            let value = *i;
            *i = value | *j;
            if value != *i {
                changed = true;
            }
        }
        changed
    }

    /// Iterates over indexes of set bits in a sorted order
    #[inline]
    pub fn iter<'a>(&'a self) -> BitIter<'a, C> {
        BitIter {
            iter: self.data.iter(),
            current: 0,
            idx: 0,
            marker: PhantomData,
        }
    }
}

pub struct BitIter<'a, C: Idx> {
    iter: ::std::slice::Iter<'a, Word>,
    current: Word,
    idx: usize,
    marker: PhantomData<C>
}

impl<'a, C: Idx> Iterator for BitIter<'a, C> {
    type Item = C;
    fn next(&mut self) -> Option<C> {
        while self.current == 0 {
            self.current = if let Some(&i) = self.iter.next() {
                if i == 0 {
                    self.idx += WORD_BITS;
                    continue;
                } else {
                    self.idx = words(self.idx) * WORD_BITS;
                    i
                }
            } else {
                return None;
            }
        }
        let offset = self.current.trailing_zeros() as usize;
        self.current >>= offset;
        self.current >>= 1; // shift otherwise overflows for 0b1000_0000_…_0000
        self.idx += offset + 1;

        Some(C::new(self.idx - 1))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

/// A "bit matrix" is basically a matrix of booleans represented as
/// one gigantic bitvector. In other words, it is as if you have
/// `rows` bitvectors, each of length `columns`.
#[derive(Clone, Debug)]
pub struct BitMatrix<R: Idx, C: Idx> {
    columns: usize,
    vector: Vec<Word>,
    phantom: PhantomData<(R, C)>,
}

impl<R: Idx, C: Idx> BitMatrix<R, C> {
    /// Create a new `rows x columns` matrix, initially empty.
    pub fn new(rows: usize, columns: usize) -> BitMatrix<R, C> {
        // For every element, we need one bit for every other
        // element. Round up to an even number of words.
        let words_per_row = words(columns);
        BitMatrix {
            columns,
            vector: vec![0; rows * words_per_row],
            phantom: PhantomData,
        }
    }

    /// The range of bits for a given row.
    fn range(&self, row: R) -> (usize, usize) {
        let row = row.index();
        let words_per_row = words(self.columns);
        let start = row * words_per_row;
        (start, start + words_per_row)
    }

    /// Sets the cell at `(row, column)` to true. Put another way, add
    /// `column` to the bitset for `row`.
    ///
    /// Returns true if this changed the matrix, and false otherwise.
    pub fn add(&mut self, row: R, column: R) -> bool {
        let (start, _) = self.range(row);
        let (word, mask) = word_mask(column);
        let vector = &mut self.vector[..];
        let v1 = vector[start + word];
        let v2 = v1 | mask;
        vector[start + word] = v2;
        v1 != v2
    }

    /// Do the bits from `row` contain `column`? Put another way, is
    /// the matrix cell at `(row, column)` true?  Put yet another way,
    /// if the matrix represents (transitive) reachability, can
    /// `row` reach `column`?
    pub fn contains(&self, row: R, column: R) -> bool {
        let (start, _) = self.range(row);
        let (word, mask) = word_mask(column);
        (self.vector[start + word] & mask) != 0
    }

    /// Returns those indices that are true in rows `a` and `b`.  This
    /// is an O(n) operation where `n` is the number of elements
    /// (somewhat independent from the actual size of the
    /// intersection, in particular).
    pub fn intersection(&self, a: R, b: R) -> Vec<C> {
        let (a_start, a_end) = self.range(a);
        let (b_start, b_end) = self.range(b);
        let mut result = Vec::with_capacity(self.columns);
        for (base, (i, j)) in (a_start..a_end).zip(b_start..b_end).enumerate() {
            let mut v = self.vector[i] & self.vector[j];
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
    pub fn merge(&mut self, read: R, write: R) -> bool {
        let (read_start, read_end) = self.range(read);
        let (write_start, write_end) = self.range(write);
        let vector = &mut self.vector[..];
        let mut changed = false;
        for (read_index, write_index) in (read_start..read_end).zip(write_start..write_end) {
            let v1 = vector[write_index];
            let v2 = v1 | vector[read_index];
            vector[write_index] = v2;
            changed |= v1 != v2;
        }
        changed
    }

    /// Iterates through all the columns set to true in a given row of
    /// the matrix.
    pub fn iter<'a>(&'a self, row: R) -> BitIter<'a, C> {
        let (start, end) = self.range(row);
        BitIter {
            iter: self.vector[start..end].iter(),
            current: 0,
            idx: 0,
            marker: PhantomData,
        }
    }
}

/// A moderately sparse bit matrix: rows are appended lazily, but columns
/// within appended rows are instantiated fully upon creation.
#[derive(Clone, Debug)]
pub struct SparseBitMatrix<R, C>
where
    R: Idx,
    C: Idx,
{
    columns: usize,
    vector: IndexVec<R, BitArray<C>>,
}

impl<R: Idx, C: Idx> SparseBitMatrix<R, C> {
    /// Create a new empty sparse bit matrix with no rows or columns.
    pub fn new(columns: usize) -> Self {
        Self {
            columns,
            vector: IndexVec::new(),
        }
    }

    fn ensure_row(&mut self, row: R) {
        let columns = self.columns;
        self.vector
            .ensure_contains_elem(row, || BitArray::new(columns));
    }

    /// Sets the cell at `(row, column)` to true. Put another way, insert
    /// `column` to the bitset for `row`.
    ///
    /// Returns true if this changed the matrix, and false otherwise.
    pub fn add(&mut self, row: R, column: C) -> bool {
        self.ensure_row(row);
        self.vector[row].insert(column)
    }

    /// Do the bits from `row` contain `column`? Put another way, is
    /// the matrix cell at `(row, column)` true?  Put yet another way,
    /// if the matrix represents (transitive) reachability, can
    /// `row` reach `column`?
    pub fn contains(&self, row: R, column: C) -> bool {
        self.vector.get(row).map_or(false, |r| r.contains(column))
    }

    /// Add the bits from row `read` to the bits from row `write`,
    /// return true if anything changed.
    ///
    /// This is used when computing transitive reachability because if
    /// you have an edge `write -> read`, because in that case
    /// `write` can reach everything that `read` can (and
    /// potentially more).
    pub fn merge(&mut self, read: R, write: R) -> bool {
        if read == write || self.vector.get(read).is_none() {
            return false;
        }

        self.ensure_row(write);
        let (bitvec_read, bitvec_write) = self.vector.pick2_mut(read, write);
        bitvec_write.merge(bitvec_read)
    }

    /// Merge a row, `from`, into the `into` row.
    pub fn merge_into(&mut self, into: R, from: &BitArray<C>) -> bool {
        self.ensure_row(into);
        self.vector[into].merge(from)
    }

    /// Add all bits to the given row.
    pub fn add_all(&mut self, row: R) {
        self.ensure_row(row);
        self.vector[row].insert_all();
    }

    /// Number of elements in the matrix.
    pub fn len(&self) -> usize {
        self.vector.len()
    }

    pub fn rows(&self) -> impl Iterator<Item = R> {
        self.vector.indices()
    }

    /// Iterates through all the columns set to true in a given row of
    /// the matrix.
    pub fn iter<'a>(&'a self, row: R) -> impl Iterator<Item = C> + 'a {
        self.vector.get(row).into_iter().flat_map(|r| r.iter())
    }

    /// Iterates through each row and the accompanying bit set.
    pub fn iter_enumerated<'a>(&'a self) -> impl Iterator<Item = (R, &'a BitArray<C>)> + 'a {
        self.vector.iter_enumerated()
    }

    pub fn row(&self, row: R) -> Option<&BitArray<C>> {
        self.vector.get(row)
    }
}

#[inline]
fn words<C: Idx>(elements: C) -> usize {
    (elements.index() + WORD_BITS - 1) / WORD_BITS
}

#[inline]
fn word_mask<C: Idx>(index: C) -> (usize, Word) {
    let index = index.index();
    let word = index / WORD_BITS;
    let mask = 1 << (index % WORD_BITS);
    (word, mask)
}

#[test]
fn bitvec_iter_works() {
    let mut bitvec: BitArray<usize> = BitArray::new(100);
    bitvec.insert(1);
    bitvec.insert(10);
    bitvec.insert(19);
    bitvec.insert(62);
    bitvec.insert(63);
    bitvec.insert(64);
    bitvec.insert(65);
    bitvec.insert(66);
    bitvec.insert(99);
    assert_eq!(
        bitvec.iter().collect::<Vec<_>>(),
        [1, 10, 19, 62, 63, 64, 65, 66, 99]
    );
}

#[test]
fn bitvec_iter_works_2() {
    let mut bitvec: BitArray<usize> = BitArray::new(319);
    bitvec.insert(0);
    bitvec.insert(127);
    bitvec.insert(191);
    bitvec.insert(255);
    bitvec.insert(319);
    assert_eq!(bitvec.iter().collect::<Vec<_>>(), [0, 127, 191, 255, 319]);
}

#[test]
fn union_two_vecs() {
    let mut vec1: BitArray<usize> = BitArray::new(65);
    let mut vec2: BitArray<usize> = BitArray::new(65);
    assert!(vec1.insert(3));
    assert!(!vec1.insert(3));
    assert!(vec2.insert(5));
    assert!(vec2.insert(64));
    assert!(vec1.merge(&vec2));
    assert!(!vec1.merge(&vec2));
    assert!(vec1.contains(3));
    assert!(!vec1.contains(4));
    assert!(vec1.contains(5));
    assert!(!vec1.contains(63));
    assert!(vec1.contains(64));
}

#[test]
fn grow() {
    let mut vec1: BitVector<usize> = BitVector::with_capacity(65);
    for index in 0..65 {
        assert!(vec1.insert(index));
        assert!(!vec1.insert(index));
    }
    vec1.grow(128);

    // Check if the bits set before growing are still set
    for index in 0..65 {
        assert!(vec1.contains(index));
    }

    // Check if the new bits are all un-set
    for index in 65..128 {
        assert!(!vec1.contains(index));
    }

    // Check that we can set all new bits without running out of bounds
    for index in 65..128 {
        assert!(vec1.insert(index));
        assert!(!vec1.insert(index));
    }
}

#[test]
fn matrix_intersection() {
    let mut vec1: BitMatrix<usize, usize> = BitMatrix::new(200, 200);

    // (*) Elements reachable from both 2 and 65.

    vec1.add(2, 3);
    vec1.add(2, 6);
    vec1.add(2, 10); // (*)
    vec1.add(2, 64); // (*)
    vec1.add(2, 65);
    vec1.add(2, 130);
    vec1.add(2, 160); // (*)

    vec1.add(64, 133);

    vec1.add(65, 2);
    vec1.add(65, 8);
    vec1.add(65, 10); // (*)
    vec1.add(65, 64); // (*)
    vec1.add(65, 68);
    vec1.add(65, 133);
    vec1.add(65, 160); // (*)

    let intersection = vec1.intersection(2, 64);
    assert!(intersection.is_empty());

    let intersection = vec1.intersection(2, 65);
    assert_eq!(intersection, &[10, 64, 160]);
}

#[test]
fn matrix_iter() {
    let mut matrix: BitMatrix<usize, usize> = BitMatrix::new(64, 100);
    matrix.add(3, 22);
    matrix.add(3, 75);
    matrix.add(2, 99);
    matrix.add(4, 0);
    matrix.merge(3, 5);

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
    matrix.add(3, 22);
    matrix.add(3, 75);
    matrix.add(2, 99);
    matrix.add(4, 0);
    matrix.merge(3, 5);

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
