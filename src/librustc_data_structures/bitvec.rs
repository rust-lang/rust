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
use std::iter;
use std::marker::PhantomData;
use std::slice;

pub type Word = u64;
pub const WORD_BYTES: usize = ::std::mem::size_of::<Word>();
pub const WORD_BITS: usize = WORD_BYTES * 8;

/// A very simple BitArray type.
///
/// It does not support resizing after creation; use `BitVector` for that.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BitArray<C: Idx> {
    data: Vec<Word>,
    marker: PhantomData<C>,
}

impl<C: Idx> BitArray<C> {
    // Do not make this method public, instead switch your use case to BitVector.
    #[inline]
    fn grow(&mut self, num_bits: C) {
        let num_words = num_words(num_bits);
        if self.data.len() <= num_words {
            self.data.resize(num_words + 1, 0)
        }
    }

    #[inline]
    pub fn new(num_bits: usize) -> BitArray<C> {
        BitArray::new_empty(num_bits)
    }

    #[inline]
    pub fn new_empty(num_bits: usize) -> BitArray<C> {
        let num_words = num_words(num_bits);
        BitArray {
            data: vec![0; num_words],
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn new_filled(num_bits: usize) -> BitArray<C> {
        let num_words = num_words(num_bits);
        let mut result = BitArray {
            data: vec![!0; num_words],
            marker: PhantomData,
        };
        result.clear_above(num_bits);
        result
    }

    #[inline]
    pub fn clear(&mut self) {
        for p in &mut self.data {
            *p = 0;
        }
    }

    /// Sets all elements up to `num_bits`.
    pub fn set_up_to(&mut self, num_bits: usize) {
        for p in &mut self.data {
            *p = !0;
        }
        self.clear_above(num_bits);
    }

    /// Clear all elements above `num_bits`.
    fn clear_above(&mut self, num_bits: usize) {
        let first_clear_block = num_bits / WORD_BITS;

        if first_clear_block < self.data.len() {
            // Within `first_clear_block`, the `num_bits % WORD_BITS` LSBs
            // should remain.
            let mask = (1 << (num_bits % WORD_BITS)) - 1;
            self.data[first_clear_block] &= mask;

            // All the blocks above `first_clear_block` are fully cleared.
            for b in &mut self.data[first_clear_block + 1..] {
                *b = 0;
            }
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
            *data = !0;
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

    pub fn words(&self) -> &[Word] {
        &self.data
    }

    pub fn words_mut(&mut self) -> &mut [Word] {
        &mut self.data
    }

    /// Iterates over indexes of set bits in a sorted order
    #[inline]
    pub fn iter<'a>(&'a self) -> BitIter<'a, C> {
        BitIter {
            cur: None,
            iter: self.data.iter().enumerate(),
            marker: PhantomData,
        }
    }
}

impl<T: Idx> rustc_serialize::Encodable for BitArray<T> {
    fn encode<E: rustc_serialize::Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
        self.data.encode(encoder)
    }
}

impl<T: Idx> rustc_serialize::Decodable for BitArray<T> {
    fn decode<D: rustc_serialize::Decoder>(d: &mut D) -> Result<BitArray<T>, D::Error> {
        let words: Vec<Word> = rustc_serialize::Decodable::decode(d)?;
        Ok(BitArray {
            data: words,
            marker: PhantomData,
        })
    }
}

pub struct BitIter<'a, C: Idx> {
    cur: Option<(Word, usize)>,
    iter: iter::Enumerate<slice::Iter<'a, Word>>,
    marker: PhantomData<C>
}

impl<'a, C: Idx> Iterator for BitIter<'a, C> {
    type Item = C;
    fn next(&mut self) -> Option<C> {
        loop {
            if let Some((ref mut word, offset)) = self.cur {
                let bit_pos = word.trailing_zeros() as usize;
                if bit_pos != WORD_BITS {
                    let bit = 1 << bit_pos;
                    *word ^= bit;
                    return Some(C::new(bit_pos + offset))
                }
            }

            let (i, word) = self.iter.next()?;
            self.cur = Some((*word, WORD_BITS * i));
        }
    }
}

pub trait BitwiseOperator {
    /// Applies some bit-operation pointwise to each of the bits in the two inputs.
    fn join(&self, pred1: Word, pred2: Word) -> Word;
}

#[inline]
pub fn bitwise<Op: BitwiseOperator>(out_vec: &mut [Word], in_vec: &[Word], op: &Op) -> bool
{
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = false;
    for (out_elem, in_elem) in out_vec.iter_mut().zip(in_vec.iter()) {
        let old_val = *out_elem;
        let new_val = op.join(old_val, *in_elem);
        *out_elem = new_val;
        changed |= old_val != new_val;
    }
    changed
}

pub struct Intersect;
impl BitwiseOperator for Intersect {
    #[inline]
    fn join(&self, a: Word, b: Word) -> Word { a & b }
}

pub struct Union;
impl BitwiseOperator for Union {
    #[inline]
    fn join(&self, a: Word, b: Word) -> Word { a | b }
}

pub struct Subtract;
impl BitwiseOperator for Subtract {
    #[inline]
    fn join(&self, a: Word, b: Word) -> Word { a & !b }
}

pub fn bits_to_string(words: &[Word], bits: usize) -> String {
    let mut result = String::new();
    let mut sep = '[';

    // Note: this is a little endian printout of bytes.

    // i tracks how many bits we have printed so far.
    let mut i = 0;
    for &word in words.iter() {
        let mut v = word;
        for _ in 0..WORD_BYTES { // for each byte in `v`:
            let remain = bits - i;
            // If less than a byte remains, then mask just that many bits.
            let mask = if remain <= 8 { (1 << remain) - 1 } else { 0xFF };
            assert!(mask <= 0xFF);
            let byte = v & mask;

            result.push_str(&format!("{}{:02x}", sep, byte));

            if remain <= 8 { break; }
            v >>= 8;
            i += 8;
            sep = '-';
        }
        sep = '|';
    }
    result.push(']');

    result
}

/// A resizable BitVector type.
#[derive(Clone, Debug, PartialEq)]
pub struct BitVector<C: Idx> {
    data: BitArray<C>,
}

impl<C: Idx> BitVector<C> {
    pub fn grow(&mut self, num_bits: C) {
        self.data.grow(num_bits)
    }

    pub fn new() -> BitVector<C> {
        BitVector { data: BitArray::new(0) }
    }

    pub fn with_capacity(bits: usize) -> BitVector<C> {
        BitVector { data: BitArray::new(bits) }
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
        let words_per_row = num_words(columns);
        BitMatrix {
            columns,
            vector: vec![0; rows * words_per_row],
            phantom: PhantomData,
        }
    }

    /// The range of bits for a given row.
    fn range(&self, row: R) -> (usize, usize) {
        let row = row.index();
        let words_per_row = num_words(self.columns);
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
            cur: None,
            iter: self.vector[start..end].iter().enumerate(),
            marker: PhantomData,
        }
    }
}

/// A moderately sparse bit matrix, in which rows are instantiated lazily.
///
/// Initially, every row has no explicit representation. If any bit within a
/// row is set, the entire row is instantiated as
/// `Some(<full-column-width-BitArray>)`. Furthermore, any previously
/// uninstantiated rows prior to it will be instantiated as `None`. Those prior
/// rows may themselves become fully instantiated later on if any of their bits
/// are set.
#[derive(Clone, Debug)]
pub struct SparseBitMatrix<R, C>
where
    R: Idx,
    C: Idx,
{
    num_columns: usize,
    rows: IndexVec<R, Option<BitArray<C>>>,
}

impl<R: Idx, C: Idx> SparseBitMatrix<R, C> {
    /// Create a new empty sparse bit matrix with no rows or columns.
    pub fn new(num_columns: usize) -> Self {
        Self {
            num_columns,
            rows: IndexVec::new(),
        }
    }

    fn ensure_row(&mut self, row: R) -> &mut BitArray<C> {
        // Instantiate any missing rows up to and including row `row` with an
        // empty BitArray.
        self.rows.ensure_contains_elem(row, || None);

        // Then replace row `row` with a full BitArray if necessary.
        let num_columns = self.num_columns;
        self.rows[row].get_or_insert_with(|| BitArray::new(num_columns))
    }

    /// Sets the cell at `(row, column)` to true. Put another way, insert
    /// `column` to the bitset for `row`.
    ///
    /// Returns true if this changed the matrix, and false otherwise.
    pub fn add(&mut self, row: R, column: C) -> bool {
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
    pub fn merge(&mut self, read: R, write: R) -> bool {
        if read == write || self.row(read).is_none() {
            return false;
        }

        self.ensure_row(write);
        if let (Some(bitvec_read), Some(bitvec_write)) = self.rows.pick2_mut(read, write) {
            bitvec_write.merge(bitvec_read)
        } else {
            unreachable!()
        }
    }

    /// Merge a row, `from`, into the `into` row.
    pub fn merge_into(&mut self, into: R, from: &BitArray<C>) -> bool {
        self.ensure_row(into).merge(from)
    }

    /// Add all bits to the given row.
    pub fn add_all(&mut self, row: R) {
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

    pub fn row(&self, row: R) -> Option<&BitArray<C>> {
        if let Some(Some(row)) = self.rows.get(row) {
            Some(row)
        } else {
            None
        }
    }
}

#[inline]
fn num_words<C: Idx>(elements: C) -> usize {
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
fn test_clear_above() {
    use std::cmp;

    for i in 0..256 {
        let mut idx_buf: BitArray<usize> = BitArray::new_filled(128);
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
            vec![BitArray::new_empty(128), BitArray::new_filled(128)]
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
        let idx_buf = BitArray::new_filled(i);
        let elems: Vec<usize> = idx_buf.iter().collect();
        let expected: Vec<usize> = (0..i).collect();
        assert_eq!(elems, expected);
    }
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
