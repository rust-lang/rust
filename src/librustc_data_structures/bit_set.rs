use crate::indexed_vec::{Idx, IndexVec};
use smallvec::SmallVec;
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::slice;
#[cfg(test)]
extern crate test;
#[cfg(test)]
use test::Bencher;

pub type Word = u64;
pub const WORD_BYTES: usize = mem::size_of::<Word>();
pub const WORD_BITS: usize = WORD_BYTES * 8;

/// A fixed-size bitset type with a dense representation. It does not support
/// resizing after creation; use `GrowableBitSet` for that.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
#[derive(Clone, Eq, PartialEq, RustcDecodable, RustcEncodable)]
pub struct BitSet<T: Idx> {
    domain_size: usize,
    words: Vec<Word>,
    marker: PhantomData<T>,
}

impl<T: Idx> BitSet<T> {
    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        BitSet {
            domain_size,
            words: vec![0; num_words],
            marker: PhantomData,
        }
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled(domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        let mut result = BitSet {
            domain_size,
            words: vec![!0; num_words],
            marker: PhantomData,
        };
        result.clear_excess_bits();
        result
    }

    /// Gets the domain size.
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
    }

    /// Clear excess bits in the final word.
    fn clear_excess_bits(&mut self) {
        let num_bits_in_final_word = self.domain_size % WORD_BITS;
        if num_bits_in_final_word > 0 {
            let mask = (1 << num_bits_in_final_word) - 1;
            let final_word_idx = self.words.len() - 1;
            self.words[final_word_idx] &= mask;
        }
    }

    /// Efficiently overwrite `self` with `other`.
    pub fn overwrite(&mut self, other: &BitSet<T>) {
        assert!(self.domain_size == other.domain_size);
        self.words.clone_from_slice(&other.words);
    }

    /// Count the number of set bits in the set.
    pub fn count(&self) -> usize {
        self.words.iter().map(|e| e.count_ones() as usize).sum()
    }

    /// Returns `true` if `self` contains `elem`.
    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        (self.words[word_index] & mask) != 0
    }

    /// Is `self` is a (non-strict) superset of `other`?
    #[inline]
    pub fn superset(&self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        self.words.iter().zip(&other.words).all(|(a, b)| (a & b) == *b)
    }

    /// Is the set empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|a| *a == 0)
    }

    /// Insert `elem`. Returns whether the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
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
        self.clear_excess_bits();
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        let word_ref = &mut self.words[word_index];
        let word = *word_ref;
        let new_word = word & !mask;
        *word_ref = new_word;
        new_word != word
    }

    /// Sets `self = self | other` and returns `true` if `self` changed
    /// (i.e., if new bits were added).
    pub fn union(&mut self, other: &impl UnionIntoBitSet<T>) -> bool {
        other.union_into(self)
    }

    /// Sets `self = self - other` and returns `true` if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn subtract(&mut self, other: &impl SubtractFromBitSet<T>) -> bool {
        other.subtract_from(self)
    }

    /// Sets `self = self & other` and return `true` if `self` changed.
    /// (i.e., if any bits were removed).
    pub fn intersect(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut self.words, &other.words, |a, b| { a & b })
    }

    /// Gets a slice of the underlying words.
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        BitIter {
            cur: None,
            iter: self.words.iter().enumerate(),
            marker: PhantomData,
        }
    }

    /// Duplicates the set as a hybrid set.
    pub fn to_hybrid(&self) -> HybridBitSet<T> {
        // Note: we currently don't bother trying to make a Sparse set.
        HybridBitSet::Dense(self.to_owned())
    }

    /// Set `self = self | other`. In contrast to `union` returns `true` if the set contains at
    /// least one bit that is not in `other` (i.e. `other` is not a superset of `self`).
    ///
    /// This is an optimization for union of a hybrid bitset.
    fn reverse_union_sparse(&mut self, sparse: &SparseBitSet<T>) -> bool {
        assert!(sparse.domain_size == self.domain_size);
        self.clear_excess_bits();

        let mut not_already = false;
        // Index of the current word not yet merged.
        let mut current_index = 0;
        // Mask of bits that came from the sparse set in the current word.
        let mut new_bit_mask = 0;
        for (word_index, mask) in sparse.iter().map(|x| word_index_and_mask(*x)) {
            // Next bit is in a word not inspected yet.
            if word_index > current_index {
                self.words[current_index] |= new_bit_mask;
                // Were there any bits in the old word that did not occur in the sparse set?
                not_already |= (self.words[current_index] ^ new_bit_mask) != 0;
                // Check all words we skipped for any set bit.
                not_already |= self.words[current_index+1..word_index].iter().any(|&x| x != 0);
                // Update next word.
                current_index = word_index;
                // Reset bit mask, no bits have been merged yet.
                new_bit_mask = 0;
            }
            // Add bit and mark it as coming from the sparse set.
            // self.words[word_index] |= mask;
            new_bit_mask |= mask;
        }
        self.words[current_index] |= new_bit_mask;
        // Any bits in the last inspected word that were not in the sparse set?
        not_already |= (self.words[current_index] ^ new_bit_mask) != 0;
        // Any bits in the tail? Note `clear_excess_bits` before.
        not_already |= self.words[current_index+1..].iter().any(|&x| x != 0);

        not_already
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
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut other.words, &self.words, |a, b| { a | b })
    }
}

impl<T: Idx> SubtractFromBitSet<T> for BitSet<T> {
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut other.words, &self.words, |a, b| { a & !b })
    }
}

impl<T: Idx> fmt::Debug for BitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list()
         .entries(self.iter())
         .finish()
    }
}

impl<T: Idx> ToString for BitSet<T> {
    fn to_string(&self) -> String {
        let mut result = String::new();
        let mut sep = '[';

        // Note: this is a little endian printout of bytes.

        // i tracks how many bits we have printed so far.
        let mut i = 0;
        for word in &self.words {
            let mut word = *word;
            for _ in 0..WORD_BYTES { // for each byte in `word`:
                let remain = self.domain_size - i;
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
pub struct SparseBitSet<T: Idx> {
    domain_size: usize,
    elems: SmallVec<[T; SPARSE_MAX]>,
}

impl<T: Idx> SparseBitSet<T> {
    fn new_empty(domain_size: usize) -> Self {
        SparseBitSet {
            domain_size,
            elems: SmallVec::new()
        }
    }

    fn len(&self) -> usize {
        self.elems.len()
    }

    fn is_empty(&self) -> bool {
        self.elems.len() == 0
    }

    fn contains(&self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        self.elems.contains(&elem)
    }

    fn insert(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        let changed = if let Some(i) = self.elems.iter().position(|&e| e >= elem) {
            if self.elems[i] == elem {
                // `elem` is already in the set.
                false
            } else {
                // `elem` is smaller than one or more existing elements.
                self.elems.insert(i, elem);
                true
            }
        } else {
            // `elem` is larger than all existing elements.
            self.elems.push(elem);
            true
        };
        assert!(self.len() <= SPARSE_MAX);
        changed
    }

    fn remove(&mut self, elem: T) -> bool {
        assert!(elem.index() < self.domain_size);
        if let Some(i) = self.elems.iter().position(|&e| e == elem) {
            self.elems.remove(i);
            true
        } else {
            false
        }
    }

    fn to_dense(&self) -> BitSet<T> {
        let mut dense = BitSet::new_empty(self.domain_size);
        for elem in self.elems.iter() {
            dense.insert(*elem);
        }
        dense
    }

    fn iter(&self) -> slice::Iter<'_, T> {
        self.elems.iter()
    }
}

impl<T: Idx> UnionIntoBitSet<T> for SparseBitSet<T> {
    fn union_into(&self, other: &mut BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        let mut changed = false;
        for elem in self.iter() {
            changed |= other.insert(*elem);
        }
        changed
    }
}

impl<T: Idx> SubtractFromBitSet<T> for SparseBitSet<T> {
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
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
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
#[derive(Clone, Debug)]
pub enum HybridBitSet<T: Idx> {
    Sparse(SparseBitSet<T>),
    Dense(BitSet<T>),
}

impl<T: Idx> HybridBitSet<T> {
    pub fn new_empty(domain_size: usize) -> Self {
        HybridBitSet::Sparse(SparseBitSet::new_empty(domain_size))
    }

    fn domain_size(&self) -> usize {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.domain_size,
            HybridBitSet::Dense(dense) => dense.domain_size,
        }
    }

    pub fn clear(&mut self) {
        let domain_size = self.domain_size();
        *self = HybridBitSet::new_empty(domain_size);
    }

    pub fn contains(&self, elem: T) -> bool {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.contains(elem),
            HybridBitSet::Dense(dense) => dense.contains(elem),
        }
    }

    pub fn superset(&self, other: &HybridBitSet<T>) -> bool {
        match (self, other) {
            (HybridBitSet::Dense(self_dense), HybridBitSet::Dense(other_dense)) => {
                self_dense.superset(other_dense)
            }
            _ => {
                assert!(self.domain_size() == other.domain_size());
                other.iter().all(|elem| self.contains(elem))
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.is_empty(),
            HybridBitSet::Dense(dense) => dense.is_empty(),
        }
    }

    pub fn insert(&mut self, elem: T) -> bool {
        // No need to check `elem` against `self.domain_size` here because all
        // the match cases check it, one way or another.
        match self {
            HybridBitSet::Sparse(sparse) if sparse.len() < SPARSE_MAX => {
                // The set is sparse and has space for `elem`.
                sparse.insert(elem)
            }
            HybridBitSet::Sparse(sparse) if sparse.contains(elem) => {
                // The set is sparse and does not have space for `elem`, but
                // that doesn't matter because `elem` is already present.
                false
            }
            HybridBitSet::Sparse(sparse) => {
                // The set is sparse and full. Convert to a dense set.
                let mut dense = sparse.to_dense();
                let changed = dense.insert(elem);
                assert!(changed);
                *self = HybridBitSet::Dense(dense);
                changed
            }
            HybridBitSet::Dense(dense) => dense.insert(elem),
        }
    }

    pub fn insert_all(&mut self) {
        let domain_size = self.domain_size();
        match self {
            HybridBitSet::Sparse(_) => {
                *self = HybridBitSet::Dense(BitSet::new_filled(domain_size));
            }
            HybridBitSet::Dense(dense) => dense.insert_all(),
        }
    }

    pub fn remove(&mut self, elem: T) -> bool {
        // Note: we currently don't bother going from Dense back to Sparse.
        match self {
            HybridBitSet::Sparse(sparse) => sparse.remove(elem),
            HybridBitSet::Dense(dense) => dense.remove(elem),
        }
    }

    pub fn union(&mut self, other: &HybridBitSet<T>) -> bool {
        match self {
            HybridBitSet::Sparse(self_sparse) => {
                match other {
                    HybridBitSet::Sparse(other_sparse) => {
                        // Both sets are sparse. Add the elements in
                        // `other_sparse` to `self` one at a time. This
                        // may or may not cause `self` to be densified.
                        assert_eq!(self.domain_size(), other.domain_size());
                        let mut changed = false;
                        for elem in other_sparse.iter() {
                            changed |= self.insert(*elem);
                        }
                        changed
                    }
                    HybridBitSet::Dense(other_dense) => {
                        // `self` is sparse and `other` is dense. To
                        // merge them, we have two available strategies:
                        // * Densify `self` then merge other
                        // * Clone other then integrate bits from `self`
                        // The second strategy requires dedicated method
                        // since the usual `union` returns the wrong
                        // result. In the dedicated case the computation
                        // is slightly faster if the bits of the sparse
                        // bitset map to only few words of the dense
                        // representation, i.e. indices are near each
                        // other.
                        //
                        // Benchmarking seems to suggest that the second
                        // option is worth it.
                        let mut new_dense = other_dense.clone();
                        let changed = new_dense.reverse_union_sparse(self_sparse);
                        *self = HybridBitSet::Dense(new_dense);
                        changed
                    }
                }
            }

            HybridBitSet::Dense(self_dense) => self_dense.union(other),
        }
    }

    /// Converts to a dense set, consuming itself in the process.
    pub fn to_dense(self) -> BitSet<T> {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.to_dense(),
            HybridBitSet::Dense(dense) => dense,
        }
    }

    pub fn iter(&self) -> HybridIter<'_, T> {
        match self {
            HybridBitSet::Sparse(sparse) => HybridIter::Sparse(sparse.iter()),
            HybridBitSet::Dense(dense) => HybridIter::Dense(dense.iter()),
        }
    }
}

impl<T: Idx> UnionIntoBitSet<T> for HybridBitSet<T> {
    fn union_into(&self, other: &mut BitSet<T>) -> bool {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.union_into(other),
            HybridBitSet::Dense(dense) => dense.union_into(other),
        }
    }
}

impl<T: Idx> SubtractFromBitSet<T> for HybridBitSet<T> {
    fn subtract_from(&self, other: &mut BitSet<T>) -> bool {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.subtract_from(other),
            HybridBitSet::Dense(dense) => dense.subtract_from(other),
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
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size.
#[derive(Clone, Debug, PartialEq)]
pub struct GrowableBitSet<T: Idx> {
    bit_set: BitSet<T>,
}

impl<T: Idx> GrowableBitSet<T> {
    /// Ensure that the set can hold at least `min_domain_size` elements.
    pub fn ensure(&mut self, min_domain_size: usize) {
        if self.bit_set.domain_size < min_domain_size {
            self.bit_set.domain_size = min_domain_size;
        }

        let min_num_words = num_words(min_domain_size);
        if self.bit_set.words.len() < min_num_words {
            self.bit_set.words.resize(min_num_words, 0)
        }
    }

    pub fn new_empty() -> GrowableBitSet<T> {
        GrowableBitSet { bit_set: BitSet::new_empty(0) }
    }

    pub fn with_capacity(capacity: usize) -> GrowableBitSet<T> {
        GrowableBitSet { bit_set: BitSet::new_empty(capacity) }
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        self.ensure(elem.index() + 1);
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
/// All operations that involve a row and/or column index will panic if the
/// index exceeds the relevant bound.
#[derive(Clone, Debug, Eq, PartialEq, RustcDecodable, RustcEncodable)]
pub struct BitMatrix<R: Idx, C: Idx> {
    num_rows: usize,
    num_columns: usize,
    words: Vec<Word>,
    marker: PhantomData<(R, C)>,
}

impl<R: Idx, C: Idx> BitMatrix<R, C> {
    /// Creates a new `rows x columns` matrix, initially empty.
    pub fn new(num_rows: usize, num_columns: usize) -> BitMatrix<R, C> {
        // For every element, we need one bit for every other
        // element. Round up to an even number of words.
        let words_per_row = num_words(num_columns);
        BitMatrix {
            num_rows,
            num_columns,
            words: vec![0; num_rows * words_per_row],
            marker: PhantomData,
        }
    }

    /// Creates a new matrix, with `row` used as the value for every row.
    pub fn from_row_n(row: &BitSet<C>, num_rows: usize) -> BitMatrix<R, C> {
        let num_columns = row.domain_size();
        let words_per_row = num_words(num_columns);
        assert_eq!(words_per_row, row.words().len());
        BitMatrix {
            num_rows,
            num_columns,
            words: iter::repeat(row.words()).take(num_rows).flatten().cloned().collect(),
            marker: PhantomData,
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = R> {
        (0..self.num_rows).map(R::new)
    }

    /// The range of bits for a given row.
    fn range(&self, row: R) -> (usize, usize) {
        let words_per_row = num_words(self.num_columns);
        let start = row.index() * words_per_row;
        (start, start + words_per_row)
    }

    /// Sets the cell at `(row, column)` to true. Put another way, insert
    /// `column` to the bitset for `row`.
    ///
    /// Returns `true` if this changed the matrix.
    pub fn insert(&mut self, row: R, column: C) -> bool {
        assert!(row.index() < self.num_rows && column.index() < self.num_columns);
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
    pub fn contains(&self, row: R, column: C) -> bool {
        assert!(row.index() < self.num_rows && column.index() < self.num_columns);
        let (start, _) = self.range(row);
        let (word_index, mask) = word_index_and_mask(column);
        (self.words[start + word_index] & mask) != 0
    }

    /// Returns those indices that are true in rows `a` and `b`. This
    /// is an O(n) operation where `n` is the number of elements
    /// (somewhat independent from the actual size of the
    /// intersection, in particular).
    pub fn intersect_rows(&self, row1: R, row2: R) -> Vec<C> {
        assert!(row1.index() < self.num_rows && row2.index() < self.num_rows);
        let (row1_start, row1_end) = self.range(row1);
        let (row2_start, row2_end) = self.range(row2);
        let mut result = Vec::with_capacity(self.num_columns);
        for (base, (i, j)) in (row1_start..row1_end).zip(row2_start..row2_end).enumerate() {
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

    /// Adds the bits from row `read` to the bits from row `write`, and
    /// returns `true` if anything changed.
    ///
    /// This is used when computing transitive reachability because if
    /// you have an edge `write -> read`, because in that case
    /// `write` can reach everything that `read` can (and
    /// potentially more).
    pub fn union_rows(&mut self, read: R, write: R) -> bool {
        assert!(read.index() < self.num_rows && write.index() < self.num_rows);
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

    /// Adds the bits from `with` to the bits from row `write`, and
    /// returns `true` if anything changed.
    pub fn union_row_with(&mut self, with: &BitSet<C>, write: R) -> bool {
        assert!(write.index() < self.num_rows);
        assert_eq!(with.domain_size(), self.num_columns);
        let (write_start, write_end) = self.range(write);
        let mut changed = false;
        for (read_index, write_index) in (0..with.words().len()).zip(write_start..write_end) {
            let word = self.words[write_index];
            let new_word = word | with.words()[read_index];
            self.words[write_index] = new_word;
            changed |= word != new_word;
        }
        changed
    }

    /// Sets every cell in `row` to true.
    pub fn insert_all_into_row(&mut self, row: R) {
        assert!(row.index() < self.num_rows);
        let (start, end) = self.range(row);
        let words = &mut self.words[..];
        for index in start..end {
            words[index] = !0;
        }
        self.clear_excess_bits(row);
    }

    /// Clear excess bits in the final word of the row.
    fn clear_excess_bits(&mut self, row: R) {
        let num_bits_in_final_word = self.num_columns % WORD_BITS;
        if num_bits_in_final_word > 0 {
            let mask = (1 << num_bits_in_final_word) - 1;
            let (_, end) = self.range(row);
            let final_word_idx = end - 1;
            self.words[final_word_idx] &= mask;
        }
    }

    /// Gets a slice of the underlying words.
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// Iterates through all the columns set to true in a given row of
    /// the matrix.
    pub fn iter(&self, row: R) -> BitIter<'_, C> {
        assert!(row.index() < self.num_rows);
        let (start, end) = self.range(row);
        BitIter {
            cur: None,
            iter: self.words[start..end].iter().enumerate(),
            marker: PhantomData,
        }
    }

    /// Returns the number of elements in `row`.
    pub fn count(&self, row: R) -> usize {
        let (start, end) = self.range(row);
        self.words[start..end].iter().map(|e| e.count_ones() as usize).sum()
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
    /// Creates a new empty sparse bit matrix with no rows or columns.
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
    /// Returns `true` if this changed the matrix.
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

    /// Adds the bits from row `read` to the bits from row `write`, and
    /// returns `true` if anything changed.
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
fn num_words<T: Idx>(domain_size: T) -> usize {
    (domain_size.index() + WORD_BITS - 1) / WORD_BITS
}

#[inline]
fn word_index_and_mask<T: Idx>(elem: T) -> (usize, Word) {
    let elem = elem.index();
    let word_index = elem / WORD_BITS;
    let mask = 1 << (elem % WORD_BITS);
    (word_index, mask)
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
    let mut bitset: BitSet<usize> = BitSet::new_empty(320);
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
    set.ensure(128);

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
    matrix.insert_all_into_row(6);

    let expected = [99];
    let mut iter = expected.iter();
    for i in matrix.iter(2) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    assert_eq!(matrix.count(3), expected.len());
    for i in matrix.iter(3) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [0];
    let mut iter = expected.iter();
    assert_eq!(matrix.count(4), expected.len());
    for i in matrix.iter(4) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    assert_eq!(matrix.count(5), expected.len());
    for i in matrix.iter(5) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    assert_eq!(matrix.count(6), 100);
    let mut count = 0;
    for (idx, i) in matrix.iter(6).enumerate() {
        assert_eq!(idx, i);
        count += 1;
    }
    assert_eq!(count, 100);

    if let Some(i) = matrix.iter(7).next() {
        panic!("expected no elements in row, but contains element {:?}", i);
    }
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

/// Merge dense hybrid set into empty sparse hybrid set.
#[bench]
fn union_hybrid_sparse_empty_to_dense(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    for i in 0..10 {
        assert!(pre_dense.insert(i));
    }
    let pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into full hybrid set with same indices.
#[bench]
fn union_hybrid_sparse_full_to_dense(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    for i in 0..10 {
        assert!(pre_dense.insert(i));
    }
    let mut pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    for i in 0..SPARSE_MAX {
        assert!(pre_sparse.insert(i));
    }
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into full hybrid set with indices over the whole domain.
#[bench]
fn union_hybrid_sparse_domain_to_dense(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX*64);
    for i in 0..10 {
        assert!(pre_dense.insert(i));
    }
    let mut pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX*64);
    for i in 0..SPARSE_MAX {
        assert!(pre_sparse.insert(i*64));
    }
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into empty hybrid set where the domain is very small.
#[bench]
fn union_hybrid_sparse_empty_small_domain(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    for i in 0..SPARSE_MAX {
        assert!(pre_dense.insert(i));
    }
    let pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into full hybrid set where the domain is very small.
#[bench]
fn union_hybrid_sparse_full_small_domain(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    for i in 0..SPARSE_MAX {
        assert!(pre_dense.insert(i));
    }
    let mut pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    for i in 0..SPARSE_MAX {
        assert!(pre_sparse.insert(i));
    }
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}
