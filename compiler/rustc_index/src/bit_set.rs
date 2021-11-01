use crate::vec::{Idx, IndexVec};
use arrayvec::ArrayVec;
use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::mem;
use std::ops::{BitAnd, BitAndAssign, BitOrAssign, Bound, Not, Range, RangeBounds, Shl};
use std::slice;

use rustc_macros::{Decodable, Encodable};

#[cfg(test)]
mod tests;

pub type Word = u64;
pub const WORD_BYTES: usize = mem::size_of::<Word>();
pub const WORD_BITS: usize = WORD_BYTES * 8;

pub trait BitRelations<Rhs> {
    fn union(&mut self, other: &Rhs) -> bool;
    fn subtract(&mut self, other: &Rhs) -> bool;
    fn intersect(&mut self, other: &Rhs) -> bool;
}

#[inline]
fn inclusive_start_end<T: Idx>(
    range: impl RangeBounds<T>,
    domain: usize,
) -> Option<(usize, usize)> {
    // Both start and end are inclusive.
    let start = match range.start_bound().cloned() {
        Bound::Included(start) => start.index(),
        Bound::Excluded(start) => start.index() + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound().cloned() {
        Bound::Included(end) => end.index(),
        Bound::Excluded(end) => end.index().checked_sub(1)?,
        Bound::Unbounded => domain - 1,
    };
    assert!(end < domain);
    if start > end {
        return None;
    }
    Some((start, end))
}

macro_rules! bit_relations_inherent_impls {
    () => {
        /// Sets `self = self | other` and returns `true` if `self` changed
        /// (i.e., if new bits were added).
        pub fn union<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::union(self, other)
        }

        /// Sets `self = self - other` and returns `true` if `self` changed.
        /// (i.e., if any bits were removed).
        pub fn subtract<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::subtract(self, other)
        }

        /// Sets `self = self & other` and return `true` if `self` changed.
        /// (i.e., if any bits were removed).
        pub fn intersect<Rhs>(&mut self, other: &Rhs) -> bool
        where
            Self: BitRelations<Rhs>,
        {
            <Self as BitRelations<Rhs>>::intersect(self, other)
        }
    };
}

/// A fixed-size bitset type with a dense representation.
///
/// NOTE: Use [`GrowableBitSet`] if you need support for resizing after creation.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
///
#[derive(Eq, PartialEq, Decodable, Encodable)]
pub struct BitSet<T> {
    domain_size: usize,
    words: Vec<Word>,
    marker: PhantomData<T>,
}

impl<T> BitSet<T> {
    /// Gets the domain size.
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }
}

impl<T: Idx> BitSet<T> {
    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        BitSet { domain_size, words: vec![0; num_words], marker: PhantomData }
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled(domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        let mut result = BitSet { domain_size, words: vec![!0; num_words], marker: PhantomData };
        result.clear_excess_bits();
        result
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

    #[inline]
    pub fn insert_range(&mut self, elems: impl RangeBounds<T>) {
        let Some((start, end)) = inclusive_start_end(elems, self.domain_size) else {
            return;
        };

        let (start_word_index, start_mask) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);

        // Set all words in between start and end (exclusively of both).
        for word_index in (start_word_index + 1)..end_word_index {
            self.words[word_index] = !0;
        }

        if start_word_index != end_word_index {
            // Start and end are in different words, so we handle each in turn.
            //
            // We set all leading bits. This includes the start_mask bit.
            self.words[start_word_index] |= !(start_mask - 1);
            // And all trailing bits (i.e. from 0..=end) in the end word,
            // including the end.
            self.words[end_word_index] |= end_mask | end_mask - 1;
        } else {
            self.words[start_word_index] |= end_mask | (end_mask - start_mask);
        }
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

    /// Gets a slice of the underlying words.
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        BitIter::new(&self.words)
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
                not_already |= self.words[current_index + 1..word_index].iter().any(|&x| x != 0);
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
        not_already |= self.words[current_index + 1..].iter().any(|&x| x != 0);

        not_already
    }

    fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T> {
        let (start, end) = inclusive_start_end(range, self.domain_size)?;
        let (start_word_index, _) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);

        let end_word = self.words[end_word_index] & (end_mask | (end_mask - 1));
        if end_word != 0 {
            let pos = max_bit(end_word) + WORD_BITS * end_word_index;
            if start <= pos {
                return Some(T::new(pos));
            }
        }

        // We exclude end_word_index from the range here, because we don't want
        // to limit ourselves to *just* the last word: the bits set it in may be
        // after `end`, so it may not work out.
        if let Some(offset) =
            self.words[start_word_index..end_word_index].iter().rposition(|&w| w != 0)
        {
            let word_idx = start_word_index + offset;
            let start_word = self.words[word_idx];
            let pos = max_bit(start_word) + WORD_BITS * word_idx;
            if start <= pos {
                return Some(T::new(pos));
            }
        }

        None
    }

    bit_relations_inherent_impls! {}
}

// dense REL dense
impl<T: Idx> BitRelations<BitSet<T>> for BitSet<T> {
    fn union(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut self.words, &other.words, |a, b| a | b)
    }

    fn subtract(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut self.words, &other.words, |a, b| a & !b)
    }

    fn intersect(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(&mut self.words, &other.words, |a, b| a & b)
    }
}

// Applies a function to mutate a bitset, and returns true if any
// of the applications return true
fn sequential_update<T: Idx>(
    mut self_update: impl FnMut(T) -> bool,
    it: impl Iterator<Item = T>,
) -> bool {
    let mut changed = false;
    for elem in it {
        changed |= self_update(elem);
    }
    changed
}

// Optimization of intersection for SparseBitSet that's generic
// over the RHS
fn sparse_intersect<T: Idx>(
    set: &mut SparseBitSet<T>,
    other_contains: impl Fn(&T) -> bool,
) -> bool {
    let size = set.elems.len();
    set.elems.retain(|elem| other_contains(elem));
    set.elems.len() != size
}

// Optimization of dense/sparse intersection. The resulting set is
// guaranteed to be at most the size of the sparse set, and hence can be
// represented as a sparse set. Therefore the sparse set is copied and filtered,
// then returned as the new set.
fn dense_sparse_intersect<T: Idx>(
    dense: &BitSet<T>,
    sparse: &SparseBitSet<T>,
) -> (SparseBitSet<T>, bool) {
    let mut sparse_copy = sparse.clone();
    sparse_intersect(&mut sparse_copy, |el| dense.contains(*el));
    let n = sparse_copy.len();
    (sparse_copy, n != dense.count())
}

// hybrid REL dense
impl<T: Idx> BitRelations<BitSet<T>> for HybridBitSet<T> {
    fn union(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size);
        match self {
            HybridBitSet::Sparse(sparse) => {
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
                let mut new_dense = other.clone();
                let changed = new_dense.reverse_union_sparse(sparse);
                *self = HybridBitSet::Dense(new_dense);
                changed
            }

            HybridBitSet::Dense(dense) => dense.union(other),
        }
    }

    fn subtract(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size);
        match self {
            HybridBitSet::Sparse(sparse) => {
                sequential_update(|elem| sparse.remove(elem), other.iter())
            }
            HybridBitSet::Dense(dense) => dense.subtract(other),
        }
    }

    fn intersect(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size);
        match self {
            HybridBitSet::Sparse(sparse) => sparse_intersect(sparse, |elem| other.contains(*elem)),
            HybridBitSet::Dense(dense) => dense.intersect(other),
        }
    }
}

// dense REL hybrid
impl<T: Idx> BitRelations<HybridBitSet<T>> for BitSet<T> {
    fn union(&mut self, other: &HybridBitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size());
        match other {
            HybridBitSet::Sparse(sparse) => {
                sequential_update(|elem| self.insert(elem), sparse.iter().cloned())
            }
            HybridBitSet::Dense(dense) => self.union(dense),
        }
    }

    fn subtract(&mut self, other: &HybridBitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size());
        match other {
            HybridBitSet::Sparse(sparse) => {
                sequential_update(|elem| self.remove(elem), sparse.iter().cloned())
            }
            HybridBitSet::Dense(dense) => self.subtract(dense),
        }
    }

    fn intersect(&mut self, other: &HybridBitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size());
        match other {
            HybridBitSet::Sparse(sparse) => {
                let (updated, changed) = dense_sparse_intersect(self, sparse);

                // We can't directly assign the SparseBitSet to the BitSet, and
                // doing `*self = updated.to_dense()` would cause a drop / reallocation. Instead,
                // the BitSet is cleared and `updated` is copied into `self`.
                self.clear();
                for elem in updated.iter() {
                    self.insert(*elem);
                }
                changed
            }
            HybridBitSet::Dense(dense) => self.intersect(dense),
        }
    }
}

// hybrid REL hybrid
impl<T: Idx> BitRelations<HybridBitSet<T>> for HybridBitSet<T> {
    fn union(&mut self, other: &HybridBitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size());
        match self {
            HybridBitSet::Sparse(_) => {
                match other {
                    HybridBitSet::Sparse(other_sparse) => {
                        // Both sets are sparse. Add the elements in
                        // `other_sparse` to `self` one at a time. This
                        // may or may not cause `self` to be densified.
                        let mut changed = false;
                        for elem in other_sparse.iter() {
                            changed |= self.insert(*elem);
                        }
                        changed
                    }

                    HybridBitSet::Dense(other_dense) => self.union(other_dense),
                }
            }

            HybridBitSet::Dense(self_dense) => self_dense.union(other),
        }
    }

    fn subtract(&mut self, other: &HybridBitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size());
        match self {
            HybridBitSet::Sparse(self_sparse) => {
                sequential_update(|elem| self_sparse.remove(elem), other.iter())
            }
            HybridBitSet::Dense(self_dense) => self_dense.subtract(other),
        }
    }

    fn intersect(&mut self, other: &HybridBitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size());
        match self {
            HybridBitSet::Sparse(self_sparse) => {
                sparse_intersect(self_sparse, |elem| other.contains(*elem))
            }
            HybridBitSet::Dense(self_dense) => match other {
                HybridBitSet::Sparse(other_sparse) => {
                    let (updated, changed) = dense_sparse_intersect(self_dense, other_sparse);
                    *self = HybridBitSet::Sparse(updated);
                    changed
                }
                HybridBitSet::Dense(other_dense) => self_dense.intersect(other_dense),
            },
        }
    }
}

impl<T> Clone for BitSet<T> {
    fn clone(&self) -> Self {
        BitSet { domain_size: self.domain_size, words: self.words.clone(), marker: PhantomData }
    }

    fn clone_from(&mut self, from: &Self) {
        if self.domain_size != from.domain_size {
            self.words.resize(from.domain_size, 0);
            self.domain_size = from.domain_size;
        }

        self.words.copy_from_slice(&from.words);
    }
}

impl<T: Idx> fmt::Debug for BitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
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
            for _ in 0..WORD_BYTES {
                // for each byte in `word`:
                let remain = self.domain_size - i;
                // If less than a byte remains, then mask just that many bits.
                let mask = if remain <= 8 { (1 << remain) - 1 } else { 0xFF };
                assert!(mask <= 0xFF);
                let byte = word & mask;

                result.push_str(&format!("{}{:02x}", sep, byte));

                if remain <= 8 {
                    break;
                }
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
    /// A copy of the current word, but with any already-visited bits cleared.
    /// (This lets us use `trailing_zeros()` to find the next set bit.) When it
    /// is reduced to 0, we move onto the next word.
    word: Word,

    /// The offset (measured in bits) of the current word.
    offset: usize,

    /// Underlying iterator over the words.
    iter: slice::Iter<'a, Word>,

    marker: PhantomData<T>,
}

impl<'a, T: Idx> BitIter<'a, T> {
    #[inline]
    fn new(words: &'a [Word]) -> BitIter<'a, T> {
        // We initialize `word` and `offset` to degenerate values. On the first
        // call to `next()` we will fall through to getting the first word from
        // `iter`, which sets `word` to the first word (if there is one) and
        // `offset` to 0. Doing it this way saves us from having to maintain
        // additional state about whether we have started.
        BitIter {
            word: 0,
            offset: usize::MAX - (WORD_BITS - 1),
            iter: words.iter(),
            marker: PhantomData,
        }
    }
}

impl<'a, T: Idx> Iterator for BitIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        loop {
            if self.word != 0 {
                // Get the position of the next set bit in the current word,
                // then clear the bit.
                let bit_pos = self.word.trailing_zeros() as usize;
                let bit = 1 << bit_pos;
                self.word ^= bit;
                return Some(T::new(bit_pos + self.offset));
            }

            // Move onto the next word. `wrapping_add()` is needed to handle
            // the degenerate initial value given to `offset` in `new()`.
            let word = self.iter.next()?;
            self.word = *word;
            self.offset = self.offset.wrapping_add(WORD_BITS);
        }
    }
}

#[inline]
fn bitwise<Op>(out_vec: &mut [Word], in_vec: &[Word], op: Op) -> bool
where
    Op: Fn(Word, Word) -> Word,
{
    assert_eq!(out_vec.len(), in_vec.len());
    let mut changed = 0;
    for (out_elem, in_elem) in iter::zip(out_vec, in_vec) {
        let old_val = *out_elem;
        let new_val = op(old_val, *in_elem);
        *out_elem = new_val;
        // This is essentially equivalent to a != with changed being a bool, but
        // in practice this code gets auto-vectorized by the compiler for most
        // operators. Using != here causes us to generate quite poor code as the
        // compiler tries to go back to a boolean on each loop iteration.
        changed |= old_val ^ new_val;
    }
    changed != 0
}

const SPARSE_MAX: usize = 8;

/// A fixed-size bitset type with a sparse representation and a maximum of
/// `SPARSE_MAX` elements. The elements are stored as a sorted `ArrayVec` with
/// no duplicates.
///
/// This type is used by `HybridBitSet`; do not use directly.
#[derive(Clone, Debug)]
pub struct SparseBitSet<T> {
    domain_size: usize,
    elems: ArrayVec<T, SPARSE_MAX>,
}

impl<T: Idx> SparseBitSet<T> {
    fn new_empty(domain_size: usize) -> Self {
        SparseBitSet { domain_size, elems: ArrayVec::new() }
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

    fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T> {
        let mut last_leq = None;
        for e in self.iter() {
            if range.contains(e) {
                last_leq = Some(*e);
            }
        }
        last_leq
    }

    bit_relations_inherent_impls! {}
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
#[derive(Clone)]
pub enum HybridBitSet<T> {
    Sparse(SparseBitSet<T>),
    Dense(BitSet<T>),
}

impl<T: Idx> fmt::Debug for HybridBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sparse(b) => b.fmt(w),
            Self::Dense(b) => b.fmt(w),
        }
    }
}

impl<T: Idx> HybridBitSet<T> {
    pub fn new_empty(domain_size: usize) -> Self {
        HybridBitSet::Sparse(SparseBitSet::new_empty(domain_size))
    }

    pub fn domain_size(&self) -> usize {
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

    /// Returns the previous element present in the bitset from `elem`,
    /// inclusively of elem. That is, will return `Some(elem)` if elem is in the
    /// bitset.
    pub fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T> {
        match self {
            HybridBitSet::Sparse(sparse) => sparse.last_set_in(range),
            HybridBitSet::Dense(dense) => dense.last_set_in(range),
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

    pub fn insert_range(&mut self, elems: impl RangeBounds<T>) {
        // No need to check `elem` against `self.domain_size` here because all
        // the match cases check it, one way or another.
        let start = match elems.start_bound().cloned() {
            Bound::Included(start) => start.index(),
            Bound::Excluded(start) => start.index() + 1,
            Bound::Unbounded => 0,
        };
        let end = match elems.end_bound().cloned() {
            Bound::Included(end) => end.index() + 1,
            Bound::Excluded(end) => end.index(),
            Bound::Unbounded => self.domain_size() - 1,
        };
        let len = if let Some(l) = end.checked_sub(start) {
            l
        } else {
            return;
        };
        match self {
            HybridBitSet::Sparse(sparse) if sparse.len() + len < SPARSE_MAX => {
                // The set is sparse and has space for `elems`.
                for elem in start..end {
                    sparse.insert(T::new(elem));
                }
            }
            HybridBitSet::Sparse(sparse) => {
                // The set is sparse and full. Convert to a dense set.
                let mut dense = sparse.to_dense();
                dense.insert_range(elems);
                *self = HybridBitSet::Dense(dense);
            }
            HybridBitSet::Dense(dense) => dense.insert_range(elems),
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

    bit_relations_inherent_impls! {}
}

pub enum HybridIter<'a, T: Idx> {
    Sparse(slice::Iter<'a, T>),
    Dense(BitIter<'a, T>),
}

impl<'a, T: Idx> Iterator for HybridIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match self {
            HybridIter::Sparse(sparse) => sparse.next().copied(),
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

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        self.ensure(elem.index() + 1);
        self.bit_set.remove(elem)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bit_set.is_empty()
    }

    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        let (word_index, mask) = word_index_and_mask(elem);
        self.bit_set.words.get(word_index).map_or(false, |word| (word & mask) != 0)
    }
}

/// A fixed-size 2D bit matrix type with a dense representation.
///
/// `R` and `C` are index types used to identify rows and columns respectively;
/// typically newtyped `usize` wrappers, but they can also just be `usize`.
///
/// All operations that involve a row and/or column index will panic if the
/// index exceeds the relevant bound.
#[derive(Clone, Eq, PartialEq, Decodable, Encodable)]
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
    /// is an *O*(*n*) operation where *n* is the number of elements
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
        for (read_index, write_index) in iter::zip(read_start..read_end, write_start..write_end) {
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
        for (read_index, write_index) in iter::zip(0..with.words().len(), write_start..write_end) {
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
        BitIter::new(&self.words[start..end])
    }

    /// Returns the number of elements in `row`.
    pub fn count(&self, row: R) -> usize {
        let (start, end) = self.range(row);
        self.words[start..end].iter().map(|e| e.count_ones() as usize).sum()
    }
}

impl<R: Idx, C: Idx> fmt::Debug for BitMatrix<R, C> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        /// Forces its contents to print in regular mode instead of alternate mode.
        struct OneLinePrinter<T>(T);
        impl<T: fmt::Debug> fmt::Debug for OneLinePrinter<T> {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(fmt, "{:?}", self.0)
            }
        }

        write!(fmt, "BitMatrix({}x{}) ", self.num_rows, self.num_columns)?;
        let items = self.rows().flat_map(|r| self.iter(r).map(move |c| (r, c)));
        fmt.debug_set().entries(items.map(OneLinePrinter)).finish()
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
        Self { num_columns, rows: IndexVec::new() }
    }

    fn ensure_row(&mut self, row: R) -> &mut HybridBitSet<C> {
        // Instantiate any missing rows up to and including row `row` with an empty HybridBitSet.
        // Then replace row `row` with a full HybridBitSet if necessary.
        self.rows.get_or_insert_with(row, || HybridBitSet::new_empty(self.num_columns))
    }

    /// Sets the cell at `(row, column)` to true. Put another way, insert
    /// `column` to the bitset for `row`.
    ///
    /// Returns `true` if this changed the matrix.
    pub fn insert(&mut self, row: R, column: C) -> bool {
        self.ensure_row(row).insert(column)
    }

    /// Sets the cell at `(row, column)` to false. Put another way, delete
    /// `column` from the bitset for `row`. Has no effect if `row` does not
    /// exist.
    ///
    /// Returns `true` if this changed the matrix.
    pub fn remove(&mut self, row: R, column: C) -> bool {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.remove(column),
            _ => false,
        }
    }

    /// Sets all columns at `row` to false. Has no effect if `row` does
    /// not exist.
    pub fn clear(&mut self, row: R) {
        if let Some(Some(row)) = self.rows.get_mut(row) {
            row.clear();
        }
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
        if let Some(Some(row)) = self.rows.get(row) { Some(row) } else { None }
    }

    /// Interescts `row` with `set`. `set` can be either `BitSet` or
    /// `HybridBitSet`. Has no effect if `row` does not exist.
    ///
    /// Returns true if the row was changed.
    pub fn intersect_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        HybridBitSet<C>: BitRelations<Set>,
    {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.intersect(set),
            _ => false,
        }
    }

    /// Subtracts `set from `row`. `set` can be either `BitSet` or
    /// `HybridBitSet`. Has no effect if `row` does not exist.
    ///
    /// Returns true if the row was changed.
    pub fn subtract_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        HybridBitSet<C>: BitRelations<Set>,
    {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.subtract(set),
            _ => false,
        }
    }

    /// Unions `row` with `set`. `set` can be either `BitSet` or
    /// `HybridBitSet`.
    ///
    /// Returns true if the row was changed.
    pub fn union_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        HybridBitSet<C>: BitRelations<Set>,
    {
        self.ensure_row(row).union(set)
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

#[inline]
fn max_bit(word: Word) -> usize {
    WORD_BITS - 1 - word.leading_zeros() as usize
}

/// Integral type used to represent the bit set.
pub trait FiniteBitSetTy:
    BitAnd<Output = Self>
    + BitAndAssign
    + BitOrAssign
    + Clone
    + Copy
    + Shl
    + Not<Output = Self>
    + PartialEq
    + Sized
{
    /// Size of the domain representable by this type, e.g. 64 for `u64`.
    const DOMAIN_SIZE: u32;

    /// Value which represents the `FiniteBitSet` having every bit set.
    const FILLED: Self;
    /// Value which represents the `FiniteBitSet` having no bits set.
    const EMPTY: Self;

    /// Value for one as the integral type.
    const ONE: Self;
    /// Value for zero as the integral type.
    const ZERO: Self;

    /// Perform a checked left shift on the integral type.
    fn checked_shl(self, rhs: u32) -> Option<Self>;
    /// Perform a checked right shift on the integral type.
    fn checked_shr(self, rhs: u32) -> Option<Self>;
}

impl FiniteBitSetTy for u32 {
    const DOMAIN_SIZE: u32 = 32;

    const FILLED: Self = Self::MAX;
    const EMPTY: Self = Self::MIN;

    const ONE: Self = 1u32;
    const ZERO: Self = 0u32;

    fn checked_shl(self, rhs: u32) -> Option<Self> {
        self.checked_shl(rhs)
    }

    fn checked_shr(self, rhs: u32) -> Option<Self> {
        self.checked_shr(rhs)
    }
}

impl std::fmt::Debug for FiniteBitSet<u32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032b}", self.0)
    }
}

impl FiniteBitSetTy for u64 {
    const DOMAIN_SIZE: u32 = 64;

    const FILLED: Self = Self::MAX;
    const EMPTY: Self = Self::MIN;

    const ONE: Self = 1u64;
    const ZERO: Self = 0u64;

    fn checked_shl(self, rhs: u32) -> Option<Self> {
        self.checked_shl(rhs)
    }

    fn checked_shr(self, rhs: u32) -> Option<Self> {
        self.checked_shr(rhs)
    }
}

impl std::fmt::Debug for FiniteBitSet<u64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:064b}", self.0)
    }
}

impl FiniteBitSetTy for u128 {
    const DOMAIN_SIZE: u32 = 128;

    const FILLED: Self = Self::MAX;
    const EMPTY: Self = Self::MIN;

    const ONE: Self = 1u128;
    const ZERO: Self = 0u128;

    fn checked_shl(self, rhs: u32) -> Option<Self> {
        self.checked_shl(rhs)
    }

    fn checked_shr(self, rhs: u32) -> Option<Self> {
        self.checked_shr(rhs)
    }
}

impl std::fmt::Debug for FiniteBitSet<u128> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:0128b}", self.0)
    }
}

/// A fixed-sized bitset type represented by an integer type. Indices outwith than the range
/// representable by `T` are considered set.
#[derive(Copy, Clone, Eq, PartialEq, Decodable, Encodable)]
pub struct FiniteBitSet<T: FiniteBitSetTy>(pub T);

impl<T: FiniteBitSetTy> FiniteBitSet<T> {
    /// Creates a new, empty bitset.
    pub fn new_empty() -> Self {
        Self(T::EMPTY)
    }

    /// Sets the `index`th bit.
    pub fn set(&mut self, index: u32) {
        self.0 |= T::ONE.checked_shl(index).unwrap_or(T::ZERO);
    }

    /// Unsets the `index`th bit.
    pub fn clear(&mut self, index: u32) {
        self.0 &= !T::ONE.checked_shl(index).unwrap_or(T::ZERO);
    }

    /// Sets the `i`th to `j`th bits.
    pub fn set_range(&mut self, range: Range<u32>) {
        let bits = T::FILLED
            .checked_shl(range.end - range.start)
            .unwrap_or(T::ZERO)
            .not()
            .checked_shl(range.start)
            .unwrap_or(T::ZERO);
        self.0 |= bits;
    }

    /// Is the set empty?
    pub fn is_empty(&self) -> bool {
        self.0 == T::EMPTY
    }

    /// Returns the domain size of the bitset.
    pub fn within_domain(&self, index: u32) -> bool {
        index < T::DOMAIN_SIZE
    }

    /// Returns if the `index`th bit is set.
    pub fn contains(&self, index: u32) -> Option<bool> {
        self.within_domain(index)
            .then(|| ((self.0.checked_shr(index).unwrap_or(T::ONE)) & T::ONE) == T::ONE)
    }
}

impl<T: FiniteBitSetTy> Default for FiniteBitSet<T> {
    fn default() -> Self {
        Self::new_empty()
    }
}
