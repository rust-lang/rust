use std::fmt;
use std::iter::ExactSizeIterator;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext};
use smallvec::{SmallVec, smallvec};

use super::Chunk::*;
use super::{
    BitIter, BitRelations, CHUNK_WORDS, ChunkedBitSet, WORD_BITS, WORD_BYTES, Word,
    bit_relations_inherent_impls, bitwise, clear_excess_bits_in_final_word, num_words,
    word_index_and_mask,
};
use crate::Idx;

/// A fixed-size bitset type with a dense representation.
///
/// Note 1: Since this bitset is dense, if your domain is big, and/or relatively
/// homogeneous (for example, with long runs of bits set or unset), then it may
/// be preferable to instead use a [MixedBitSet], or an
/// [IntervalSet](crate::interval::IntervalSet). They should be more suited to
/// sparse, or highly-compressible, domains.
///
/// Note 2: Use [`GrowableBitSet`] if you need support for resizing after creation.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also
/// just be `usize`.
///
/// All operations that involve an element will panic if the element is equal
/// to or greater than the domain size. All operations that involve two bitsets
/// will panic if the bitsets have differing domain sizes.
///
#[cfg_attr(feature = "nightly", derive(Decodable_NoContext, Encodable_NoContext))]
#[derive(Eq, PartialEq, Hash)]
pub struct DenseBitSet<T> {
    domain_size: usize,
    words: SmallVec<[Word; 2]>,
    marker: PhantomData<T>,
}

impl<T> DenseBitSet<T> {
    /// Gets the domain size.
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }
}

impl<T: Idx> DenseBitSet<T> {
    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> DenseBitSet<T> {
        let num_words = num_words(domain_size);
        DenseBitSet { domain_size, words: smallvec![0; num_words], marker: PhantomData }
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled(domain_size: usize) -> DenseBitSet<T> {
        let num_words = num_words(domain_size);
        let mut result =
            DenseBitSet { domain_size, words: smallvec![!0; num_words], marker: PhantomData };
        result.clear_excess_bits();
        result
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Clear excess bits in the final word.
    fn clear_excess_bits(&mut self) {
        clear_excess_bits_in_final_word(self.domain_size, &mut self.words);
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
    pub fn superset(&self, other: &DenseBitSet<T>) -> bool {
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
        assert!(
            elem.index() < self.domain_size,
            "inserting element at index {} but domain size is {}",
            elem.index(),
            self.domain_size,
        );
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
            self.words[end_word_index] |= end_mask | (end_mask - 1);
        } else {
            self.words[start_word_index] |= end_mask | (end_mask - start_mask);
        }
    }

    /// Sets all bits to true.
    pub fn insert_all(&mut self, _domain_size: usize) {
        self.words.fill(!0);
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

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        BitIter::from_slice(&self.words)
    }

    pub fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T> {
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

    /// Sets `self = self | !other`.
    ///
    /// FIXME: Incorporate this into [`BitRelations`] and fill out
    /// implementations for other bitset types, if needed.
    pub fn union_not(&mut self, other: &DenseBitSet<T>) {
        assert_eq!(self.domain_size, other.domain_size);

        // FIXME(Zalathar): If we were to forcibly _set_ all excess bits before
        // the bitwise update, and then clear them again afterwards, we could
        // quickly and accurately detect whether the update changed anything.
        // But that's only worth doing if there's an actual use-case.

        bitwise(self.words.iter_mut(), other.words.iter().copied(), |a, b| a | !b);
        // The bitwise update `a | !b` can result in the last word containing
        // out-of-domain bits, so we need to clear them.
        self.clear_excess_bits();
    }

    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        self.domain_size()
    }

    // FIXME: Just a wrapper around insert_range.
    #[inline]
    pub fn insert_range_inclusive(&mut self, elems: impl RangeBounds<T>) {
        self.insert_range(elems);
    }
}

impl<T> DenseBitSet<T> {
    #[inline]
    pub(crate) fn words(&self) -> impl ExactSizeIterator<Item = Word> {
        self.words.iter().copied()
    }
}

// dense REL dense
impl<T: Idx> BitRelations<DenseBitSet<T>> for DenseBitSet<T> {
    fn union(&mut self, other: &DenseBitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(self.words.iter_mut(), other.words.iter().copied(), |a, b| a | b)
    }

    fn subtract(&mut self, other: &DenseBitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(self.words.iter_mut(), other.words.iter().copied(), |a, b| a & !b)
    }

    fn intersect(&mut self, other: &DenseBitSet<T>) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise(self.words.iter_mut(), other.words.iter().copied(), |a, b| a & b)
    }
}

impl<T: Idx> From<GrowableBitSet<T>> for DenseBitSet<T> {
    fn from(bit_set: GrowableBitSet<T>) -> Self {
        bit_set.bit_set
    }
}

impl<T> Clone for DenseBitSet<T> {
    fn clone(&self) -> Self {
        DenseBitSet {
            domain_size: self.domain_size,
            words: self.words.clone(),
            marker: PhantomData,
        }
    }

    fn clone_from(&mut self, from: &Self) {
        self.domain_size = from.domain_size;
        self.words.clone_from(&from.words);
    }
}

impl<T: Idx> fmt::Debug for DenseBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Idx> ToString for DenseBitSet<T> {
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

                result.push_str(&format!("{sep}{byte:02x}"));

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

impl<T: Idx> BitRelations<ChunkedBitSet<T>> for DenseBitSet<T> {
    fn union(&mut self, other: &ChunkedBitSet<T>) -> bool {
        sequential_update(|elem| self.insert(elem), other.iter())
    }

    fn subtract(&mut self, _other: &ChunkedBitSet<T>) -> bool {
        unimplemented!("implement if/when necessary");
    }

    fn intersect(&mut self, other: &ChunkedBitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size);
        let mut changed = false;
        for (i, chunk) in other.chunks.iter().enumerate() {
            let mut words = &mut self.words[i * CHUNK_WORDS..];
            if words.len() > CHUNK_WORDS {
                words = &mut words[..CHUNK_WORDS];
            }
            match chunk {
                Zeros(..) => {
                    for word in words {
                        if *word != 0 {
                            changed = true;
                            *word = 0;
                        }
                    }
                }
                Ones(..) => (),
                Mixed(_, _, data) => {
                    for (i, word) in words.iter_mut().enumerate() {
                        let new_val = *word & data[i];
                        if new_val != *word {
                            changed = true;
                            *word = new_val;
                        }
                    }
                }
            }
        }
        changed
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
    bit_set: DenseBitSet<T>,
}

impl<T: Idx> Default for GrowableBitSet<T> {
    fn default() -> Self {
        GrowableBitSet::new_empty()
    }
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
        GrowableBitSet { bit_set: DenseBitSet::new_empty(0) }
    }

    pub fn with_capacity(capacity: usize) -> GrowableBitSet<T> {
        GrowableBitSet { bit_set: DenseBitSet::new_empty(capacity) }
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
        self.bit_set.words.get(word_index).is_some_and(|word| (word & mask) != 0)
    }

    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        self.bit_set.iter()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bit_set.count()
    }
}

impl<T: Idx> From<DenseBitSet<T>> for GrowableBitSet<T> {
    fn from(bit_set: DenseBitSet<T>) -> Self {
        Self { bit_set }
    }
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

#[inline]
fn max_bit(word: Word) -> usize {
    WORD_BITS - 1 - word.leading_zeros() as usize
}

// Applies a function to mutate a bitset, and returns true if any
// of the applications return true
fn sequential_update<T: Idx>(
    mut self_update: impl FnMut(T) -> bool,
    it: impl Iterator<Item = T>,
) -> bool {
    it.fold(false, |changed, elem| self_update(elem) | changed)
}
