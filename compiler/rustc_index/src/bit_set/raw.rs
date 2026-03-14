use std::ops::RangeBounds;
use std::{fmt, iter, slice};

use crate::bit_set::{
    BitRelations, WORD_BITS, Word, bitwise_update_words, clear_excess_bits_in_final_word,
    count_ones, inclusive_start_end, max_bit, num_words, word_index_and_mask,
};

#[cfg_attr(
    feature = "nightly",
    derive(rustc_macros::Decodable_NoContext, rustc_macros::Encodable_NoContext)
)]
#[derive(Eq, PartialEq, Hash)]
pub(crate) struct RawBitSet {
    domain_size: usize,
    words: Vec<Word>,
}

impl Clone for RawBitSet {
    #[inline]
    fn clone(&self) -> Self {
        let &Self { domain_size, ref words } = self;
        Self { domain_size, words: words.clone() }
    }

    #[inline]
    fn clone_from(&mut self, other: &Self) {
        self.domain_size = other.domain_size;
        self.words.clone_from(&other.words);
    }
}

impl RawBitSet {
    #[inline]
    pub(crate) fn domain_size(&self) -> usize {
        self.domain_size
    }

    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub(crate) fn new_empty(domain_size: usize) -> Self {
        let num_words = num_words(domain_size);
        RawBitSet { domain_size, words: vec![0; num_words] }
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub(crate) fn new_filled(domain_size: usize) -> Self {
        let num_words = num_words(domain_size);
        let mut result = RawBitSet { domain_size, words: vec![!0; num_words] };
        result.clear_excess_bits();
        result
    }

    /// Provides direct access to the underlying slice of words, as currently
    /// needed by a few methods in `BitMatrix`.
    /// Try to avoid using this for any other purpose.
    #[inline]
    pub(crate) fn raw_words(&self) -> &[Word] {
        self.words.as_slice()
    }

    /// Returns true if this bitset contains no elements.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.words.as_slice().iter().all(|&w| w == 0)
    }

    /// Count the number of set bits in the set.
    #[inline]
    pub(crate) fn count(&self) -> usize {
        count_ones(self.words.as_slice())
    }

    /// Iterates over the indices of set bits, in ascending order.
    #[inline]
    pub(crate) fn iter(&self) -> RawBitIter<'_> {
        RawBitIter::new(self.words.as_slice())
    }

    /// Returns true if this bitset contains `elem`.
    #[inline]
    pub(crate) fn contains(&self, elem: usize) -> bool {
        assert!(elem < self.domain_size);

        let (word_index, mask) = word_index_and_mask(elem);
        let words = self.words.as_slice();

        (words[word_index] & mask) != 0
    }

    /// Alternate implementation of [`Self::contains`] that does not panic
    /// if `elem >= self.domain_size`, for use by [`super::GrowableBitSet`].
    #[inline]
    pub(crate) fn growable_contains(&self, elem: usize) -> bool {
        let (word_index, mask) = word_index_and_mask(elem);
        self.words.get(word_index).is_some_and(|word| (word & mask) != 0)
    }

    /// Returns true if this bitset contains any values in the specified range,
    /// i.e. any bit in the given range is a 1.
    #[inline]
    pub(crate) fn contains_any(&self, elems: impl RangeBounds<usize>) -> bool {
        let Some((start, end)) = inclusive_start_end(elems, self.domain_size) else {
            return false;
        };

        let (start_word_index, start_mask) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);
        let words = self.words.as_slice();

        if start_word_index == end_word_index {
            words[start_word_index] & (end_mask | (end_mask - start_mask)) != 0
        } else {
            if words[start_word_index] & !(start_mask - 1) != 0 {
                return true;
            }

            let remaining = start_word_index + 1..end_word_index;
            if remaining.start <= remaining.end {
                words[remaining].iter().any(|&w| w != 0)
                    || words[end_word_index] & (end_mask | (end_mask - 1)) != 0
            } else {
                false
            }
        }
    }

    #[inline]
    pub(crate) fn last_set_in(&self, range: impl RangeBounds<usize>) -> Option<usize> {
        let (start, end) = inclusive_start_end(range, self.domain_size)?;
        let (start_word_index, _) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);
        let words = self.words.as_slice();

        let end_word = words[end_word_index] & (end_mask | (end_mask - 1));
        if end_word != 0 {
            let pos = max_bit(end_word) + WORD_BITS * end_word_index;
            if start <= pos {
                return Some(pos);
            }
        }

        // We exclude end_word_index from the range here, because we don't want
        // to limit ourselves to *just* the last word: the bits set it in may be
        // after `end`, so it may not work out.
        if let Some(offset) = words[start_word_index..end_word_index].iter().rposition(|&w| w != 0)
        {
            let word_idx = start_word_index + offset;
            let start_word = words[word_idx];
            let pos = max_bit(start_word) + WORD_BITS * word_idx;
            if start <= pos {
                return Some(pos);
            }
        }

        None
    }

    /// Returns true if this bitset is a (non-strict) superset of `other`,
    /// i.e. it contains every element of `other`.
    ///
    /// Equal sets are considered "supersets" of each other.
    ///
    /// Panics if the two bitsets have different domain sizes.
    #[inline]
    pub(crate) fn superset(&self, other: &RawBitSet) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        iter::zip(self.words.as_slice(), other.words.as_slice()).all(|(&a, &b)| (a & b) == b)
    }

    /// Remove all elements from this bitset.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.words.as_mut_slice().fill(0);
    }

    /// Clears any bits in the final word that are beyond `self.domain_size`,
    /// to restore the invariant that out-of-domain bits are always clear.
    #[inline]
    fn clear_excess_bits(&mut self) {
        clear_excess_bits_in_final_word(self.domain_size, self.words.as_mut_slice());
    }

    /// Sets all bits to true.
    #[inline]
    pub(crate) fn insert_all(&mut self) {
        self.words.as_mut_slice().fill(!0);
        self.clear_excess_bits();
    }

    /// Inserts `elem` into this set.
    ///
    /// Returns true if the set changed, i.e. the element was not previously in the set.
    #[inline]
    pub(crate) fn insert(&mut self, elem: usize) -> bool {
        let domain_size = self.domain_size;
        assert!(
            elem < domain_size,
            "inserting element at index {elem} but domain size is {domain_size}",
        );

        let (word_index, mask) = word_index_and_mask(elem);
        let words = self.words.as_mut_slice();
        let word_slot = &mut words[word_index];

        let old_word = *word_slot;
        let new_word = old_word | mask;

        *word_slot = new_word;
        old_word != new_word
    }

    /// Removes `elem` from this set, if present.
    ///
    /// Returns true if the set changed, i.e. the element was previously in the set.
    #[inline]
    pub(crate) fn remove(&mut self, elem: usize) -> bool {
        assert!(elem < self.domain_size);

        let (word_index, mask) = word_index_and_mask(elem);
        let words = self.words.as_mut_slice();
        let word_slot = &mut words[word_index];

        let old_word = *word_slot;
        let new_word = old_word & !mask;

        *word_slot = new_word;
        old_word != new_word
    }

    #[inline]
    pub(crate) fn insert_range(&mut self, elems: impl RangeBounds<usize>) {
        let Some((start, end)) = inclusive_start_end(elems, self.domain_size) else {
            return;
        };

        let (start_word_index, start_mask) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);
        let words = self.words.as_mut_slice();

        // Set all words in between start and end (exclusively of both).
        for word_index in (start_word_index + 1)..end_word_index {
            words[word_index] = !0;
        }

        if start_word_index != end_word_index {
            // Start and end are in different words, so we handle each in turn.
            //
            // We set all leading bits. This includes the start_mask bit.
            words[start_word_index] |= !(start_mask - 1);
            // And all trailing bits (i.e. from 0..=end) in the end word,
            // including the end.
            words[end_word_index] |= end_mask | (end_mask - 1);
        } else {
            words[start_word_index] |= end_mask | (end_mask - start_mask);
        }
    }

    /// Sets `self = self | !other`.
    ///
    /// FIXME: Incorporate this into [`BitRelations`] and fill out
    /// implementations for other bitset types, if needed.
    #[inline]
    pub(crate) fn union_not(&mut self, other: &RawBitSet) {
        assert_eq!(self.domain_size, other.domain_size);

        // FIXME(Zalathar): If we were to forcibly _set_ all excess bits before
        // the bitwise update, and then clear them again afterwards, we could
        // quickly and accurately detect whether the update changed anything.
        // But that's only worth doing if there's an actual use-case.

        bitwise_update_words(self.words.as_mut_slice(), other.words.as_slice(), |a, b| a | !b);
        // The bitwise update `a | !b` can result in the last word containing
        // out-of-domain bits, so we need to clear them.
        self.clear_excess_bits();
    }

    /// Adjust this bitset's domain size to be at least `min_domain_size`,
    /// for use by [`super::GrowableBitSet`].
    #[inline]
    pub(crate) fn growable_ensure(&mut self, min_domain_size: usize) {
        if self.domain_size < min_domain_size {
            self.domain_size = min_domain_size;
        }

        let min_num_words = num_words(min_domain_size);
        if self.words.len() < min_num_words {
            self.words.resize(min_num_words, 0)
        }
    }
}

pub(crate) struct RawBitIter<'a> {
    /// A copy of the current word, but with any already-visited bits cleared.
    /// (This lets us use `trailing_zeros()` to find the next set bit.) When it
    /// is reduced to 0, we move onto the next word.
    word: Word,

    /// The offset (measured in bits) of the current word.
    offset: usize,

    /// Underlying iterator over the words.
    iter: slice::Iter<'a, Word>,
}

impl<'a> RawBitIter<'a> {
    #[inline]
    pub(crate) fn new(words: &'a [Word]) -> Self {
        // Initialize `offset` to `0 - WORD_BITS`, so that the first iteration
        // will see `word == 0` and increase the offset to its starting value of 0.
        //
        // This avoids having to explicitly track whether the iterator has started.
        RawBitIter { word: 0, offset: (0usize).wrapping_sub(WORD_BITS), iter: words.iter() }
    }
}

impl<'a> Iterator for RawBitIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Keep looping until we find a non-empty word, or run out of words.
        loop {
            if self.word != 0 {
                // Get the position of the next set bit in the current word,
                // then clear the bit.
                let bit_pos = self.word.trailing_zeros() as usize;
                self.word ^= 1 << bit_pos;
                return Some(bit_pos + self.offset);
            }

            // Move onto the next word, or stop if there isn't one.
            self.word = self.iter.next().copied()?;
            // This needs to be a wrapping add so that the first iteration will
            // correctly overflow to a starting offset of 0.
            self.offset = self.offset.wrapping_add(WORD_BITS);
        }
    }
}

impl BitRelations<RawBitSet> for RawBitSet {
    #[inline]
    fn union(&mut self, other: &RawBitSet) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise_update_words(&mut self.words.as_mut(), &other.words.as_ref(), |a, b| a | b)
    }

    #[inline]
    fn subtract(&mut self, other: &RawBitSet) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise_update_words(&mut self.words.as_mut(), &other.words.as_ref(), |a, b| a & !b)
    }

    #[inline]
    fn intersect(&mut self, other: &RawBitSet) -> bool {
        assert_eq!(self.domain_size, other.domain_size);
        bitwise_update_words(&mut self.words.as_mut(), &other.words.as_ref(), |a, b| a & b)
    }
}

impl fmt::Debug for RawBitSet {
    #[inline]
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
    }
}
