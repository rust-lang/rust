use std::ops::Bound;
use std::slice;

use crate::bit_set::{WORD_BITS, Word, inclusive_start_end, max_bit};

#[inline]
pub(crate) fn contains_any(
    domain_size: usize,
    words: &[Word],
    range: (Bound<usize>, Bound<usize>),
) -> bool {
    let Some((start, end)) = inclusive_start_end(range, domain_size) else {
        return false;
    };

    let (start_word_index, start_mask) = word_index_and_mask(start);
    let (end_word_index, end_mask) = word_index_and_mask(end);

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
pub(crate) fn last_set_in(
    domain_size: usize,
    words: &[Word],
    range: (Bound<usize>, Bound<usize>),
) -> Option<usize> {
    let (start, end) = inclusive_start_end(range, domain_size)?;

    let (start_word_index, _) = word_index_and_mask(start);
    let (end_word_index, end_mask) = word_index_and_mask(end);

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
    if let Some(offset) = words[start_word_index..end_word_index].iter().rposition(|&w| w != 0) {
        let word_idx = start_word_index + offset;
        let start_word = words[word_idx];
        let pos = max_bit(start_word) + WORD_BITS * word_idx;
        if start <= pos {
            return Some(pos);
        }
    }

    None
}

#[inline(always)]
pub(crate) fn insert(domain_size: usize, words: &mut [Word], index: usize) -> bool {
    if index >= domain_size {
        index_not_in_domain("inserting", index, domain_size);
    }

    let (word_index, mask) = word_index_and_mask(index);
    modify_word(&mut words[word_index], |w| w | mask)
}

#[inline(always)]
pub(crate) fn remove(domain_size: usize, words: &mut [Word], index: usize) -> bool {
    if index >= domain_size {
        index_not_in_domain("removing", index, domain_size);
    }

    let (word_index, mask) = word_index_and_mask(index);
    modify_word(&mut words[word_index], |w| w & !mask)
}

#[cold]
#[inline(never)]
fn index_not_in_domain(verb: &'static str, index: usize, domain_size: usize) -> ! {
    panic!("{verb} at index {index} but domain size is {domain_size}");
}

/// Updates a `&mut Word` using the given function, and returns true if the value was changed.
#[inline(always)]
fn modify_word(word: &mut Word, modify_fn: impl Fn(Word) -> Word) -> bool {
    let old = *word;
    *word = modify_fn(old);
    old != *word
}

#[inline]
pub(crate) fn insert_range(
    domain_size: usize,
    words: &mut [Word],
    range: (Bound<usize>, Bound<usize>),
) {
    let Some((start, end)) = inclusive_start_end(range, domain_size) else {
        return;
    };

    let (start_word_index, start_mask) = word_index_and_mask(start);
    let (end_word_index, end_mask) = word_index_and_mask(end);

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
    #[inline(always)]
    pub(crate) fn new(words: &'a [Word]) -> Self {
        // Initialize `offset` to `0 - WORD_BITS`, so that the first iteration
        // will see `word == 0` and increase the offset to its starting value of 0.
        //
        // This avoids having to explicitly track whether the iterator has started.
        RawBitIter {
            word: 0,
            offset: const { (0usize).wrapping_sub(WORD_BITS) },
            iter: words.iter(),
        }
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

#[inline(always)]
pub(crate) fn word_index_and_mask(bit_index: usize) -> (usize, Word) {
    let word_index = bit_index / WORD_BITS;
    let mask = 1 << (bit_index % WORD_BITS);
    (word_index, mask)
}
