use std::slice;

use crate::bit_set::{WORD_BITS, Word};

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
