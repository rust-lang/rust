use rustc_macros::{Decodable, Encodable};

use std::fmt;
use std::iter;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::slice;

use crate::bit_set::{
    bit_relations_inherent_impls, inclusive_start_end, sequential_update, BitRelations, Chunk,
    ChunkedBitSet, GrowableBitSet, SparseBitSet, CHUNK_WORDS,
};
use crate::vec::Idx;

type Word = u8; // lmao
const WORD_BYTES: usize = 1;
const WORD_BITS: usize = WORD_BYTES * 8;

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
#[derive(Eq, PartialEq, Hash, Encodable, Decodable)]
pub struct BitSet<T> {
    inner: BitSetImpl,
    marker: PhantomData<T>,
}

const INLINE_BITSET_BYTES: usize = 30;
const INLINE_BITSET_BITS: usize = INLINE_BITSET_BYTES * 8;

#[derive(Eq, PartialEq, Hash, Encodable, Decodable, Clone)]
enum BitSetImpl {
    Inline { domain_size: u8, words: [Word; INLINE_BITSET_BYTES] },
    Heap { domain_size: usize, words: Box<[Word]> },
}

impl<T: Idx> BitSet<T> {
    /// Creates a new, empty bitset with a given `domain_size`.
    #[inline]
    pub fn new_empty(domain_size: usize) -> BitSet<T> {
        let inner = if domain_size <= INLINE_BITSET_BITS {
            BitSetImpl::Inline { domain_size: domain_size as u8, words: [0; INLINE_BITSET_BYTES] }
        } else {
            let num_words = num_words(domain_size);
            BitSetImpl::Heap { domain_size, words: vec![0; num_words].into_boxed_slice() }
        };
        let result = BitSet { inner, marker: PhantomData };
        result
    }

    /// Creates a new, filled bitset with a given `domain_size`.
    #[inline]
    pub fn new_filled(domain_size: usize) -> BitSet<T> {
        let num_words = num_words(domain_size);
        let inner = if domain_size <= INLINE_BITSET_BITS {
            let mut words = [0; INLINE_BITSET_BYTES];
            for i in 0..num_words {
                words[i] = !0;
            }
            BitSetImpl::Inline { domain_size: domain_size as u8, words }
        } else {
            BitSetImpl::Heap { domain_size, words: vec![!0; num_words].into_boxed_slice() }
        };
        let mut result = BitSet { inner, marker: PhantomData };
        result.clear_excess_bits();
        result
    }

    /// Gets the domain size.
    #[inline]
    pub fn domain_size(&self) -> usize {
        match &self.inner {
            BitSetImpl::Inline { domain_size, .. } => *domain_size as usize,
            BitSetImpl::Heap { domain_size, .. } => *domain_size,
        }
    }

    #[inline]
    fn words_mut(&mut self) -> &mut [Word] {
        let used_words = num_words(self.domain_size());
        match &mut self.inner {
            BitSetImpl::Inline { words, .. } => &mut words[..used_words],
            BitSetImpl::Heap { words, .. } => &mut words[..used_words],
        }
    }

    #[inline]
    fn raw_parts_mut(&mut self) -> (&mut [Word], usize) {
        match &mut self.inner {
            BitSetImpl::Inline { domain_size, words } => (&mut words[..], *domain_size as usize),
            BitSetImpl::Heap { domain_size, words } => (&mut words[..], *domain_size),
        }
    }

    #[inline]
    fn words(&self) -> &[Word] {
        let used_words = num_words(self.domain_size());
        match &self.inner {
            BitSetImpl::Inline { words, .. } => &words[..used_words],
            BitSetImpl::Heap { words, .. } => &words[..used_words],
        }
    }

    #[inline]
    fn raw_parts(&self) -> (&[Word], usize) {
        match &self.inner {
            BitSetImpl::Inline { domain_size, words } => (&words[..], *domain_size as usize),
            BitSetImpl::Heap { domain_size, words } => (&words[..], *domain_size),
        }
    }

    #[inline]
    pub fn words_wide(&self) -> impl Iterator<Item = u64> + '_ {
        self.words().chunks(8).map(|chunk| {
            let mut bytes = [0u8; 8];
            for (i, b) in chunk.iter().enumerate() {
                bytes[i] = *b;
            }
            u64::from_ne_bytes(bytes)
        })
    }

    /// Clear all elements.
    #[inline]
    pub fn clear(&mut self) {
        self.words_mut().fill(0);
    }

    /// Clear excess bits in the final word.
    fn clear_excess_bits(&mut self) {
        clear_excess_bits_in_final_word(self.domain_size(), self.words_mut());
    }

    /// Count the number of set bits in the set.
    pub fn count(&self) -> usize {
        self.words().iter().map(|e| e.count_ones() as usize).sum()
    }

    /// Returns `true` if `self` contains `elem`.
    #[inline]
    pub fn contains(&self, elem: T) -> bool {
        let (words, domain_size) = self.raw_parts();
        assert!(elem.index() < domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        (words[word_index] & mask) != 0
    }

    /// Is `self` is a (non-strict) superset of `other`?
    #[inline]
    pub fn superset(&self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size());
        self.words().iter().zip(other.words()).all(|(a, b)| (a & b) == *b)
    }

    /// Is the set empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.words().iter().all(|a| *a == 0)
    }

    /// Insert `elem`. Returns whether the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        let (words, domain_size) = self.raw_parts_mut();
        assert!(elem.index() < domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        let word_ref = &mut words[word_index];
        let word = *word_ref;
        let new_word = word | mask;
        *word_ref = new_word;
        new_word != word
    }

    #[inline]
    pub fn insert_range(&mut self, elems: impl RangeBounds<T>) {
        let Some((start, end)) = inclusive_start_end(elems, self.domain_size()) else {
            return;
        };

        let (start_word_index, start_mask) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);

        // Set all words in between start and end (exclusively of both).
        for word_index in (start_word_index + 1)..end_word_index {
            self.words_mut()[word_index] = !0;
        }

        if start_word_index != end_word_index {
            // Start and end are in different words, so we handle each in turn.
            //
            // We set all leading bits. This includes the start_mask bit.
            self.words_mut()[start_word_index] |= !(start_mask - 1);
            // And all trailing bits (i.e. from 0..=end) in the end word,
            // including the end.
            self.words_mut()[end_word_index] |= end_mask | (end_mask - 1);
        } else {
            self.words_mut()[start_word_index] |= end_mask | (end_mask - start_mask);
        }
    }

    /// Sets all bits to true.
    pub fn insert_all(&mut self) {
        self.words_mut().fill(!0);
        self.clear_excess_bits();
    }

    /// Returns `true` if the set has changed.
    #[inline]
    pub fn remove(&mut self, elem: T) -> bool {
        let (words, domain_size) = self.raw_parts_mut();
        assert!(elem.index() < domain_size);
        let (word_index, mask) = word_index_and_mask(elem);
        let word_ref = &mut words[word_index];
        let word = *word_ref;
        let new_word = word & !mask;
        *word_ref = new_word;
        new_word != word
    }

    /// Iterates over the indices of set bits in a sorted order.
    #[inline]
    pub fn iter(&self) -> BitIter<'_, T> {
        BitIter::new(self.words())
    }

    /*
    pub fn to_hybrid(&self) -> HybridBitSet<T> {
        // Note: we currently don't bother trying to make a Sparse set.
        HybridBitSet::Dense(self.to_owned())
    }
    */

    /// Set `self = self | other`. In contrast to `union` returns `true` if the set contains at
    /// least one bit that is not in `other` (i.e. `other` is not a superset of `self`).
    ///
    /// This is an optimization for union of a hybrid bitset.
    pub fn reverse_union_sparse(&mut self, sparse: &SparseBitSet<T>) -> bool {
        assert!(sparse.domain_size == self.domain_size());
        self.clear_excess_bits();

        let mut not_already = false;
        // Index of the current word not yet merged.
        let mut current_index = 0;
        // Mask of bits that came from the sparse set in the current word.
        let mut new_bit_mask = 0;
        for (word_index, mask) in sparse.iter().map(|x| word_index_and_mask(*x)) {
            // Next bit is in a word not inspected yet.
            if word_index > current_index {
                self.words_mut()[current_index] |= new_bit_mask;
                // Were there any bits in the old word that did not occur in the sparse set?
                not_already |= (self.words()[current_index] ^ new_bit_mask) != 0;
                // Check all words we skipped for any set bit.
                not_already |= self.words()[current_index + 1..word_index].iter().any(|&x| x != 0);
                // Update next word.
                current_index = word_index;
                // Reset bit mask, no bits have been merged yet.
                new_bit_mask = 0;
            }
            // Add bit and mark it as coming from the sparse set.
            // self.words[word_index] |= mask;
            new_bit_mask |= mask;
        }
        self.words_mut()[current_index] |= new_bit_mask;
        // Any bits in the last inspected word that were not in the sparse set?
        not_already |= (self.words()[current_index] ^ new_bit_mask) != 0;
        // Any bits in the tail? Note `clear_excess_bits` before.
        not_already |= self.words()[current_index + 1..].iter().any(|&x| x != 0);

        not_already
    }

    pub(crate) fn last_set_in(&self, range: impl RangeBounds<T>) -> Option<T> {
        let (start, end) = inclusive_start_end(range, self.domain_size())?;
        let (start_word_index, _) = word_index_and_mask(start);
        let (end_word_index, end_mask) = word_index_and_mask(end);

        let end_word = self.words()[end_word_index] & (end_mask | (end_mask - 1));
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
            self.words()[start_word_index..end_word_index].iter().rposition(|&w| w != 0)
        {
            let word_idx = start_word_index + offset;
            let start_word = self.words()[word_idx];
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
        assert_eq!(self.domain_size(), other.domain_size());
        bitwise(self.words_mut(), other.words(), |a, b| a | b)
    }

    fn subtract(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size());
        bitwise(self.words_mut(), other.words(), |a, b| a & !b)
    }

    fn intersect(&mut self, other: &BitSet<T>) -> bool {
        assert_eq!(self.domain_size(), other.domain_size());
        bitwise(self.words_mut(), other.words(), |a, b| a & b)
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

impl<T: Idx> From<GrowableBitSet<T>> for BitSet<T> {
    fn from(bit_set: GrowableBitSet<T>) -> Self {
        let mut new = BitSet::new_empty(bit_set.domain_size);
        for bit in bit_set.iter() {
            new.insert(bit);
        }
        new
    }
}

impl<T> Clone for BitSet<T> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), marker: PhantomData }
    }

    /*
    fn clone_from(&mut self, from: &Self) {
        self.domain_size = from.domain_size;
        self.words.clone_from(from.words());
    }
    */
}

impl<T: Idx> fmt::Debug for BitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
    }
}

/*
impl<T: Idx> ToString for BitSet<T> {
    fn to_string(&self) -> String {
        let mut result = String::new();
        let mut sep = '[';

        // Note: this is a little endian printout of bytes.

        // i tracks how many bits we have printed so far.
        let mut i = 0;
        for word in self.words() {
            let mut word = *word;
            for _ in 0..WORD_BYTES {
                // for each byte in `word`:
                let remain = self.domain_size() - i;
                // If less than a byte remains, then mask just that many bits.
                let mask = if remain <= 8 { (1 << remain) - 1 } else { 0xFF };
                // assert!(mask <= 0xFF); // FIXME
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
*/

impl<T: Idx> BitRelations<ChunkedBitSet<T>> for BitSet<T> {
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
            let mut words = &mut self.words_mut()[i * CHUNK_WORDS..];
            if words.len() > CHUNK_WORDS {
                words = &mut words[..CHUNK_WORDS];
            }
            match chunk {
                Chunk::Zeros(..) => {
                    for word in words {
                        if *word != 0 {
                            changed = true;
                            *word = 0;
                        }
                    }
                }
                Chunk::Ones(..) => (),
                Chunk::Mixed(_, _, _data) => {
                    unimplemented!("Stop being lazy");
                    /*
                    for (i, word) in words.iter_mut().enumerate() {
                        let new_val = *word & data[i];
                        if new_val != *word {
                            changed = true;
                            *word = new_val;
                        }
                    }
                    */
                }
            }
        }
        changed
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

fn clear_excess_bits_in_final_word(domain_size: usize, words: &mut [Word]) {
    let num_bits_in_final_word = domain_size % WORD_BITS;
    if num_bits_in_final_word > 0 {
        let mask = (1 << num_bits_in_final_word) - 1;
        words[words.len() - 1] &= mask;
    }
}

#[inline]
fn max_bit(word: Word) -> usize {
    WORD_BITS - 1 - word.leading_zeros() as usize
}
