use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, handle_alloc_error};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Range, RangeInclusive};
use std::ptr::NonNull;
use std::{fmt, slice};

use super::{
    BitRelations, CHUNK_WORDS, Chunk, ChunkedBitSet, WORD_BITS, Word, max_bit, word_index_and_mask,
};
use crate::{Idx, IndexVec};
/// A fixed-size bitset type with a dense representation, using only one [`Word`] on the stack.
///
/// This bit set occupies only a single [`Word`] of stack space. It can represent a domain size
/// of up to `[WORD_BITS] - 1` directly inline. If the domain size exceeds this limit, it instead
/// becomes a pointer to a sequence of [`Word`]s on the heap. This makes it very efficient for
/// domain sizes smaller than `[WORD_BITS]`.
///
/// Additionally, if the set does not fit in one [`Word`], there is a special inline
/// variant for the empty set. In this case, the domain size is stored inline along with a few
/// bits indicating that the set is empty. Allocation is deferred until needed, such as on
/// the first insert or remove operation. This avoids the need to wrap a lazily initialised bit set
/// in a [`OnceCell`] or an [`Option`]—you can simply create an empty set and populate it if needed.
///
/// Note 1: Since this bitset is dense, if your domain is large and/or relatively homogeneous (e.g.
/// long runs of set or unset bits), it may be more efficient to use a [MixedBitSet] or an
/// [IntervalSet](crate::interval::IntervalSet), which are better suited for sparse or highly
/// compressible domains.
///
/// Note 2: Use [`GrowableBitSet`] if you require support for resizing after creation.
///
/// `T` is an index type—typically a newtyped `usize` wrapper, but it may also simply be `usize`.
///
/// Any operation involving an element may panic if the element is equal to or greater than the
/// domain size. Operations involving two bitsets may panic if their domain sizes differ. Panicking
/// is not garranteed though as we store the domain size rounded up to the next multiple of
/// [`WORD_BITS`].
#[repr(C)]
pub union ThinBitSet<T> {
    /// The bit set fits in a single [`Word`] stored inline on the stack.
    ///
    /// The most significant bit is set to 1 to distinguish this from the other variants. You
    /// must never change that "tag bit" after the bit set has been created.
    ///
    /// The remaining bits makes up the bit set. The exact domain size is not stored.
    inline: Word,

    /// The bit set doesn't fit in a single word, but is empty and not yet allocated.
    ///
    /// The first (most significant) two bits are set to `[0, 1]` to distinguish this variant
    /// from others. This tag is stored in [`Self::EMPTY_UNALLOCATED_TAG_BITS`]. The remaining bits
    /// hold the domain size **in words** of the set, which is needed if the set is eventually
    /// allocated.
    ///
    /// Note that because the domain size is stored in words, not in bits, there is plenty of room
    /// for the two tag bits.
    empty_unallocated: usize,

    /// The bit set is stored on the heap.
    ///
    /// The two most significant bits are set to zero if this field is active.
    on_heap: ManuallyDrop<BitSetOnHeap>,

    /// This variant will never be created.
    marker: PhantomData<T>,
}

impl<T> ThinBitSet<T> {
    /// The maximum domain size that could be stored inlined on the stack.
    pub const MAX_INLINE_DOMAIN_SIZE: usize = WORD_BITS - 1;

    /// A [`Word`] with the most significant bit set. That is the tag bit telling that the set is
    /// inlined.
    const IS_INLINE_TAG_BIT: Word = 0x1 << (WORD_BITS - 1);

    /// The tag for the `empty_unallocated` variant. The two most significant bits are
    /// `[0, 1]`.
    const EMPTY_UNALLOCATED_TAG_BITS: usize = 0b01 << (WORD_BITS - 2);

    /// Create a new empty bit set with a given domain_size.
    ///
    /// If `domain_size` is <= [`Self::MAX_INLINE_DOMAIN_SIZE`], then it is stored inline on the stack,
    /// otherwise it is stored on the heap.
    #[inline]
    pub fn new_empty(domain_size: usize) -> Self {
        if domain_size <= Self::MAX_INLINE_DOMAIN_SIZE {
            // The first bit is set to indicate the union variant.
            Self { inline: Self::IS_INLINE_TAG_BIT }
        } else {
            let num_words = domain_size.div_ceil(WORD_BITS);
            debug_assert!(num_words.leading_zeros() >= 2);
            Self { empty_unallocated: Self::EMPTY_UNALLOCATED_TAG_BITS | num_words }
        }
    }

    /// Create a new filled bit set.
    #[inline]
    pub fn new_filled(domain_size: usize) -> Self {
        if domain_size <= Self::MAX_INLINE_DOMAIN_SIZE {
            if domain_size == 0 {
                // FIXME: Remove this.
                return Self::new_empty(domain_size);
            }
            Self {
                inline: Word::MAX >> (WORD_BITS - domain_size) % WORD_BITS
                    | Self::IS_INLINE_TAG_BIT,
            }
        } else {
            let num_words = domain_size.div_ceil(WORD_BITS);
            let mut on_heap = BitSetOnHeap::new_empty(num_words);
            let words = on_heap.as_mut_slice();
            for word in words.iter_mut() {
                *word = Word::MAX;
            }
            // Remove excessive bits on the last word.
            *words.last_mut().unwrap() >>= WORD_BITS - domain_size % WORD_BITS;
            Self { on_heap: ManuallyDrop::new(on_heap) }
        }
    }

    /// Check if `self` is inlined.
    // If this function returns `true`, it is safe to assume `self.inline`. Else, it is safe to
    // assume `self.empty_unallocated`, or `self.on_heap`.
    #[inline(always)]
    pub fn is_inline(&self) -> bool {
        // We check if the first bit is set. If so, it is inlined, otherwise it is on the heap.
        (unsafe { self.inline } & Self::IS_INLINE_TAG_BIT) != 0
    }

    /// Check if `self` has a too large domain to be stored inline, is empty, and is not yet
    /// allocated.
    // If this function returns `true`, it is safe to assume `self.empty_unallocated`. Else, it is
    // safe to assume `self.inline`, or `self.on_heap`.
    #[inline(always)]
    pub const fn is_empty_unallocated(&self) -> bool {
        (unsafe { self.empty_unallocated }) >> usize::BITS as u32 - 2
            == Self::EMPTY_UNALLOCATED_TAG_BITS >> usize::BITS as u32 - 2
    }

    /// Check if `self` is allocated on the heap and return a reference to it in that case.
    fn on_heap(&self) -> Option<&BitSetOnHeap> {
        let self_word = unsafe { self.inline };
        // Check if the two most significant bits are 0.
        if self_word & Word::MAX >> 2 == self_word { Some(unsafe { &self.on_heap }) } else { None }
    }

    /// Check if `self` is allocated on the heap and return a mutable reference to it in that case.
    fn on_heap_mut(&mut self) -> Option<&mut ManuallyDrop<BitSetOnHeap>> {
        let self_word = unsafe { self.inline };
        // Check if the two most significant bits are 0.
        if self_word & Word::MAX >> 2 == self_word {
            Some(unsafe { &mut self.on_heap })
        } else {
            None
        }
    }

    /// If `self` is `empty_unallocated`, allocate it, otherwise return `self.on_heap_mut()`.
    fn on_heap_get_or_alloc(&mut self) -> &mut BitSetOnHeap {
        if self.is_empty_unallocated() {
            let num_words = unsafe { self.empty_unallocated } ^ Self::EMPTY_UNALLOCATED_TAG_BITS;
            *self = Self { on_heap: ManuallyDrop::new(BitSetOnHeap::new_empty(num_words)) };
            unsafe { &mut self.on_heap }
        } else {
            self.on_heap_mut().unwrap()
        }
    }

    /// Checks if the bit set is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        if self.is_inline() {
            let x = unsafe { self.inline };
            x == Self::IS_INLINE_TAG_BIT
        } else if self.is_empty_unallocated() {
            true
        } else {
            self.on_heap().unwrap().is_empty()
        }
    }

    /// Clear the set.
    #[inline(always)]
    pub fn clear(&mut self) {
        if self.is_inline() {
            self.inline = Self::IS_INLINE_TAG_BIT
        } else if let Some(on_heap) = self.on_heap_mut() {
            for word in on_heap.as_mut_slice() {
                *word = 0x0;
            }
        }
    }

    /// Checks if `self` is a (non-strict) superset of `other`.
    ///
    /// May panic if `self` and other have different sizes.
    #[inline(always)]
    pub fn superset(&self, other: &Self) -> bool {
        // Function to check that a usize is a superset of another.
        let word_is_superset = |x: Word, other: Word| (!x & other) == 0;

        if self.is_inline() {
            let x = unsafe { self.inline };
            assert!(other.is_inline(), "bit sets has different domain sizes");
            let y = unsafe { other.inline };
            word_is_superset(x, y)
        } else if other.is_empty_unallocated() {
            true
        } else {
            let other_on_heap = other.on_heap().unwrap();
            if self.is_empty_unallocated() {
                other_on_heap.is_empty()
            } else {
                let on_heap = self.on_heap().unwrap();
                let self_slice = on_heap.as_slice();
                let other_slice = other_on_heap.as_slice();
                debug_assert_eq!(
                    self_slice.len(),
                    other_slice.len(),
                    "bit sets have different domain sizes"
                );
                self_slice.iter().zip(other_slice).all(|(&x, &y)| (!x & y) == 0)
            }
        }
    }

    /// Count the number of set bits in the set.
    #[inline(always)]
    pub fn count(&self) -> usize {
        if self.is_inline() {
            let x = unsafe { self.inline };
            x.count_ones() as usize - 1
        } else if self.is_empty_unallocated() {
            0
        } else {
            self.on_heap().unwrap().as_slice().iter().map(|w| w.count_ones() as usize).sum()
        }
    }
    /// Common function for union/intersection-like operations.
    ///
    /// This function takes two bit sets—one mutably, one immutably. Neither must be the
    /// `empty_unallocated` variant. It asserts that they have the same `domain_size`, then applies a function to
    /// each pair of words, effectively performing a zip-like operation.
    /// It checks whether `self` has changed; if so, it returns `true`, otherwise `false`.
    ///
    /// ## Safety
    ///
    /// - Neither set must be `self.empty_unallocated`.
    /// - If the sets are inlined, this will leave the tag bit set to 1. You must not modify it—doing so
    ///   results in undefined behaviour. This may be inconvenient for operations such as subtraction;
    ///   in such cases, use `binary_operation_safe` instead.
    #[inline(always)]
    unsafe fn binary_operation(&mut self, other: &Self, op: impl Fn(&mut Word, Word)) -> bool {
        debug_assert!(!self.is_empty_unallocated());
        debug_assert!(!other.is_empty_unallocated());

        // Apply `op` and return if the word changed.
        let apply_and_check_change = |x: &mut Word, y: Word| -> bool {
            let old = *x;
            op(x, y);
            *x != old
        };

        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            assert!(other.is_inline(), "bit sets has different domain sizes");
            let y = unsafe { other.inline };
            apply_and_check_change(x, y)
        } else {
            let self_on_heap = unsafe { &mut self.on_heap };
            assert!(!other.is_inline(), "bit sets has different domain sizes");
            let other_on_heap = unsafe { &other.on_heap };
            let self_slice = self_on_heap.as_mut_slice();
            let other_slice = other_on_heap.as_slice();
            assert_eq!(self_slice.len(), other_slice.len(), "bit sets have different domain sizes");
            let mut has_changed = false;
            for (x, y) in self_slice.iter_mut().zip(other_slice) {
                has_changed |= apply_and_check_change(x, *y);
            }
            has_changed
        }
    }

    /// Similar to [`Self::binary_operation`], but restores the tag bit if it has changed.
    ///
    /// Note that the tag bit will still be set in the call to `op`, but there is no danger in
    /// changing it as it will be restored afterwords.
    ///
    /// ## Safety
    ///
    /// Neither set must be `self.empty_unallocated`.
    #[inline(always)]
    unsafe fn binary_operation_safe(&mut self, other: &Self, op: impl Fn(&mut Word, Word)) -> bool {
        debug_assert!(!self.is_empty_unallocated());
        debug_assert!(!other.is_empty_unallocated());

        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            assert!(other.is_inline(), "bit sets has different domain sizes");
            let y = unsafe { other.inline };

            let old = *x;
            op(x, y);
            *x |= Self::IS_INLINE_TAG_BIT;
            old != *x
        } else {
            let self_on_heap = unsafe { &mut self.on_heap };
            assert!(!other.is_inline(), "bit sets has different domain sizes");
            let other_on_heap = unsafe { &other.on_heap };
            let self_slice = self_on_heap.as_mut_slice();
            let other_slice = other_on_heap.as_slice();
            assert_eq!(self_slice.len(), other_slice.len(), "bit sets have different domain sizes");
            let mut has_changed = false;
            for (x, y) in self_slice.iter_mut().zip(other_slice) {
                let old = *x;
                op(x, *y);
                has_changed |= old != *x;
            }
            has_changed
        }
    }

    super::bit_relations_inherent_impls! {}
}

impl<T> BitRelations<ThinBitSet<T>> for ThinBitSet<T> {
    #[inline(always)]
    fn union(&mut self, other: &Self) -> bool {
        if self.is_empty_unallocated() {
            debug_assert!(!other.is_inline());
            *self = other.clone();
            !self.is_empty()
        } else if other.is_empty_unallocated() {
            false
        } else {
            // SAFETY: The union operation does not remove any bit set to 1, so the tag bit is
            // unaffected.
            unsafe { self.binary_operation(other, |x, y| *x |= y) }
        }
    }

    #[inline(always)]
    fn intersect(&mut self, other: &Self) -> bool {
        if self.is_empty_unallocated() {
            false
        } else if other.is_empty_unallocated() {
            debug_assert!(!self.is_inline());
            let was_empty = self.is_empty();
            self.clear();
            !was_empty
        } else {
            // SAFETY: Since the tag bit is set in both `self` and `other`, the intersection won't
            // remove it.
            unsafe { self.binary_operation(other, |x, y| *x &= y) }
        }
    }

    #[inline(always)]
    fn subtract(&mut self, other: &Self) -> bool {
        if self.is_empty_unallocated() || other.is_empty_unallocated() {
            false
        } else {
            unsafe { self.binary_operation_safe(other, |x, y| *x &= !y) }
        }
    }
}

impl<T: Idx> ThinBitSet<T> {
    /// Checks if the bit set contains `elem`.
    #[inline(always)]
    pub fn contains(&self, elem: T) -> bool {
        // Check if the `i`th bit is set in a word.
        let contains_bit = |word: Word, bit_idx: u32| {
            let mask = 0x01 << bit_idx;
            (word & mask) != 0
        };

        let idx = elem.index();
        if self.is_inline() {
            let x = unsafe { self.inline };
            debug_assert!(idx < Self::MAX_INLINE_DOMAIN_SIZE, "index too large: {idx}");
            contains_bit(x, idx as u32)
        } else if let Some(on_heap) = self.on_heap() {
            let word_idx = idx / WORD_BITS;
            let bit_idx = (idx % WORD_BITS) as u32;
            let word = on_heap.as_slice()[word_idx];
            contains_bit(word, bit_idx)
        } else {
            debug_assert!(self.is_empty_unallocated());
            false
        }
    }

    /// Insert `elem`. Returns `true` if the set has changed.
    #[inline(always)]
    pub fn insert(&mut self, elem: T) -> bool {
        // Insert the `i`th bit in a word and return `true` if it changed.
        let insert_bit = |word: &mut Word, bit_idx: u32| {
            let mask = 0x01 << bit_idx;
            let old = *word;
            *word |= mask;
            *word != old
        };

        let idx = elem.index();
        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            debug_assert!(idx < Self::MAX_INLINE_DOMAIN_SIZE, "index too large: {idx}");
            insert_bit(x, idx as u32)
        } else {
            let words = self.on_heap_get_or_alloc().as_mut_slice();

            let word_idx = idx / WORD_BITS;
            let bit_idx = (idx % WORD_BITS) as u32;
            let word = &mut words[word_idx];
            insert_bit(word, bit_idx)
        }
    }

    /// Remove `elem`. Returns `true` if the set has changed.
    #[inline(always)]
    pub fn remove(&mut self, elem: T) -> bool {
        // Remove the `i`th bit in a word and return `true` if it changed.
        let remove_bit = |word: &mut Word, bit_idx: u32| {
            let mask = !(0x01 << bit_idx);
            let old = *word;
            *word &= mask;
            *word != old
        };

        let idx = elem.index();
        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            debug_assert!(idx < Self::MAX_INLINE_DOMAIN_SIZE, "index too large: {idx}");
            remove_bit(x, idx as u32)
        } else if let Some(on_heap) = self.on_heap_mut() {
            let word_idx = idx / WORD_BITS;
            let bit_idx = (idx % WORD_BITS) as u32;
            let word = &mut on_heap.as_mut_slice()[word_idx];
            remove_bit(word, bit_idx)
        } else {
            debug_assert!(self.is_empty_unallocated());
            // Nothing to be removed.
            false
        }
    }

    /// Returns an iterator over all elements in this set.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = T> + use<'_, T> {
        if self.is_inline() {
            let x = unsafe { self.inline };
            // Remove the tag bit.
            let without_tag_bit = x ^ Self::IS_INLINE_TAG_BIT;
            BitIter::from_single_word(without_tag_bit)
        } else if let Some(on_heap) = self.on_heap() {
            BitIter::from_slice(on_heap.as_slice())
        } else {
            debug_assert!(self.is_empty_unallocated());
            BitIter::from_single_word(0)
        }
    }

    #[inline]
    pub fn insert_range(&mut self, range: Range<T>) {
        if let Some(end) = range.end.index().checked_sub(1) {
            self.insert_range_inclusive(RangeInclusive::new(range.start, Idx::new(end)));
        }
    }

    #[inline(always)]
    pub fn insert_range_inclusive(&mut self, range: RangeInclusive<T>) {
        let start = range.start().index();
        let end = range.end().index();

        if start > end {
            return;
        }

        if self.is_inline() {
            debug_assert!(end < Self::MAX_INLINE_DOMAIN_SIZE);
            let mask = (1 << end) | ((1 << end) - (1 << start));
            unsafe { self.inline |= mask };
        } else {
            let words = self.on_heap_get_or_alloc().as_mut_slice();

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
    }

    #[inline(always)]
    pub fn last_set_in(&self, range: RangeInclusive<T>) -> Option<T> {
        let start = range.start().index();
        let end = range.end().index();

        if start > end {
            return None;
        }

        if self.is_inline() {
            debug_assert!(end < Self::MAX_INLINE_DOMAIN_SIZE);
            let mut word = unsafe { self.inline } ^ Self::IS_INLINE_TAG_BIT;
            let end_bit = 1 << end;
            // Set all bits mor significant than `end_bit` to zero.
            word &= end_bit | (end_bit - 1);
            if word != 0 {
                let pos = max_bit(word);
                if start <= pos { Some(T::new(pos)) } else { None }
            } else {
                None
            }
        } else if let Some(on_heap) = self.on_heap() {
            let words = on_heap.as_slice();

            let (start_word_index, _) = word_index_and_mask(start);
            let (end_word_index, end_mask) = word_index_and_mask(end);

            let end_word = words[end_word_index] & (end_mask | (end_mask - 1));
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
                words[start_word_index..end_word_index].iter().rposition(|&w| w != 0)
            {
                let word_idx = start_word_index + offset;
                let start_word = words[word_idx];
                let pos = max_bit(start_word) + WORD_BITS * word_idx;
                if start <= pos { Some(T::new(pos)) } else { None }
            } else {
                None
            }
        } else {
            debug_assert!(self.is_empty_unallocated());
            None
        }
    }
}

impl<T: Idx> BitRelations<ChunkedBitSet<T>> for ThinBitSet<T> {
    fn union(&mut self, other: &ChunkedBitSet<T>) -> bool {
        other.iter().fold(false, |changed, elem| self.insert(elem) || changed)
    }

    fn subtract(&mut self, _other: &ChunkedBitSet<T>) -> bool {
        unimplemented!("implement if/when necessary");
    }

    fn intersect(&mut self, other: &ChunkedBitSet<T>) -> bool {
        if self.is_inline() {
            assert!(other.domain_size <= Self::MAX_INLINE_DOMAIN_SIZE);
            if other.domain_size == 0 {
                return false;
            }

            let word = unsafe { &mut self.inline };
            let old_word = *word;
            match &other.chunks[0] {
                Chunk::Zeros(d) => {
                    debug_assert_eq!(usize::from(*d), other.domain_size);
                    let mask = Word::MAX << other.domain_size();
                    *word &= mask;
                }
                Chunk::Ones(_) => (),
                Chunk::Mixed(d, _, words) => {
                    debug_assert_eq!(usize::from(*d), other.domain_size);
                    *word &= words[0] | Self::IS_INLINE_TAG_BIT;
                }
            }
            *word != old_word
        } else if let Some(on_heap) = self.on_heap_mut() {
            let all_words = on_heap.as_mut_slice();

            let mut changed = false;
            for (i, chunk) in other.chunks.iter().enumerate() {
                let mut words = &mut all_words[i * CHUNK_WORDS..];
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
                    Chunk::Mixed(_, _, data) => {
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
        } else {
            debug_assert!(self.is_empty_unallocated());
            false
        }
    }
}

impl<T> Clone for ThinBitSet<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        if self.is_inline() {
            let inline = unsafe { self.inline };
            Self { inline }
        } else if self.is_empty_unallocated() {
            let empty_unallocated = unsafe { self.empty_unallocated };
            Self { empty_unallocated }
        } else {
            let old_on_heap = unsafe { &self.on_heap };
            let on_heap = old_on_heap.clone();
            Self { on_heap }
        }
    }
}

impl<T> Drop for ThinBitSet<T> {
    #[inline(always)]
    fn drop(&mut self) {
        // Deallocate if `self` is not inlined.
        if let Some(on_heap) = self.on_heap_mut() {
            unsafe {
                ManuallyDrop::drop(on_heap);
            }
        }
    }
}

/// A pointer to a dense bit set stored on the heap.
///
/// This struct is a `usize`, with its two most significant bits always set to 0. If the value is
/// shifted left by 2 bits, it yields a pointer to a sequence of words on the heap. The first word
/// in this sequence represents the length—it indicates how many words follow. These subsequent
/// words make up the actual bit set.
///
/// For example, suppose the bit set should support a domain size of 240 bits. We first determine
/// how many words are needed to store 240 bits—that’s 4 words, assuming `[WORD_BITS] == 64`.
/// The pointer in this struct then points to a sequence of five words allocated on the heap. The
/// first word has the value 4 (the length), and the remaining four words comprise the bit set.
#[repr(transparent)]
struct BitSetOnHeap(usize);

impl BitSetOnHeap {
    fn new_empty(len: usize) -> Self {
        // The first word is used to store the total number of words. The rest of the words
        // store the bits.
        let num_words = len + 1;

        let layout = Layout::array::<Word>(num_words).expect("Bit set too large");
        // SAFETY: `num_words` is always at least `1` so we never allocate zero size.
        let ptr = unsafe { alloc_zeroed(layout).cast::<Word>() };
        let Some(ptr) = NonNull::<Word>::new(ptr) else {
            handle_alloc_error(layout);
        };

        // Store the length in the first word.
        unsafe { ptr.write(len as Word) };

        // Convert `ptr` to a `usize` and shift it two bits to the right.
        BitSetOnHeap((ptr.as_ptr() as usize) >> 2)
    }

    /// Get a slice with all bits in this bit set.
    ///
    /// Note that the number of bits in the set is rounded up to the next power of `Usize::BITS`. So
    /// if the user requested a domain_size of 216 bits, a slice with 4 words will be returned on a
    /// 64-bit machine.
    #[inline]
    fn as_slice(&self) -> &[Word] {
        let ptr = (self.0 << 2) as *const Word;
        let len = unsafe { ptr.read() } as usize;
        // The slice starts at the second word.
        unsafe { slice::from_raw_parts(ptr.add(1), len) }
    }

    /// Get a mutable slice with all bits in this bit set.
    ///
    /// Note that the number of bits in the set is rounded up to the next power of `Usize::BITS`. So
    /// if the user requested a domain_size of 216 bits, a slice with 4 words will be returned on a
    /// 64-bit machine.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Word] {
        let ptr = (self.0 << 2) as *mut Word;
        let len = unsafe { ptr.read() } as usize;
        // The slice starts at the second word.
        unsafe { slice::from_raw_parts_mut(ptr.add(1), len) }
    }

    /// Check if the set is empty.
    fn is_empty(&self) -> bool {
        self.as_slice().iter().all(|&x| x == 0)
    }
}

impl Clone for BitSetOnHeap {
    fn clone(&self) -> Self {
        let ptr = (self.0 << 2) as *const Word;
        let len = unsafe { ptr.read() } as usize;
        let num_words = len + 1;

        let layout = Layout::array::<usize>(num_words).expect("Bit set too large");
        // SAFETY: `num_words` is always at least `1` so we never allocate zero size.
        let new_ptr = unsafe { alloc(layout).cast::<Word>() };
        let Some(new_ptr) = NonNull::<Word>::new(new_ptr) else {
            handle_alloc_error(layout);
        };

        unsafe { ptr.copy_to_nonoverlapping(new_ptr.as_ptr(), num_words) };

        BitSetOnHeap((new_ptr.as_ptr() as usize) >> 2)
    }
}

impl Drop for BitSetOnHeap {
    fn drop(&mut self) {
        let ptr = (self.0 << 2) as *mut Word;

        // SAFETY: The first word stores the number of words for the bit set. We have to add 1
        // because the first word storing the length is allocated as well.
        let num_words = unsafe { ptr.read() } as usize + 1;
        let layout = Layout::array::<Word>(num_words).expect("Bit set too large");
        // SAFETY: We know that `on_heap` has been allocated with the same layout. See the
        // `new` method for reference.
        unsafe { dealloc(ptr.cast::<u8>(), layout) };
    }
}

struct BitIter<'a, T: Idx> {
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
    fn from_slice(words: &'a [Word]) -> Self {
        // We initialize `word` and `offset` to degenerate values. On the first
        // call to `next()` we will fall through to getting the first word from
        // `iter`, which sets `word` to the first word (if there is one) and
        // `offset` to 0. Doing it this way saves us from having to maintain
        // additional state about whether we have started.
        Self {
            word: 0,
            offset: usize::MAX - (WORD_BITS - 1),
            iter: words.iter(),
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn from_single_word(word: Word) -> Self {
        Self { word, offset: 0, iter: [].iter(), marker: PhantomData }
    }
}

impl<'a, T: Idx> Iterator for BitIter<'a, T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        loop {
            if self.word != 0 {
                // Get the position of the next set bit in the current word,
                // then clear the bit.
                let bit_pos = self.word.trailing_zeros() as usize;
                self.word ^= 0x01 << bit_pos;
                return Some(T::new(bit_pos + self.offset));
            }

            // Move onto the next word. `wrapping_add()` is needed to handle
            // the degenerate initial value given to `offset` in `new()`.
            self.word = *self.iter.next()?;
            self.offset = self.offset.wrapping_add(WORD_BITS);
        }
    }
}

impl<T: Idx> fmt::Debug for ThinBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
    }
}

/// A fixed-column-size, variable-row-size 2D bit matrix with a moderately
/// sparse representation.
///
/// Initially, every row has no explicit representation. If any bit within a row
/// is set, the entire row is instantiated as `Some(<ThinBitSet>)`.
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
    rows: IndexVec<R, Option<ThinBitSet<C>>>,
}

impl<R: Idx, C: Idx> SparseBitMatrix<R, C> {
    /// Creates a new empty sparse bit matrix with no rows or columns.
    pub fn new(num_columns: usize) -> Self {
        Self { num_columns, rows: IndexVec::new() }
    }

    pub fn ensure_row(&mut self, row: R) -> &mut ThinBitSet<C> {
        // Instantiate any missing rows up to and including row `row` with an empty `ThinBitSet`.
        // Then replace row `row` with a full `ThinBitSet` if necessary.
        self.rows.get_or_insert_with(row, || ThinBitSet::new_empty(self.num_columns))
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
        self.row(row).is_some_and(|r| r.contains(column))
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

    pub fn rows(&self) -> impl Iterator<Item = R> {
        self.rows.indices()
    }

    /// Iterates through all the columns set to true in a given row of
    /// the matrix.
    pub fn iter(&self, row: R) -> impl Iterator<Item = C> + '_ {
        self.row(row).into_iter().flat_map(|r| r.iter())
    }

    pub fn row(&self, row: R) -> Option<&ThinBitSet<C>> {
        self.rows.get(row)?.as_ref()
    }

    /// Intersects `row` with `set`. `set` can be either `ThinBitSet` or
    /// `ChunkedBitSet`. Has no effect if `row` does not exist.
    ///
    /// Returns true if the row was changed.
    pub fn intersect_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        ThinBitSet<C>: BitRelations<Set>,
    {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.intersect(set),
            _ => false,
        }
    }

    /// Subtracts `set` from `row`. `set` can be either `ThinBitSet` or
    /// `ChunkedBitSet`. Has no effect if `row` does not exist.
    ///
    /// Returns true if the row was changed.
    pub fn subtract_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        ThinBitSet<C>: BitRelations<Set>,
    {
        match self.rows.get_mut(row) {
            Some(Some(row)) => row.subtract(set),
            _ => false,
        }
    }

    /// Unions `row` with `set`. `set` can be either `ThinBitSet` or
    /// `ChunkedBitSet`.
    ///
    /// Returns true if the row was changed.
    pub fn union_row<Set>(&mut self, row: R, set: &Set) -> bool
    where
        ThinBitSet<C>: BitRelations<Set>,
    {
        self.ensure_row(row).union(set)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::bit_set::DenseBitSet;

    const TEST_ITERATIONS: u32 = 512;

    /// A very simple pseudo random generator using linear xorshift.
    ///
    /// [See Wikipedia](https://en.wikipedia.org/wiki/Xorshift). This has 64-bit state and a period
    /// of `2^64 - 1`.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed)
        }

        fn next(&mut self) -> usize {
            self.0 ^= self.0 << 7;
            self.0 ^= self.0 >> 9;
            self.0 as usize
        }

        fn next_bool(&mut self) -> bool {
            self.next() % 2 == 0
        }

        /// Sample a range, a subset of `0..=max`.
        ///
        /// The purpose of this method is to make edge cases such as `0..=max` more common.
        fn sample_range(&mut self, max: usize) -> RangeInclusive<usize> {
            let start = match self.next() % 3 {
                0 => 0,
                1 => max,
                2 => self.next() % (max + 1),
                _ => unreachable!(),
            };
            let end = match self.next() % 3 {
                0 => 0,
                1 => max,
                2 => self.next() % (max + 1),
                _ => unreachable!(),
            };
            RangeInclusive::new(start, end)
        }
    }

    fn test_with_domain_size(domain_size: usize) {
        let mut set_1 = ThinBitSet::<usize>::new_empty(domain_size);
        let mut set_1_reference = DenseBitSet::<usize>::new_empty(domain_size);
        let mut set_2 = ThinBitSet::<usize>::new_empty(domain_size);
        let mut set_2_reference = DenseBitSet::<usize>::new_empty(domain_size);

        let mut rng = Rng::new(42);

        for _ in 0..TEST_ITERATIONS {
            // Make a random operation.
            match rng.next() % 100 {
                0..20 => {
                    // Insert in one of the sets.
                    if domain_size == 0 {
                        continue;
                    }
                    let elem = rng.next() % domain_size;
                    // Choose set to insert into.
                    if rng.next_bool() {
                        assert_eq!(set_1.insert(elem), set_1_reference.insert(elem));
                    } else {
                        assert_eq!(set_2.insert(elem), set_2_reference.insert(elem));
                    }
                }
                20..50 => {
                    // Insert a range in one of the sets.
                    if domain_size == 0 {
                        continue;
                    }

                    let range = rng.sample_range(domain_size - 1);
                    // Choose set to insert into.
                    if rng.next_bool() {
                        set_1.insert_range_inclusive(range.clone());
                        set_1_reference.insert_range(range);
                    } else {
                        set_2.insert_range_inclusive(range.clone());
                        set_2_reference.insert_range(range);
                    }
                }
                50..70 => {
                    // Remove from one of the sets.
                    if domain_size == 0 {
                        continue;
                    }
                    let elem = rng.next() % domain_size;
                    // Choose set to remove into.
                    if rng.next_bool() {
                        assert_eq!(set_1.remove(elem), set_1_reference.remove(elem));
                    } else {
                        assert_eq!(set_2.remove(elem), set_2_reference.remove(elem));
                    }
                }
                70..79 => {
                    // Union
                    assert_eq!(set_1.union(&set_2), set_1_reference.union(&set_2_reference));
                }
                79..88 => {
                    // Intersection
                    assert_eq!(
                        set_1.intersect(&set_2),
                        set_1_reference.intersect(&set_2_reference)
                    );
                }
                88..97 => {
                    // Subtraction
                    assert_eq!(set_1.subtract(&set_2), set_1_reference.subtract(&set_2_reference));
                }
                97..99 => {
                    // Clear
                    if rng.next_bool() {
                        set_1.clear();
                        set_1_reference.clear();
                    } else {
                        set_2.clear();
                        set_2_reference.clear();
                    }
                }
                99..100 => {
                    // Fill.
                    if rng.next_bool() {
                        set_1 = ThinBitSet::new_filled(domain_size);
                        set_1_reference = DenseBitSet::new_filled(domain_size);
                    } else {
                        set_2 = ThinBitSet::new_filled(domain_size);
                        set_2_reference = DenseBitSet::new_filled(domain_size);
                    }
                }
                _ => unreachable!(),
            }

            // Check the contains function.
            for i in 0..domain_size {
                assert_eq!(set_1.contains(i), set_1_reference.contains(i));
                assert_eq!(set_2.contains(i), set_2_reference.contains(i));
            }

            // Check iter function.
            assert!(
                set_1.iter().eq(set_1_reference.iter()),
                "{:?}",
                set_1.iter().collect::<Vec<_>>()
            );
            assert!(set_2.iter().eq(set_2_reference.iter()));

            // Check the superset relation.
            assert_eq!(set_1.superset(&set_2), set_1_reference.superset(&set_2_reference));

            // Check the count function.
            assert_eq!(set_1.count(), set_1_reference.count());
            assert_eq!(set_2.count(), set_2_reference.count());

            // Check `last_set_in()`.
            if domain_size > 0 {
                let range = rng.sample_range(domain_size - 1);
                assert_eq!(
                    set_1.last_set_in(range.clone()),
                    set_1_reference.last_set_in(range.clone())
                );
                assert_eq!(
                    set_2.last_set_in(range.clone()),
                    set_2_reference.last_set_in(range.clone())
                );
            }
        }
    }

    #[test]
    fn test_fixed_size_bit_set() {
        assert_eq!(
            size_of::<ThinBitSet<usize>>(),
            size_of::<usize>(),
            "ThinBitSet should have the same size as a usize"
        );

        test_with_domain_size(0);
        test_with_domain_size(1);
        test_with_domain_size(63);
        test_with_domain_size(64);
        test_with_domain_size(65);
        test_with_domain_size(127);
        test_with_domain_size(128);
        test_with_domain_size(129);
    }
}
