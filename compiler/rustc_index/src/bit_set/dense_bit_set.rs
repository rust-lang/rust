use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, handle_alloc_error, realloc};
use std::hash::{Hash, Hasher};
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Range, RangeInclusive};
use std::ptr::NonNull;
use std::{fmt, iter, slice};

use itertools::Either;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use super::{
    BitRelations, CHUNK_WORDS, Chunk, ChunkedBitSet, WORD_BITS, Word, word_index_and_mask,
};
use crate::Idx;

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
/// in a [`OnceCell`](std::cell::OnceCell) or an [`Option`]—you can simply create an empty set and
/// populate it if needed.
///
/// Note 1: Since this bitset is dense, if your domain is large and/or relatively homogeneous (e.g.
/// long runs of set or unset bits), it may be more efficient to use a
/// [`MixedBitSet`](crate::bit_set::MixedBitSet) or an
/// [`IntervalSet`](crate::interval::IntervalSet), which are better suited for sparse or highly
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
pub union DenseBitSet<T> {
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
    /// hold the domain size (capacity) **in words** of the set, which is needed if the set is
    /// eventually allocated.
    ///
    /// Note that because the capacity is stored in words, not in bits, there is plenty of room
    /// for the two tag bits.
    empty_unallocated: usize,

    /// The bit set is stored on the heap.
    ///
    /// The two most significant bits are set to zero if this field is active.
    on_heap: ManuallyDrop<BitSetOnHeap>,

    /// This variant will never be created.
    marker: PhantomData<T>,
}

impl<T> DenseBitSet<T> {
    /// The maximum domain size that could be stored inlined on the stack.
    pub const INLINE_CAPACITY: usize = WORD_BITS - 1;

    /// A [`Word`] with the most significant bit set. That is the tag bit telling that the set is
    /// inlined.
    const IS_INLINE_TAG_BIT: Word = 0x1 << (WORD_BITS - 1);

    /// The tag for the `empty_unallocated` variant. The two most significant bits are
    /// `[0, 1]`.
    const EMPTY_UNALLOCATED_TAG_BITS: usize = 0b01 << (usize::BITS - 2);

    /// Create a new empty bit set with a given domain_size.
    ///
    /// If `domain_size` is <= [`Self::INLINE_CAPACITY`], then it is stored inline on the stack,
    /// otherwise it is stored on the heap.
    #[inline]
    pub fn new_empty(domain_size: usize) -> Self {
        if domain_size <= Self::INLINE_CAPACITY {
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
        if domain_size <= Self::INLINE_CAPACITY {
            Self {
                inline: Word::MAX.unbounded_shr((WORD_BITS - domain_size) as u32)
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
            // Trust me: this mask is correct.
            let last_word_mask = Word::MAX.wrapping_shr(domain_size.wrapping_neg() as u32);
            *words.last_mut().unwrap() &= last_word_mask;
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
        const MASK: usize = usize::MAX << usize::BITS - 2;
        (unsafe { self.empty_unallocated } & MASK) == Self::EMPTY_UNALLOCATED_TAG_BITS
    }

    /// Check if `self` is `empty_unallocated` and if so return the number of words required to
    /// store the expected capacity.
    // If this function returns `true`, it is safe to assume `self.empty_unallocated`. Else, it is
    // safe to assume `self.inline`, or `self.on_heap`.
    #[inline(always)]
    pub const fn empty_unallocated_get_num_words(&self) -> Option<usize> {
        if self.is_empty_unallocated() {
            Some(unsafe { self.empty_unallocated } ^ Self::EMPTY_UNALLOCATED_TAG_BITS)
        } else {
            None
        }
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
        if let Some(num_words) = self.empty_unallocated_get_num_words() {
            *self = Self { on_heap: ManuallyDrop::new(BitSetOnHeap::new_empty(num_words)) };
            unsafe { &mut self.on_heap }
        } else {
            self.on_heap_mut().unwrap()
        }
    }

    /// Get the capacity of this set. This is >= the initial domain size.
    #[inline(always)]
    pub(super) fn capacity(&self) -> usize {
        if self.is_inline() {
            Self::INLINE_CAPACITY
        } else if let Some(num_words) = self.empty_unallocated_get_num_words() {
            num_words * WORD_BITS
        } else {
            self.on_heap().unwrap().capacity()
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

    /// Get an iterator of all words making up the set.
    pub(super) fn words(&self) -> impl ExactSizeIterator<Item = Word> {
        if self.is_inline() {
            let word = unsafe { self.inline } ^ Self::IS_INLINE_TAG_BIT;
            Either::Left(iter::once(word))
        } else if let Some(num_words) = self.empty_unallocated_get_num_words() {
            Either::Right(Either::Left(iter::repeat_n(0, num_words)))
        } else {
            Either::Right(Either::Right(self.on_heap().unwrap().as_slice().iter().copied()))
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

    /// Returns an iterator over the indices for all elements in this set.
    #[inline(always)]
    pub fn iter_usizes(&self) -> BitIter<'_, usize> {
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

    /// Insert the elem with index `idx`. Returns `true` if the set has changed.
    #[inline(always)]
    fn insert_usize(&mut self, idx: usize) -> bool {
        // Insert the `i`th bit in a word and return `true` if it changed.
        let insert_bit = |word: &mut Word, bit_idx: u32| {
            let mask = 0x01 << bit_idx;
            let old = *word;
            *word |= mask;
            *word != old
        };

        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            debug_assert!(idx < Self::INLINE_CAPACITY, "index too large: {idx}");
            insert_bit(x, idx as u32)
        } else {
            let words = self.on_heap_get_or_alloc().as_mut_slice();

            let word_idx = idx / WORD_BITS;
            let bit_idx = (idx % WORD_BITS) as u32;
            let word = &mut words[word_idx];
            insert_bit(word, bit_idx)
        }
    }

    /// Insert `0..domain_size` in the set.
    ///
    /// We would like an insert all function that doesn't require the domain size, but the exact
    /// domain size is not stored so that is not possible.
    #[inline(always)]
    pub fn insert_all(&mut self, domain_size: usize) {
        if self.is_inline() {
            debug_assert!(domain_size <= Self::INLINE_CAPACITY);
            unsafe {
                self.inline |= Word::MAX.unbounded_shr(WORD_BITS as u32 - domain_size as u32)
            };
        } else {
            let on_heap = self.on_heap_get_or_alloc();
            debug_assert!(on_heap.capacity() >= domain_size, "domain size too big");
            let words = on_heap.as_mut_slice();

            let (end_word_index, end_mask) = word_index_and_mask(domain_size - 1);

            for word_index in 0..end_word_index {
                words[word_index] = Word::MAX;
            }

            words[end_word_index] |= end_mask | (end_mask - 1);
        }
    }

    /// Sets `self = self | !other` for all elements less than `domain_size`.
    #[inline(always)]
    pub fn union_not(&mut self, other: &Self, domain_size: usize) {
        if self.is_inline() {
            assert!(other.is_inline());

            let self_word = unsafe { &mut self.inline };
            let other_word = unsafe { other.inline };

            debug_assert!(domain_size <= Self::INLINE_CAPACITY);

            *self_word |= !other_word & Word::MAX.unbounded_shr((WORD_BITS - domain_size) as u32);
        } else if other.is_empty_unallocated() {
            self.insert_all(domain_size);
        } else {
            let self_words = self.on_heap_get_or_alloc().as_mut_slice();
            let other_words = other.on_heap().unwrap().as_slice();

            // Set all but the last word if domain_size is not divisible by `WORD_BITS`.
            for (self_word, other_word) in
                self_words.iter_mut().zip(other_words).take(domain_size / WORD_BITS)
            {
                *self_word |= !other_word;
            }

            let remaining_bits = domain_size % WORD_BITS;
            if remaining_bits > 0 {
                let last_idx = domain_size / WORD_BITS;
                self_words[last_idx] |= !other_words[last_idx] & !(Word::MAX << remaining_bits);
            }
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

impl<T> BitRelations<DenseBitSet<T>> for DenseBitSet<T> {
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

impl<T: Idx> DenseBitSet<T> {
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
            debug_assert!(idx < Self::INLINE_CAPACITY, "index too large: {idx}");
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
        self.insert_usize(elem.index())
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
            debug_assert!(idx < Self::INLINE_CAPACITY, "index too large: {idx}");
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
    pub fn iter(&self) -> BitIter<'_, T> {
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

    /// Returns `Some(elem)` if the set contains exactly one elemement otherwise returns `None`.
    #[inline(always)]
    pub fn only_one_elem(&self) -> Option<T> {
        if self.is_inline() {
            let word = unsafe { self.inline } ^ Self::IS_INLINE_TAG_BIT;
            if word.is_power_of_two() { Some(T::new(word.trailing_zeros() as usize)) } else { None }
        } else if self.is_empty_unallocated() {
            None
        } else {
            let words = self.on_heap().unwrap().as_slice();
            let mut found_elem = None;
            for (i, &word) in words.iter().enumerate() {
                if word > 0 {
                    if found_elem.is_some() {
                        return None;
                    }
                    if word.is_power_of_two() {
                        found_elem =
                            Some(T::new(i * WORD_BITS as usize + word.trailing_zeros() as usize));
                    } else {
                        return None;
                    }
                }
            }
            found_elem
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
            debug_assert!(end < Self::INLINE_CAPACITY);
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
            debug_assert!(end < Self::INLINE_CAPACITY);
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

impl<T: Idx> BitRelations<ChunkedBitSet<T>> for DenseBitSet<T> {
    fn union(&mut self, other: &ChunkedBitSet<T>) -> bool {
        other.iter().fold(false, |changed, elem| self.insert(elem) || changed)
    }

    fn subtract(&mut self, _other: &ChunkedBitSet<T>) -> bool {
        unimplemented!("implement if/when necessary");
    }

    fn intersect(&mut self, other: &ChunkedBitSet<T>) -> bool {
        if self.is_inline() {
            assert!(other.domain_size <= Self::INLINE_CAPACITY);
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

impl<S: Encoder, T> Encodable<S> for DenseBitSet<T> {
    #[inline(never)] // FIXME: For profiling purposes
    fn encode(&self, s: &mut S) {
        // The encoding is as follows:
        //
        // The `inline` and `empty_unallocated` variants are encoded as a single `Word`. Here, we
        // consider the `empty_unallocated` variant as the `inline` variant because
        // `empty_unallocated: usize`, `inline: Word`, and `usize` is smaller than `Word`.
        //
        // The `on_heap` variant is encoded as follows: First, the number of `Word`s are encoded
        // with a single `Word`. We assert that the two most significant bits of this number are 0
        // to distinguish it from the `inline` and `empty_unallocated` variants. Then all the words are
        // encoded in sequence.

        if let Some(on_heap) = self.on_heap() {
            let n_words: Word = on_heap.n_words();
            debug_assert_eq!(
                n_words >> WORD_BITS - 2,
                0x0,
                "the two most significant bits must be 0"
            );
            n_words.encode(s);
            debug_assert_eq!(n_words as usize, on_heap.as_slice().len());
            for word in on_heap.as_slice().iter() {
                word.encode(s);
            }
        } else {
            let word = unsafe { self.inline };
            debug_assert!(word >> WORD_BITS - 2 != 0, "the 2 most significant bits must not be 0");
            word.encode(s);
        }
    }
}

impl<D: Decoder, T> Decodable<D> for DenseBitSet<T> {
    #[inline(never)] // FIXME: For profiling purposes
    fn decode(d: &mut D) -> Self {
        // First we read one `Word` and check the variant.
        let word = Word::decode(d);
        if word >> WORD_BITS - 2 == 0x0 {
            // If the two most significant bits are 0, then this is the `on_heap` variant and the
            // number of words is encoded by `word`.
            let n_words = word as usize;
            assert!(
                n_words > 0,
                "DenseBitSet decoder error: At least one word must be stored with the `on_heap` variant."
            );
            let mut on_heap = BitSetOnHeap::new_empty(n_words);

            let words = on_heap.as_mut_slice();
            // All `words` are now initialised to 0x0.
            debug_assert_eq!(words.len(), n_words);

            // Decode the words one-by-one.
            for word in words.iter_mut() {
                *word = Word::decode(d);
            }

            DenseBitSet { on_heap: ManuallyDrop::new(on_heap) }
        } else {
            // Both the `inline` and `empty_unallocated` variants are encoded by one `Word`. We can
            // just assume the `inline` variant because the `empty_unallocated` variant is smaller
            // and the union is `repr(C)`.
            Self { inline: word }
        }
    }
}

impl<T> Clone for DenseBitSet<T> {
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

impl<T> Drop for DenseBitSet<T> {
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
        debug_assert!(len >= 1);

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

    /// Get the number of words.
    #[allow(dead_code)] // FIXME
    #[inline]
    fn n_words(&self) -> Word {
        let ptr = (self.0 << 2) as *const Word;
        unsafe { ptr.read() }
    }

    /// Get the capacity, that is the number of elements that can be stored in this set.
    fn capacity(&self) -> usize {
        let ptr = (self.0 << 2) as *const Word;
        let len = unsafe { ptr.read() } as usize;
        len * WORD_BITS
    }

    /// Make sure the set can hold at least `min_domain_size` elements. Reallocate if necessary.
    fn ensure_capacity(&mut self, min_domain_size: usize) {
        let len = min_domain_size.div_ceil(WORD_BITS);

        let old_ptr = (self.0 << 2) as *const Word;
        let old_len = unsafe { old_ptr.read() } as usize;

        if len <= old_len {
            return;
        }

        // The first word is used to store the total number of words. The rest of the words
        // store the bits.
        let num_words = len + 1;
        let old_num_words = old_len + 1;

        let new_layout = Layout::array::<Word>(num_words).expect("Bit set too large");
        let old_layout = Layout::array::<usize>(old_num_words).expect("Bit set too large");

        // SAFETY: `num_words` is always at least `1` so we never allocate zero size.
        let ptr =
            unsafe { realloc(old_ptr as *mut u8, old_layout, new_layout.size()).cast::<Word>() };
        let Some(ptr) = NonNull::<Word>::new(ptr) else {
            handle_alloc_error(new_layout);
        };

        // Store the length in the first word.
        unsafe { ptr.write(len as Word) };

        // Set all the new words to 0.
        for word_idx in old_num_words..num_words {
            unsafe { ptr.add(word_idx).write(0x0) }
        }

        // Convert `ptr` to a `usize` and shift it two bits to the right.
        self.0 = (ptr.as_ptr() as usize) >> 2
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
    pub(super) fn from_slice(words: &'a [Word]) -> Self {
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

impl<'a, T: Idx> FusedIterator for BitIter<'a, T> {}

impl<T: Idx> fmt::Debug for DenseBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        w.debug_list().entries(self.iter()).finish()
    }
}

impl<T> PartialEq for DenseBitSet<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.is_inline() {
            if other.is_inline() {
                unsafe { self.inline == other.inline }
            } else if other.is_empty_unallocated() {
                self.is_empty()
            } else {
                let other_words = other.on_heap().unwrap().as_slice();
                let self_word = unsafe { self.inline } ^ Self::IS_INLINE_TAG_BIT;
                other_words[0] == self_word && other_words[1..].iter().all(|&w| w == 0)
            }
        } else if self.is_empty_unallocated() {
            other.is_empty()
        } else {
            let self_words = self.on_heap().unwrap().as_slice();
            if other.is_empty_unallocated() {
                self_words.iter().all(|&w| w == 0)
            } else if other.is_inline() {
                let other_word = unsafe { other.inline } ^ Self::IS_INLINE_TAG_BIT;
                self_words[0] == other_word && self_words[1..].iter().all(|&w| w == 0)
            } else {
                let mut self_words = self_words.iter();
                let mut other_words = other.on_heap().unwrap().as_slice().iter();
                loop {
                    match (self_words.next(), other_words.next()) {
                        (Some(w1), Some(w2)) if w1 == w2 => (),
                        (Some(_), Some(_)) => break false,
                        (Some(0), None) | (None, Some(0)) => (),
                        (Some(_), None) | (None, Some(_)) => break false,
                        (None, None) => break true,
                    }
                }
            }
        }
    }
}

impl<T> Eq for DenseBitSet<T> {}

impl<T> Hash for DenseBitSet<T> {
    #[inline]
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        if self.is_inline() {
            let inline = unsafe { self.inline };
            inline.hash(hasher);
        } else if let Some(num_words) = self.empty_unallocated_get_num_words() {
            // Now we hash 0 for `num_words` times so that this hash should be equal to a cleared
            // set with the `on_heap` variant.
            for _ in 0..num_words {
                let zero_word: Word = 0x0;
                zero_word.hash(hasher);
            }
        } else {
            let words = self.on_heap().unwrap().as_slice();
            for word in words {
                word.hash(hasher);
            }
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
#[derive(Clone, PartialEq)]
pub struct GrowableBitSet<T> {
    bit_set: DenseBitSet<T>,
}

impl<T> Default for GrowableBitSet<T> {
    fn default() -> Self {
        GrowableBitSet::new_empty()
    }
}

impl<T> GrowableBitSet<T> {
    /// Ensure that the set can hold at least `min_domain_size` elements.
    pub fn ensure(&mut self, min_domain_size: usize) {
        if min_domain_size <= self.bit_set.capacity() {
            return;
        }

        if self.bit_set.is_inline() {
            // The set must change from being inlined to allocate on the heap.
            debug_assert!(min_domain_size > DenseBitSet::<T>::INLINE_CAPACITY);

            let mut new_bit_set = DenseBitSet::new_empty(min_domain_size);
            if !self.bit_set.is_empty() {
                // SAFETY: We know that `self.is_inline()` is true.
                let word = unsafe { self.bit_set.inline } ^ DenseBitSet::<T>::IS_INLINE_TAG_BIT;
                new_bit_set.on_heap_get_or_alloc().as_mut_slice()[0] = word;
            }
            self.bit_set = new_bit_set;
        } else if self.bit_set.is_empty_unallocated() {
            self.bit_set = DenseBitSet::new_empty(min_domain_size);
        } else {
            self.bit_set.on_heap_mut().unwrap().ensure_capacity(min_domain_size);
        }
    }

    pub fn new_empty() -> GrowableBitSet<T> {
        GrowableBitSet { bit_set: DenseBitSet::new_empty(0) }
    }

    pub fn with_capacity(capacity: usize) -> GrowableBitSet<T> {
        GrowableBitSet { bit_set: DenseBitSet::new_empty(capacity) }
    }

    /// Insert the element with index `idx`. Returns `true` if the set has changed.
    #[inline]
    pub fn insert_usize(&mut self, idx: usize) -> bool {
        self.ensure(idx + 1);
        self.bit_set.insert_usize(idx)
    }
}

impl<T: Idx> GrowableBitSet<T> {
    /// Insert `elem` into the set, resizing if necessary. Returns `true` if the set has changed.
    #[inline]
    pub fn insert(&mut self, elem: T) -> bool {
        self.insert_usize(elem.index())
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
        elem.index() < self.bit_set.capacity() && self.bit_set.contains(elem)
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

impl<T> From<DenseBitSet<T>> for GrowableBitSet<T> {
    fn from(bit_set: DenseBitSet<T>) -> Self {
        Self { bit_set }
    }
}

impl<T> From<GrowableBitSet<T>> for DenseBitSet<T> {
    fn from(bit_set: GrowableBitSet<T>) -> Self {
        bit_set.bit_set
    }
}

impl<T: Idx> fmt::Debug for GrowableBitSet<T> {
    fn fmt(&self, w: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.bit_set.fmt(w)
    }
}

#[inline]
fn max_bit(word: Word) -> usize {
    WORD_BITS - 1 - word.leading_zeros() as usize
}
