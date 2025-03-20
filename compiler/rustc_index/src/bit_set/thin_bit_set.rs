use std::alloc::{Layout, alloc, alloc_zeroed, dealloc, handle_alloc_error};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::{fmt, slice};

use super::BitRelations;
use crate::{Idx, IndexVec};

/// A fixed-size bitset type with a dense representation, only using one word on the stack.
///
/// This bit set only takes the space of a `usize` on the stack. Assuming the [usize] is 64 bits
/// large, it can hold a domain size up to 63 bits inline on the stack. If the domain size is
/// larger, it will turn into a pointer instead pointing to a sequence of usizes on the heap. This
/// means that this is really cheap for domain sizes <64 bits on a 64-bit machine.
///
/// Note 1: Since this bitset is dense, if your domain is big, and/or relatively homogeneous (for
/// example, with long runs of bits set or unset), then it may be preferable to instead use a
/// [MixedBitSet], or an [IntervalSet](crate::interval::IntervalSet). They should be more suited to
/// sparse, or highly-compressible, domains.
///
/// Note 2: Use [`GrowableBitSet`] if you need support for resizing after creation.
///
/// Note 3: Be aware that the inline capacity is `usize::BITS - 1`. This might lead to performance
/// surprizes if you run on a 32-bit machine as all domain sizes larger than 31 will be stored on
/// the heap.
///
/// `T` is an index type, typically a newtyped `usize` wrapper, but it can also just be `usize`.
///
/// All operations that involve an element may panic if the element is equal to or greater than the
/// domain size. All operations that involve two bitsets may panic if the bitsets have differing
/// domain sizes. Note that panicking is not guaranteed, because the domain size will be rounded up
/// to the nearest multiple of [usize::BITS] upon creation of the set.
pub union ThinBitSet<T> {
    /// The bit set fits in a single `usize` stored inline on the stack.
    ///
    /// The least significant bit is set to 1 to distinguish this from a pointer to the heap. You
    /// must never change that "tag bit" after the bit set has been created.
    inline: usize,

    /// The bit set is stored on the heap.
    on_heap: ManuallyDrop<BitSetOnHeap>,

    /// This variant will never be created.
    marker: PhantomData<T>,
}

impl<T> ThinBitSet<T> {
    /// The maximum domain size that could be stored inlined on the stack.
    pub const MAX_INLINE_DOMAIN_SIZE: usize = usize::BITS as usize - 1;

    /// Create a new bit set with a given domain_size.
    ///
    /// If `domain_size` is <= [`Self::MAX_INLINE_DOMAIN_SIZE`], then it is stored inline on the stack,
    /// otherwise it is stored on the heap.
    #[inline]
    pub fn new_empty(domain_size: usize) -> Self {
        if domain_size <= Self::MAX_INLINE_DOMAIN_SIZE {
            // The last bit is set to indicate the union variant.
            Self { inline: 0x01 }
        } else {
            Self { on_heap: ManuallyDrop::new(BitSetOnHeap::new_empty(domain_size)) }
        }
    }

    /// Check if `self` is inlined or stored on the heap.
    // If this function returns `true`, it is safe to assume `self.inline`. Else, it is safe to
    // assume `self.on_heap`.
    #[inline(always)]
    pub fn is_inline(&self) -> bool {
        // We check if the last bit is set. If so, it is inlined, otherwise it represents a pointer
        // on the heap.
        // SAFETY: The union is either a `usize` or a pointer which can be cast to a `usize`.
        (unsafe { self.inline } & 0x01) != 0
    }

    /// Checks if the bit set is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        if self.is_inline() {
            let x = unsafe { self.inline };
            // The trailing 1 makes the empty bit set be 0x01.
            x == 0x01
        } else {
            let on_heap = unsafe { &self.on_heap };
            on_heap.as_slice().iter().all(|&x| x == 0)
        }
    }

    /// Clear the set.
    #[inline(always)]
    pub fn clear(&mut self) {
        if self.is_inline() {
            unsafe { self.inline &= 0x01 }
        } else {
            let on_heap = unsafe { &mut self.on_heap };
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
        let word_is_superset = |x: usize, other: usize| (!x & other) == 0;

        if self.is_inline() {
            let x = unsafe { self.inline };
            assert!(other.is_inline(), "bit sets has different domain sizes");
            let y = unsafe { other.inline };
            word_is_superset(x, y)
        } else {
            let on_heap = unsafe { &self.on_heap };
            assert!(!other.is_inline(), "bit sets has different domain sizes");
            let other_on_heap = unsafe { &other.on_heap };
            let self_slice = on_heap.as_slice();
            let other_slice = other_on_heap.as_slice();
            assert_eq!(self_slice.len(), other_slice.len(), "bit sets have different domain sizes");
            self_slice.iter().zip(other_slice).all(|(&x, &y)| (!x & y) == 0)
        }
    }

    /// Common function for union/intersection like operations.
    ///
    /// This function takes two bit sets, one mutably and one immutably. It asserts that they have
    /// the same domain_size. It then applies a function to all usizes in them. Like a zip operation.
    /// It checks if `self` has changed. If so, it returns `true`, `false` otherwise.
    ///
    /// ## Safety
    ///
    /// If the sets are inlined, this will leave the tag bit set to 1. You must not change it. If
    /// you change it, it will lead to undefined behaviour. This might be inconvenient for
    /// operations like subtraction. In that case, use `binary_operation_safe` instead.
    #[inline(always)]
    unsafe fn binary_operation(&mut self, other: &Self, op: impl Fn(&mut usize, usize)) -> bool {
        // Apply `op` and return if the word changed.
        let apply_and_check_change = |x: &mut usize, y: usize| -> bool {
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
    /// Note that the tag bit will still be set in the call to `op`, but there is no dangour in
    /// changing it as it will be restored afterusizes.
    #[inline(always)]
    fn binary_operation_safe(&mut self, other: &Self, op: impl Fn(&mut usize, usize)) -> bool {
        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            assert!(other.is_inline(), "bit sets has different domain sizes");
            let y = unsafe { other.inline };

            let old = *x;
            op(x, y);
            *x |= 0x01;
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
        // SAFETY: The union operation does not remove any bit set to 1, so the tag bit is
        // unaffected.
        unsafe { self.binary_operation(other, |x, y| *x |= y) }
    }

    #[inline(always)]
    fn intersect(&mut self, other: &Self) -> bool {
        // SAFETY: Since the tag bit is set in both `self` and `other`, the intersection won't
        // remove it.
        unsafe { self.binary_operation(other, |x, y| *x &= y) }
    }

    #[inline(always)]
    fn subtract(&mut self, other: &Self) -> bool {
        self.binary_operation_safe(other, |x, y| *x &= !y)
    }
}

impl<T: Idx> ThinBitSet<T> {
    /// Checks if the bit set contains `elem`.
    #[inline(always)]
    pub fn contains(&self, elem: T) -> bool {
        // Check if the `i`th bit is set in a word.
        let contains_bit = |word: usize, bit_idx: u32| {
            let mask = 0x01 << bit_idx;
            (word & mask) != 0
        };

        let idx = elem.index();
        if self.is_inline() {
            let x = unsafe { self.inline };
            assert!(idx <= Self::MAX_INLINE_DOMAIN_SIZE, "index too large: {idx}");
            // Add 1 to the bit index to account for the tag bit.
            let bit_idx = idx + 1;
            contains_bit(x, bit_idx as u32)
        } else {
            let on_heap = unsafe { &self.on_heap };
            let word_idx = idx / usize::BITS as usize;
            let bit_idx = (idx % usize::BITS as usize) as u32;
            let word = on_heap.as_slice()[word_idx];
            contains_bit(word, bit_idx)
        }
    }

    /// Insert `elem`. Returns `true` if the set has changed.
    #[inline(always)]
    pub fn insert(&mut self, elem: T) -> bool {
        // Insert the `i`th bit in a word and return `true` if it changed.
        let insert_bit = |word: &mut usize, bit_idx: u32| {
            let mask = 0x01 << bit_idx;
            let old = *word;
            *word |= mask;
            *word != old
        };

        let idx = elem.index();
        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            assert!(idx <= Self::MAX_INLINE_DOMAIN_SIZE, "index too large: {idx}");
            // Add 1 to the bit index to account for the tag bit.
            let bit_idx = idx + 1;
            insert_bit(x, bit_idx as u32)
        } else {
            let on_heap = unsafe { &mut self.on_heap };
            let word_idx = idx / usize::BITS as usize;
            let bit_idx = (idx % usize::BITS as usize) as u32;
            let word = &mut on_heap.as_mut_slice()[word_idx];
            insert_bit(word, bit_idx)
        }
    }

    /// Remove `elem`. Returns `true` if the set has changed.
    #[inline(always)]
    pub fn remove(&mut self, elem: T) -> bool {
        // Remove the `i`th bit in a word and return `true` if it changed.
        let remove_bit = |word: &mut usize, bit_idx: u32| {
            let mask = !(0x01 << bit_idx);
            let old = *word;
            *word &= mask;
            *word != old
        };

        let idx = elem.index();
        if self.is_inline() {
            let x = unsafe { &mut self.inline };
            assert!(idx <= Self::MAX_INLINE_DOMAIN_SIZE, "index too large: {idx}");
            // Add 1 to the bit index to account for the tag bit.
            let bit_idx = idx + 1;
            remove_bit(x, bit_idx as u32)
        } else {
            let on_heap = unsafe { &mut self.on_heap };
            let word_idx = idx / usize::BITS as usize;
            let bit_idx = (idx % usize::BITS as usize) as u32;
            let word = &mut on_heap.as_mut_slice()[word_idx];
            remove_bit(word, bit_idx)
        }
    }

    /// Returns an iterator over all elements in this set.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = T> + use<'_, T> {
        if self.is_inline() {
            let x = unsafe { self.inline };
            // Remove the tag bit by shifting to the right one step.
            let without_tag_bit = x >> 1;
            BitIter::from_single_word(without_tag_bit)
        } else {
            let on_heap = unsafe { &self.on_heap };
            BitIter::from_slice(on_heap.as_slice())
        }
    }
}

/// A pointer to a dense bit set on the heap.
///
/// This struct contains one single pointer, pointing to a sequence of words on the heap. The first
/// word is the length. It tells how many words (`usize`s) are stored in the remainder of the
/// sequence. The rest of the words in the sequence makes up the bit set.
///
/// For instance, if the bit set should have domain_size for 240 bits. We begin by counting the
/// number of words required to store 240 bits. That's 4 words assuming we are working on a 64-bit
/// machine. Then, the pointer in this struct points to an allocated sequence of five words. The
/// first of these has the value 4 (the length) and the subsequent four words makes up the bit set.
///
/// This struct is just a wrapper around a pointer, so the last bits should always be 0.
#[repr(transparent)]
struct BitSetOnHeap(NonNull<usize>);

impl BitSetOnHeap {
    fn new_empty(domain_size: usize) -> Self {
        // The first word is used to store the total number of words. The rest of the words
        // store the bits.
        let len = domain_size.div_ceil(usize::BITS as usize);
        let num_words = len + 1;

        let layout = Layout::array::<usize>(num_words).expect("Bit set too large");
        // SAFETY: `num_words` is always at least `1` so we never allocate zero size.
        let ptr = unsafe { alloc_zeroed(layout).cast::<usize>() };
        let Some(ptr) = NonNull::<usize>::new(ptr) else {
            handle_alloc_error(layout);
        };

        // Store the length in the first usize.
        unsafe { ptr.write(len as usize) };

        BitSetOnHeap(ptr)
    }

    /// Get a slice with all bits in this bit set.
    ///
    /// Note that the number of bits in the set is rounded up to the next power of `Usize::BITS`. So
    /// if the user requested a domain_size of 216 bits, a slice with 4 words will be returned on a
    /// 64-bit machine.
    #[inline]
    fn as_slice(&self) -> &[usize] {
        let len = unsafe { self.0.read() } as usize;
        // The slice starts at the second word.
        unsafe { slice::from_raw_parts(self.0.add(1).as_ptr(), len) }
    }

    /// Get a mutable slice with all bits in this bit set.
    ///
    /// Note that the number of bits in the set is rounded up to the next power of `Usize::BITS`. So
    /// if the user requested a domain_size of 216 bits, a slice with 4 words will be returned on a
    /// 64-bit machine.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [usize] {
        let len = unsafe { self.0.read() } as usize;
        // The slice starts at the second word.
        unsafe { slice::from_raw_parts_mut(self.0.add(1).as_ptr(), len) }
    }
}

impl<T> Clone for ThinBitSet<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        if self.is_inline() {
            let inline = unsafe { self.inline };
            Self { inline }
        } else {
            let old_on_heap = unsafe { &self.on_heap };
            let on_heap = old_on_heap.clone();
            Self { on_heap }
        }
    }
}

impl Clone for BitSetOnHeap {
    fn clone(&self) -> Self {
        let len = unsafe { self.0.read() } as usize;
        let num_words = len + 1;

        let layout = Layout::array::<usize>(num_words).expect("Bit set too large");
        // SAFETY: `num_words` is always at least `1` so we never allocate zero size.
        let ptr = unsafe { alloc(layout).cast::<usize>() };
        let Some(ptr) = NonNull::<usize>::new(ptr) else {
            handle_alloc_error(layout);
        };

        unsafe { self.0.copy_to_nonoverlapping(ptr, num_words) };

        BitSetOnHeap(ptr)
    }
}

impl<T> Drop for ThinBitSet<T> {
    #[inline(always)]
    fn drop(&mut self) {
        // Deallocate if `self` is not inlined.
        if !self.is_inline() {
            // SAFETY: `self.is_inline()` returned `false`.
            let on_heap = unsafe { &mut self.on_heap };
            unsafe {
                ManuallyDrop::drop(on_heap);
            }
        }
    }
}

impl Drop for BitSetOnHeap {
    fn drop(&mut self) {
        // SAFETY: The first word stores the number of words for the bit set. We have to add 1
        // because the first word storing the length is allocated as well.
        let num_words = unsafe { self.0.read() } as usize + 1;
        let layout = Layout::array::<usize>(num_words).expect("Bit set too large");
        // SAFETY: We know that `on_heap` has been allocated with the same layout. See the
        // `new` method for reference.
        unsafe { dealloc(self.0.as_ptr().cast::<u8>(), layout) };
    }
}

struct BitIter<'a, T: Idx> {
    /// A copy of the current word, but with any already-visited bits cleared.
    /// (This lets us use `trailing_zeros()` to find the next set bit.) When it
    /// is reduced to 0, we move onto the next word.
    word: usize,

    /// The offset (measured in bits) of the current word.
    offset: usize,

    /// Underlying iterator over the words.
    iter: slice::Iter<'a, usize>,

    marker: PhantomData<T>,
}

impl<'a, T: Idx> BitIter<'a, T> {
    fn from_slice(words: &'a [usize]) -> Self {
        // We initialize `word` and `offset` to degenerate values. On the first
        // call to `next()` we will fall through to getting the first word from
        // `iter`, which sets `word` to the first word (if there is one) and
        // `offset` to 0. Doing it this way saves us from having to maintain
        // additional state about whether we have started.
        Self {
            word: 0,
            offset: usize::MAX - (usize::BITS as usize - 1),
            iter: words.iter(),
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn from_single_word(word: usize) -> Self {
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
            self.offset = self.offset.wrapping_add(usize::BITS as usize);
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

    const TEST_ITERATIONS: u32 = 256;

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
                0..50 => {
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
                97..100 => {
                    // Clear
                    if rng.next_bool() {
                        set_1.clear();
                        set_1_reference.clear();
                    } else {
                        set_2.clear();
                        set_2_reference.clear();
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
            assert!(set_1.iter().eq(set_1_reference.iter()));
            assert!(set_2.iter().eq(set_2_reference.iter()));

            // Check the superset relation.
            assert_eq!(set_1.superset(&set_2), set_1_reference.superset(&set_2_reference));
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
