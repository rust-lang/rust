#[cfg(test)]
mod tests;

use std::hash;
use std::iter;
use std::ops::Range;

use rustc_target::abi::Size;

use super::AllocRange;

type Block = u64;

/// A bitmask where each bit refers to the byte with the same index. If the bit is `true`, the byte
/// is initialized. If it is `false` the byte is uninitialized.
/// The actual bits are only materialized when needed, and we try to keep this data lazy as long as
/// possible. Currently, if all the blocks have the same value, then the mask represents either a
/// fully initialized or fully uninitialized const allocation, so we can only store that single
/// value.
#[derive(Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
pub struct InitMask {
    blocks: InitMaskBlocks,
    len: Size,
}

#[derive(Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
enum InitMaskBlocks {
    Lazy {
        /// Whether the lazy init mask is fully initialized or uninitialized.
        state: bool,
    },
    Materialized(InitMaskMaterialized),
}

impl InitMask {
    pub fn new(size: Size, state: bool) -> Self {
        // Blocks start lazily allocated, until we have to materialize them.
        let blocks = InitMaskBlocks::Lazy { state };
        InitMask { len: size, blocks }
    }

    /// Checks whether the `range` is entirely initialized.
    ///
    /// Returns `Ok(())` if it's initialized. Otherwise returns a range of byte
    /// indexes for the first contiguous span of the uninitialized access.
    #[inline]
    pub fn is_range_initialized(&self, range: AllocRange) -> Result<(), AllocRange> {
        let end = range.end();
        if end > self.len {
            return Err(AllocRange::from(self.len..end));
        }

        match self.blocks {
            InitMaskBlocks::Lazy { state } => {
                // Lazily allocated blocks represent the full mask, and cover the requested range by
                // definition.
                if state { Ok(()) } else { Err(range) }
            }
            InitMaskBlocks::Materialized(ref blocks) => {
                blocks.is_range_initialized(range.start, end)
            }
        }
    }

    /// Sets a specified range to a value. If the range is out-of-bounds, the mask will grow to
    /// accomodate it entirely.
    pub fn set_range(&mut self, range: AllocRange, new_state: bool) {
        let start = range.start;
        let end = range.end();

        let is_full_overwrite = start == Size::ZERO && end >= self.len;

        // Optimize the cases of a full init/uninit state, while handling growth if needed.
        match self.blocks {
            InitMaskBlocks::Lazy { ref mut state } if is_full_overwrite => {
                // This is fully overwriting the mask, and we'll still have a single initialization
                // state: the blocks can stay lazy.
                *state = new_state;
                self.len = end;
            }
            InitMaskBlocks::Materialized(_) if is_full_overwrite => {
                // This is also fully overwriting materialized blocks with a single initialization
                // state: we'll have no need for these blocks anymore and can make them lazy.
                self.blocks = InitMaskBlocks::Lazy { state: new_state };
                self.len = end;
            }
            InitMaskBlocks::Lazy { state } if state == new_state => {
                // Here we're partially overwriting the mask but the initialization state doesn't
                // change: the blocks can stay lazy.
                if end > self.len {
                    self.len = end;
                }
            }
            _ => {
                // Otherwise, we have a partial overwrite that can result in a mix of initialization
                // states, so we'll need materialized blocks.
                let len = self.len;
                let blocks = self.materialize_blocks();

                // There are 3 cases of interest here, if we have:
                //
                //         [--------]
                //         ^        ^
                //         0        len
                //
                // 1) the range to set can be in-bounds:
                //
                //            xxxx = [start, end]
                //         [--------]
                //         ^        ^
                //         0        len
                //
                // Here, we'll simply set the single `start` to `end` range.
                //
                // 2) the range to set can be partially out-of-bounds:
                //
                //                xxxx = [start, end]
                //         [--------]
                //         ^        ^
                //         0        len
                //
                // We have 2 subranges to handle:
                // - we'll set the existing `start` to `len` range.
                // - we'll grow and set the `len` to `end` range.
                //
                // 3) the range to set can be fully out-of-bounds:
                //
                //                   ---xxxx = [start, end]
                //         [--------]
                //         ^        ^
                //         0        len
                //
                // Since we're growing the mask to a single `new_state` value, we consider the gap
                // from `len` to `start` to be part of the range, and have a single subrange to
                // handle: we'll grow and set the `len` to `end` range.
                //
                // Note that we have to materialize, set blocks, and grow the mask. We could
                // therefore slightly optimize things in situations where these writes overlap.
                // However, as of writing this, growing the mask doesn't happen in practice yet, so
                // we don't do this micro-optimization.

                if end <= len {
                    // Handle case 1.
                    blocks.set_range_inbounds(start, end, new_state);
                } else {
                    if start < len {
                        // Handle the first subrange of case 2.
                        blocks.set_range_inbounds(start, len, new_state);
                    }

                    // Handle the second subrange of case 2, and case 3.
                    blocks.grow(len, end - len, new_state); // `Size` operation
                    self.len = end;
                }
            }
        }
    }

    /// Materializes this mask's blocks when the mask is lazy.
    #[inline]
    fn materialize_blocks(&mut self) -> &mut InitMaskMaterialized {
        if let InitMaskBlocks::Lazy { state } = self.blocks {
            self.blocks = InitMaskBlocks::Materialized(InitMaskMaterialized::new(self.len, state));
        }

        let InitMaskBlocks::Materialized(ref mut blocks) = self.blocks else {
            bug!("initmask blocks must be materialized here")
        };
        blocks
    }

    /// Returns the initialization state at the specified in-bounds index.
    #[inline]
    pub fn get(&self, idx: Size) -> bool {
        match self.blocks {
            InitMaskBlocks::Lazy { state } => state,
            InitMaskBlocks::Materialized(ref blocks) => blocks.get(idx),
        }
    }
}

/// The actual materialized blocks of the bitmask, when we can't keep the `InitMask` lazy.
// Note: for performance reasons when interning, some of the fields can be partially
// hashed. (see the `Hash` impl below for more details), so the impl is not derived.
#[derive(Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable, HashStable)]
struct InitMaskMaterialized {
    blocks: Vec<Block>,
}

// Const allocations are only hashed for interning. However, they can be large, making the hashing
// expensive especially since it uses `FxHash`: it's better suited to short keys, not potentially
// big buffers like the allocation's init mask. We can partially hash some fields when they're
// large.
impl hash::Hash for InitMaskMaterialized {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        const MAX_BLOCKS_TO_HASH: usize = super::MAX_BYTES_TO_HASH / std::mem::size_of::<Block>();
        const MAX_BLOCKS_LEN: usize = super::MAX_HASHED_BUFFER_LEN / std::mem::size_of::<Block>();

        // Partially hash the `blocks` buffer when it is large. To limit collisions with common
        // prefixes and suffixes, we hash the length and some slices of the buffer.
        let block_count = self.blocks.len();
        if block_count > MAX_BLOCKS_LEN {
            // Hash the buffer's length.
            block_count.hash(state);

            // And its head and tail.
            self.blocks[..MAX_BLOCKS_TO_HASH].hash(state);
            self.blocks[block_count - MAX_BLOCKS_TO_HASH..].hash(state);
        } else {
            self.blocks.hash(state);
        }
    }
}

impl InitMaskMaterialized {
    pub const BLOCK_SIZE: u64 = 64;

    fn new(size: Size, state: bool) -> Self {
        let mut m = InitMaskMaterialized { blocks: vec![] };
        m.grow(Size::ZERO, size, state);
        m
    }

    #[inline]
    fn bit_index(bits: Size) -> (usize, usize) {
        // BLOCK_SIZE is the number of bits that can fit in a `Block`.
        // Each bit in a `Block` represents the initialization state of one byte of an allocation,
        // so we use `.bytes()` here.
        let bits = bits.bytes();
        let a = bits / Self::BLOCK_SIZE;
        let b = bits % Self::BLOCK_SIZE;
        (usize::try_from(a).unwrap(), usize::try_from(b).unwrap())
    }

    #[inline]
    fn size_from_bit_index(block: impl TryInto<u64>, bit: impl TryInto<u64>) -> Size {
        let block = block.try_into().ok().unwrap();
        let bit = bit.try_into().ok().unwrap();
        Size::from_bytes(block * Self::BLOCK_SIZE + bit)
    }

    /// Checks whether the `range` is entirely initialized.
    ///
    /// Returns `Ok(())` if it's initialized. Otherwise returns a range of byte
    /// indexes for the first contiguous span of the uninitialized access.
    #[inline]
    fn is_range_initialized(&self, start: Size, end: Size) -> Result<(), AllocRange> {
        let uninit_start = self.find_bit(start, end, false);

        match uninit_start {
            Some(uninit_start) => {
                let uninit_end = self.find_bit(uninit_start, end, true).unwrap_or(end);
                Err(AllocRange::from(uninit_start..uninit_end))
            }
            None => Ok(()),
        }
    }

    fn set_range_inbounds(&mut self, start: Size, end: Size, new_state: bool) {
        let (block_a, bit_a) = Self::bit_index(start);
        let (block_b, bit_b) = Self::bit_index(end);
        if block_a == block_b {
            // First set all bits except the first `bit_a`,
            // then unset the last `64 - bit_b` bits.
            let range = if bit_b == 0 {
                u64::MAX << bit_a
            } else {
                (u64::MAX << bit_a) & (u64::MAX >> (64 - bit_b))
            };
            if new_state {
                self.blocks[block_a] |= range;
            } else {
                self.blocks[block_a] &= !range;
            }
            return;
        }
        // across block boundaries
        if new_state {
            // Set `bit_a..64` to `1`.
            self.blocks[block_a] |= u64::MAX << bit_a;
            // Set `0..bit_b` to `1`.
            if bit_b != 0 {
                self.blocks[block_b] |= u64::MAX >> (64 - bit_b);
            }
            // Fill in all the other blocks (much faster than one bit at a time).
            for block in (block_a + 1)..block_b {
                self.blocks[block] = u64::MAX;
            }
        } else {
            // Set `bit_a..64` to `0`.
            self.blocks[block_a] &= !(u64::MAX << bit_a);
            // Set `0..bit_b` to `0`.
            if bit_b != 0 {
                self.blocks[block_b] &= !(u64::MAX >> (64 - bit_b));
            }
            // Fill in all the other blocks (much faster than one bit at a time).
            for block in (block_a + 1)..block_b {
                self.blocks[block] = 0;
            }
        }
    }

    #[inline]
    fn get(&self, i: Size) -> bool {
        let (block, bit) = Self::bit_index(i);
        (self.blocks[block] & (1 << bit)) != 0
    }

    fn grow(&mut self, len: Size, amount: Size, new_state: bool) {
        if amount.bytes() == 0 {
            return;
        }
        let unused_trailing_bits =
            u64::try_from(self.blocks.len()).unwrap() * Self::BLOCK_SIZE - len.bytes();

        // If there's not enough capacity in the currently allocated blocks, allocate some more.
        if amount.bytes() > unused_trailing_bits {
            let additional_blocks = amount.bytes() / Self::BLOCK_SIZE + 1;

            // We allocate the blocks to the correct value for the requested init state, so we won't
            // have to manually set them with another write.
            let block = if new_state { u64::MAX } else { 0 };
            self.blocks
                .extend(iter::repeat(block).take(usize::try_from(additional_blocks).unwrap()));
        }

        // New blocks have already been set here, so we only need to set the unused trailing bits,
        // if any.
        if unused_trailing_bits > 0 {
            let in_bounds_tail = Size::from_bytes(unused_trailing_bits);
            self.set_range_inbounds(len, len + in_bounds_tail, new_state); // `Size` operation
        }
    }

    /// Returns the index of the first bit in `start..end` (end-exclusive) that is equal to is_init.
    fn find_bit(&self, start: Size, end: Size, is_init: bool) -> Option<Size> {
        /// A fast implementation of `find_bit`,
        /// which skips over an entire block at a time if it's all 0s (resp. 1s),
        /// and finds the first 1 (resp. 0) bit inside a block using `trailing_zeros` instead of a loop.
        ///
        /// Note that all examples below are written with 8 (instead of 64) bit blocks for simplicity,
        /// and with the least significant bit (and lowest block) first:
        /// ```text
        ///        00000000|00000000
        ///        ^      ^ ^      ^
        /// index: 0      7 8      15
        /// ```
        /// Also, if not stated, assume that `is_init = true`, that is, we are searching for the first 1 bit.
        fn find_bit_fast(
            init_mask: &InitMaskMaterialized,
            start: Size,
            end: Size,
            is_init: bool,
        ) -> Option<Size> {
            /// Search one block, returning the index of the first bit equal to `is_init`.
            fn search_block(
                bits: Block,
                block: usize,
                start_bit: usize,
                is_init: bool,
            ) -> Option<Size> {
                // For the following examples, assume this function was called with:
                //   bits = 0b00111011
                //   start_bit = 3
                //   is_init = false
                // Note that, for the examples in this function, the most significant bit is written first,
                // which is backwards compared to the comments in `find_bit`/`find_bit_fast`.

                // Invert bits so we're always looking for the first set bit.
                //        ! 0b00111011
                //   bits = 0b11000100
                let bits = if is_init { bits } else { !bits };
                // Mask off unused start bits.
                //          0b11000100
                //        & 0b11111000
                //   bits = 0b11000000
                let bits = bits & (!0 << start_bit);
                // Find set bit, if any.
                //   bit = trailing_zeros(0b11000000)
                //   bit = 6
                if bits == 0 {
                    None
                } else {
                    let bit = bits.trailing_zeros();
                    Some(InitMaskMaterialized::size_from_bit_index(block, bit))
                }
            }

            if start >= end {
                return None;
            }

            // Convert `start` and `end` to block indexes and bit indexes within each block.
            // We must convert `end` to an inclusive bound to handle block boundaries correctly.
            //
            // For example:
            //
            //   (a) 00000000|00000000    (b) 00000000|
            //       ^~~~~~~~~~~^             ^~~~~~~~~^
            //     start       end          start     end
            //
            // In both cases, the block index of `end` is 1.
            // But we do want to search block 1 in (a), and we don't in (b).
            //
            // We subtract 1 from both end positions to make them inclusive:
            //
            //   (a) 00000000|00000000    (b) 00000000|
            //       ^~~~~~~~~~^              ^~~~~~~^
            //     start    end_inclusive   start end_inclusive
            //
            // For (a), the block index of `end_inclusive` is 1, and for (b), it's 0.
            // This provides the desired behavior of searching blocks 0 and 1 for (a),
            // and searching only block 0 for (b).
            // There is no concern of overflows since we checked for `start >= end` above.
            let (start_block, start_bit) = InitMaskMaterialized::bit_index(start);
            let end_inclusive = Size::from_bytes(end.bytes() - 1);
            let (end_block_inclusive, _) = InitMaskMaterialized::bit_index(end_inclusive);

            // Handle first block: need to skip `start_bit` bits.
            //
            // We need to handle the first block separately,
            // because there may be bits earlier in the block that should be ignored,
            // such as the bit marked (1) in this example:
            //
            //       (1)
            //       -|------
            //   (c) 01000000|00000000|00000001
            //          ^~~~~~~~~~~~~~~~~~^
            //        start              end
            if let Some(i) =
                search_block(init_mask.blocks[start_block], start_block, start_bit, is_init)
            {
                // If the range is less than a block, we may find a matching bit after `end`.
                //
                // For example, we shouldn't successfully find bit (2), because it's after `end`:
                //
                //             (2)
                //       -------|
                //   (d) 00000001|00000000|00000001
                //        ^~~~~^
                //      start end
                //
                // An alternative would be to mask off end bits in the same way as we do for start bits,
                // but performing this check afterwards is faster and simpler to implement.
                if i < end {
                    return Some(i);
                } else {
                    return None;
                }
            }

            // Handle remaining blocks.
            //
            // We can skip over an entire block at once if it's all 0s (resp. 1s).
            // The block marked (3) in this example is the first block that will be handled by this loop,
            // and it will be skipped for that reason:
            //
            //                   (3)
            //                --------
            //   (e) 01000000|00000000|00000001
            //          ^~~~~~~~~~~~~~~~~~^
            //        start              end
            if start_block < end_block_inclusive {
                // This loop is written in a specific way for performance.
                // Notably: `..end_block_inclusive + 1` is used for an inclusive range instead of `..=end_block_inclusive`,
                // and `.zip(start_block + 1..)` is used to track the index instead of `.enumerate().skip().take()`,
                // because both alternatives result in significantly worse codegen.
                // `end_block_inclusive + 1` is guaranteed not to wrap, because `end_block_inclusive <= end / BLOCK_SIZE`,
                // and `BLOCK_SIZE` (the number of bits per block) will always be at least 8 (1 byte).
                for (&bits, block) in init_mask.blocks[start_block + 1..end_block_inclusive + 1]
                    .iter()
                    .zip(start_block + 1..)
                {
                    if let Some(i) = search_block(bits, block, 0, is_init) {
                        // If this is the last block, we may find a matching bit after `end`.
                        //
                        // For example, we shouldn't successfully find bit (4), because it's after `end`:
                        //
                        //                               (4)
                        //                         -------|
                        //   (f) 00000001|00000000|00000001
                        //          ^~~~~~~~~~~~~~~~~~^
                        //        start              end
                        //
                        // As above with example (d), we could handle the end block separately and mask off end bits,
                        // but unconditionally searching an entire block at once and performing this check afterwards
                        // is faster and much simpler to implement.
                        if i < end {
                            return Some(i);
                        } else {
                            return None;
                        }
                    }
                }
            }

            None
        }

        #[cfg_attr(not(debug_assertions), allow(dead_code))]
        fn find_bit_slow(
            init_mask: &InitMaskMaterialized,
            start: Size,
            end: Size,
            is_init: bool,
        ) -> Option<Size> {
            (start..end).find(|&i| init_mask.get(i) == is_init)
        }

        let result = find_bit_fast(self, start, end, is_init);

        debug_assert_eq!(
            result,
            find_bit_slow(self, start, end, is_init),
            "optimized implementation of find_bit is wrong for start={:?} end={:?} is_init={} init_mask={:#?}",
            start,
            end,
            is_init,
            self
        );

        result
    }
}

/// A contiguous chunk of initialized or uninitialized memory.
pub enum InitChunk {
    Init(Range<Size>),
    Uninit(Range<Size>),
}

impl InitChunk {
    #[inline]
    pub fn is_init(&self) -> bool {
        match self {
            Self::Init(_) => true,
            Self::Uninit(_) => false,
        }
    }

    #[inline]
    pub fn range(&self) -> Range<Size> {
        match self {
            Self::Init(r) => r.clone(),
            Self::Uninit(r) => r.clone(),
        }
    }
}

impl InitMask {
    /// Returns an iterator, yielding a range of byte indexes for each contiguous region
    /// of initialized or uninitialized bytes inside the range `start..end` (end-exclusive).
    ///
    /// The iterator guarantees the following:
    /// - Chunks are nonempty.
    /// - Chunks are adjacent (each range's start is equal to the previous range's end).
    /// - Chunks span exactly `start..end` (the first starts at `start`, the last ends at `end`).
    /// - Chunks alternate between [`InitChunk::Init`] and [`InitChunk::Uninit`].
    #[inline]
    pub fn range_as_init_chunks(&self, range: AllocRange) -> InitChunkIter<'_> {
        let start = range.start;
        let end = range.end();
        assert!(end <= self.len);

        let is_init = if start < end {
            self.get(start)
        } else {
            // `start..end` is empty: there are no chunks, so use some arbitrary value
            false
        };

        InitChunkIter { init_mask: self, is_init, start, end }
    }
}

/// Yields [`InitChunk`]s. See [`InitMask::range_as_init_chunks`].
#[derive(Clone)]
pub struct InitChunkIter<'a> {
    init_mask: &'a InitMask,
    /// Whether the next chunk we will return is initialized.
    /// If there are no more chunks, contains some arbitrary value.
    is_init: bool,
    /// The current byte index into `init_mask`.
    start: Size,
    /// The end byte index into `init_mask`.
    end: Size,
}

impl<'a> Iterator for InitChunkIter<'a> {
    type Item = InitChunk;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        let end_of_chunk = match self.init_mask.blocks {
            InitMaskBlocks::Lazy { .. } => {
                // If we're iterating over the chunks of lazy blocks, we just emit a single
                // full-size chunk.
                self.end
            }
            InitMaskBlocks::Materialized(ref blocks) => {
                let end_of_chunk =
                    blocks.find_bit(self.start, self.end, !self.is_init).unwrap_or(self.end);
                end_of_chunk
            }
        };
        let range = self.start..end_of_chunk;
        let ret =
            Some(if self.is_init { InitChunk::Init(range) } else { InitChunk::Uninit(range) });

        self.is_init = !self.is_init;
        self.start = end_of_chunk;

        ret
    }
}

/// Run-length encoding of the uninit mask.
/// Used to copy parts of a mask multiple times to another allocation.
pub struct InitCopy {
    /// Whether the first range is initialized.
    initial: bool,
    /// The lengths of ranges that are run-length encoded.
    /// The initialization state of the ranges alternate starting with `initial`.
    ranges: smallvec::SmallVec<[u64; 1]>,
}

impl InitCopy {
    pub fn no_bytes_init(&self) -> bool {
        // The `ranges` are run-length encoded and of alternating initialization state.
        // So if `ranges.len() > 1` then the second block is an initialized range.
        !self.initial && self.ranges.len() == 1
    }
}

/// Transferring the initialization mask to other allocations.
impl InitMask {
    /// Creates a run-length encoding of the initialization mask; panics if range is empty.
    ///
    /// This is essentially a more space-efficient version of
    /// `InitMask::range_as_init_chunks(...).collect::<Vec<_>>()`.
    pub fn prepare_copy(&self, range: AllocRange) -> InitCopy {
        // Since we are copying `size` bytes from `src` to `dest + i * size` (`for i in 0..repeat`),
        // a naive initialization mask copying algorithm would repeatedly have to read the initialization mask from
        // the source and write it to the destination. Even if we optimized the memory accesses,
        // we'd be doing all of this `repeat` times.
        // Therefore we precompute a compressed version of the initialization mask of the source value and
        // then write it back `repeat` times without computing any more information from the source.

        // A precomputed cache for ranges of initialized / uninitialized bits
        // 0000010010001110 will become
        // `[5, 1, 2, 1, 3, 3, 1]`,
        // where each element toggles the state.

        let mut ranges = smallvec::SmallVec::<[u64; 1]>::new();

        let mut chunks = self.range_as_init_chunks(range).peekable();

        let initial = chunks.peek().expect("range should be nonempty").is_init();

        // Here we rely on `range_as_init_chunks` to yield alternating init/uninit chunks.
        for chunk in chunks {
            let len = chunk.range().end.bytes() - chunk.range().start.bytes();
            ranges.push(len);
        }

        InitCopy { ranges, initial }
    }

    /// Applies multiple instances of the run-length encoding to the initialization mask.
    pub fn apply_copy(&mut self, defined: InitCopy, range: AllocRange, repeat: u64) {
        // An optimization where we can just overwrite an entire range of initialization bits if
        // they are going to be uniformly `1` or `0`. If this happens to be a full-range overwrite,
        // we won't need materialized blocks either.
        if defined.ranges.len() <= 1 {
            let start = range.start;
            let end = range.start + range.size * repeat; // `Size` operations
            self.set_range(AllocRange::from(start..end), defined.initial);
            return;
        }

        // We're about to do one or more partial writes, so we ensure the blocks are materialized.
        let blocks = self.materialize_blocks();

        for mut j in 0..repeat {
            j *= range.size.bytes();
            j += range.start.bytes();
            let mut cur = defined.initial;
            for range in &defined.ranges {
                let old_j = j;
                j += range;
                blocks.set_range_inbounds(Size::from_bytes(old_j), Size::from_bytes(j), cur);
                cur = !cur;
            }
        }
    }
}
