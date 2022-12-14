use std::hash;
use std::iter;
use std::ops::Range;

use rustc_target::abi::Size;

use super::AllocRange;

type Block = u64;

/// A bitmask where each bit refers to the byte with the same index. If the bit is `true`, the byte
/// is initialized. If it is `false` the byte is uninitialized.
// Note: for performance reasons when interning, some of the `InitMask` fields can be partially
// hashed. (see the `Hash` impl below for more details), so the impl is not derived.
#[derive(Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct InitMask {
    blocks: Vec<Block>,
    len: Size,
}

// Const allocations are only hashed for interning. However, they can be large, making the hashing
// expensive especially since it uses `FxHash`: it's better suited to short keys, not potentially
// big buffers like the allocation's init mask. We can partially hash some fields when they're
// large.
impl hash::Hash for InitMask {
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

        // Hash the other fields as usual.
        self.len.hash(state);
    }
}

impl InitMask {
    pub const BLOCK_SIZE: u64 = 64;

    pub fn new(size: Size, state: bool) -> Self {
        let mut m = InitMask { blocks: vec![], len: Size::ZERO };
        m.grow(size, state);
        m
    }

    #[inline]
    fn bit_index(bits: Size) -> (usize, usize) {
        // BLOCK_SIZE is the number of bits that can fit in a `Block`.
        // Each bit in a `Block` represents the initialization state of one byte of an allocation,
        // so we use `.bytes()` here.
        let bits = bits.bytes();
        let a = bits / InitMask::BLOCK_SIZE;
        let b = bits % InitMask::BLOCK_SIZE;
        (usize::try_from(a).unwrap(), usize::try_from(b).unwrap())
    }

    #[inline]
    fn size_from_bit_index(block: impl TryInto<u64>, bit: impl TryInto<u64>) -> Size {
        let block = block.try_into().ok().unwrap();
        let bit = bit.try_into().ok().unwrap();
        Size::from_bytes(block * InitMask::BLOCK_SIZE + bit)
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

        let uninit_start = self.find_bit(range.start, end, false);

        match uninit_start {
            Some(uninit_start) => {
                let uninit_end = self.find_bit(uninit_start, end, true).unwrap_or(end);
                Err(AllocRange::from(uninit_start..uninit_end))
            }
            None => Ok(()),
        }
    }

    pub fn set_range(&mut self, range: AllocRange, new_state: bool) {
        let end = range.end();
        let len = self.len;
        if end > len {
            self.grow(end - len, new_state);
        }
        self.set_range_inbounds(range.start, end, new_state);
    }

    fn set_range_inbounds(&mut self, start: Size, end: Size, new_state: bool) {
        let (blocka, bita) = Self::bit_index(start);
        let (blockb, bitb) = Self::bit_index(end);
        if blocka == blockb {
            // First set all bits except the first `bita`,
            // then unset the last `64 - bitb` bits.
            let range = if bitb == 0 {
                u64::MAX << bita
            } else {
                (u64::MAX << bita) & (u64::MAX >> (64 - bitb))
            };
            if new_state {
                self.blocks[blocka] |= range;
            } else {
                self.blocks[blocka] &= !range;
            }
            return;
        }
        // across block boundaries
        if new_state {
            // Set `bita..64` to `1`.
            self.blocks[blocka] |= u64::MAX << bita;
            // Set `0..bitb` to `1`.
            if bitb != 0 {
                self.blocks[blockb] |= u64::MAX >> (64 - bitb);
            }
            // Fill in all the other blocks (much faster than one bit at a time).
            for block in (blocka + 1)..blockb {
                self.blocks[block] = u64::MAX;
            }
        } else {
            // Set `bita..64` to `0`.
            self.blocks[blocka] &= !(u64::MAX << bita);
            // Set `0..bitb` to `0`.
            if bitb != 0 {
                self.blocks[blockb] &= !(u64::MAX >> (64 - bitb));
            }
            // Fill in all the other blocks (much faster than one bit at a time).
            for block in (blocka + 1)..blockb {
                self.blocks[block] = 0;
            }
        }
    }

    #[inline]
    pub fn get(&self, i: Size) -> bool {
        let (block, bit) = Self::bit_index(i);
        (self.blocks[block] & (1 << bit)) != 0
    }

    fn grow(&mut self, amount: Size, new_state: bool) {
        if amount.bytes() == 0 {
            return;
        }
        let unused_trailing_bits =
            u64::try_from(self.blocks.len()).unwrap() * Self::BLOCK_SIZE - self.len.bytes();
        if amount.bytes() > unused_trailing_bits {
            let additional_blocks = amount.bytes() / Self::BLOCK_SIZE + 1;
            self.blocks.extend(
                // FIXME(oli-obk): optimize this by repeating `new_state as Block`.
                iter::repeat(0).take(usize::try_from(additional_blocks).unwrap()),
            );
        }
        let start = self.len;
        self.len += amount;
        self.set_range_inbounds(start, start + amount, new_state); // `Size` operation
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
            init_mask: &InitMask,
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
                    Some(InitMask::size_from_bit_index(block, bit))
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
            let (start_block, start_bit) = InitMask::bit_index(start);
            let end_inclusive = Size::from_bytes(end.bytes() - 1);
            let (end_block_inclusive, _) = InitMask::bit_index(end_inclusive);

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
            init_mask: &InitMask,
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

        let end_of_chunk =
            self.init_mask.find_bit(self.start, self.end, !self.is_init).unwrap_or(self.end);
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
        // An optimization where we can just overwrite an entire range of initialization
        // bits if they are going to be uniformly `1` or `0`.
        if defined.ranges.len() <= 1 {
            self.set_range_inbounds(
                range.start,
                range.start + range.size * repeat, // `Size` operations
                defined.initial,
            );
            return;
        }

        for mut j in 0..repeat {
            j *= range.size.bytes();
            j += range.start.bytes();
            let mut cur = defined.initial;
            for range in &defined.ranges {
                let old_j = j;
                j += range;
                self.set_range_inbounds(Size::from_bytes(old_j), Size::from_bytes(j), cur);
                cur = !cur;
            }
        }
    }
}
