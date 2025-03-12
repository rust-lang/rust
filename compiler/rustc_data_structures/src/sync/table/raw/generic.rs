use super::EMPTY;
use super::bitmask::BitMask;
use core::intrinsics::atomic_load_acq;
use core::mem;

// Use the native word size as the group size. Using a 64-bit group size on
// a 32-bit architecture will just end up being more expensive because
// shifts and multiplies will need to be emulated.
#[cfg(any(
    target_pointer_width = "64",
    target_arch = "aarch64",
    target_arch = "x86_64",
))]
type GroupWord = u64;
#[cfg(all(
    target_pointer_width = "32",
    not(target_arch = "aarch64"),
    not(target_arch = "x86_64"),
))]
type GroupWord = u32;

pub type BitMaskWord = GroupWord;
pub const BITMASK_STRIDE: usize = 8;
// We only care about the highest bit of each byte for the mask.
#[allow(clippy::cast_possible_truncation, clippy::unnecessary_cast)]
pub const BITMASK_MASK: BitMaskWord = 0x8080_8080_8080_8080_u64 as GroupWord;

/// Helper function to replicate a byte across a `GroupWord`.
#[inline]
fn repeat(byte: u8) -> GroupWord {
    GroupWord::from_ne_bytes([byte; Group::WIDTH])
}

/// Abstraction over a group of control bytes which can be scanned in
/// parallel.
///
/// This implementation uses a word-sized integer.
#[derive(Copy, Clone)]
pub struct Group(GroupWord);

// We perform all operations in the native endianess, and convert to
// little-endian just before creating a BitMask. The can potentially
// enable the compiler to eliminate unnecessary byte swaps if we are
// only checking whether a BitMask is empty.
#[allow(clippy::use_self)]
impl Group {
    /// Number of bytes in the group.
    pub const WIDTH: usize = mem::size_of::<Self>();

    /// Returns a full group of empty bytes, suitable for use as the initial
    /// value for an empty hash table.
    ///
    /// This is guaranteed to be aligned to the group size.
    pub const EMPTY: Group = {
        #[repr(C)]
        struct AlignedBytes {
            _align: [Group; 0],
            bytes: [u8; Group::WIDTH],
        }
        const ALIGNED_BYTES: AlignedBytes = AlignedBytes {
            _align: [],
            bytes: [EMPTY; Group::WIDTH],
        };
        unsafe { mem::transmute(ALIGNED_BYTES) }
    };

    /// Loads a group of bytes starting at the given address.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)] // unaligned load
    pub unsafe fn load(ptr: *const u8) -> Self {
        let mut bytes = [0u8; Group::WIDTH];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = atomic_load_acq(ptr.add(i));
        }
        mem::transmute(bytes)
    }

    /// Loads a group of bytes starting at the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub unsafe fn load_aligned(ptr: *const u8) -> Self {
        // FIXME: use align_offset once it stabilizes
        debug_assert_eq!(ptr as usize & (mem::align_of::<Self>() - 1), 0);
        Group::load(ptr)
    }

    /// Returns a `BitMask` indicating all bytes in the group which *may*
    /// have the given value.
    ///
    /// This function may return a false positive in certain cases where
    /// the byte in the group differs from the searched value only in its
    /// lowest bit. This is fine because:
    /// - This never happens for `EMPTY` and `DELETED`, only full entries.
    /// - The check for key equality will catch these.
    /// - This only happens if there is at least 1 true match.
    /// - The chance of this happening is very low (< 1% chance per byte).
    #[inline]
    pub fn match_byte(self, byte: u8) -> BitMask {
        // This algorithm is derived from
        // http://graphics.stanford.edu/~seander/bithacks.html##ValueInWord
        let cmp = self.0 ^ repeat(byte);
        BitMask((cmp.wrapping_sub(repeat(0x01)) & !cmp & repeat(0x80)).to_le())
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY`.
    #[inline]
    pub fn match_empty(self) -> BitMask {
        self.match_byte(EMPTY)
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY` or `DELETED`.
    #[inline]
    pub fn match_empty_or_deleted(self) -> BitMask {
        // A byte is EMPTY or DELETED iff the high bit is set
        BitMask((self.0 & repeat(0x80)).to_le())
    }

    /// Returns a `BitMask` indicating all bytes in the group which are full.
    #[inline]
    pub fn match_full(self) -> BitMask {
        self.match_empty_or_deleted().invert()
    }
}
