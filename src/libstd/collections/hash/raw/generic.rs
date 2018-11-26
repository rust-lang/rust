use super::bitmask::BitMask;
use super::EMPTY;
use core::{mem, ptr};

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
pub const BITMASK_SHIFT: u32 = 3;
pub const BITMASK_MASK: GroupWord = 0x8080808080808080u64 as GroupWord;

/// Helper function to replicate a byte across a `GroupWord`.
#[inline]
fn repeat(byte: u8) -> GroupWord {
    let repeat = byte as GroupWord;
    let repeat = repeat | repeat.wrapping_shl(8);
    let repeat = repeat | repeat.wrapping_shl(16);
    // This last line is a no-op with a 32-bit GroupWord
    repeat | repeat.wrapping_shl(32)
}

/// Abstraction over a group of control bytes which can be scanned in
/// parallel.
///
/// This implementation uses a word-sized integer.
pub struct Group(GroupWord);

// We perform all operations in the native endianess, and convert to
// little-endian just before creating a BitMask. The can potentially
// enable the compiler to eliminate unnecessary byte swaps if we are
// only checking whether a BitMask is empty.
impl Group {
    /// Number of bytes in the group.
    pub const WIDTH: usize = mem::size_of::<Self>();

    /// Returns a full group of empty bytes, suitable for use as the initial
    /// value for an empty hash table.
    ///
    /// This is guaranteed to be aligned to the group size.
    #[inline]
    pub fn static_empty() -> &'static [u8] {
        #[repr(C)]
        struct Dummy {
            _align: [GroupWord; 0],
            bytes: [u8; Group::WIDTH],
        };
        const DUMMY: Dummy = Dummy {
            _align: [],
            bytes: [EMPTY; Group::WIDTH],
        };
        &DUMMY.bytes
    }

    /// Loads a group of bytes starting at the given address.
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Group {
        Group(ptr::read_unaligned(ptr as *const _))
    }

    /// Loads a group of bytes starting at the given address, which must be
    /// aligned to `WIDTH`.
    #[inline]
    pub unsafe fn load_aligned(ptr: *const u8) -> Group {
        Group(ptr::read(ptr as *const _))
    }

    /// Stores the group of bytes to the given address, which must be
    /// aligned to `WIDTH`.
    #[inline]
    pub unsafe fn store_aligned(&self, ptr: *mut u8) {
        ptr::write(ptr as *mut _, self.0);
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
    pub fn match_byte(&self, byte: u8) -> BitMask {
        // This algorithm is derived from
        // http://graphics.stanford.edu/~seander/bithacks.html##ValueInWord
        let cmp = self.0 ^ repeat(byte);
        BitMask((cmp.wrapping_sub(repeat(0x01)) & !cmp & repeat(0x80)).to_le())
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY`.
    #[inline]
    pub fn match_empty(&self) -> BitMask {
        BitMask((self.0 & (self.0 << 1) & repeat(0x80)).to_le())
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY` pr `DELETED`.
    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        BitMask((self.0 & repeat(0x80)).to_le())
    }

    /// Performs the following transformation on all bytes in the group:
    /// - `EMPTY => EMPTY`
    /// - `DELETED => EMPTY`
    /// - `FULL => DELETED`
    #[inline]
    pub fn convert_special_to_empty_and_full_to_deleted(&self) -> Group {
        Group(((self.0 & repeat(0x80)) >> 7) * 0xff)
    }
}
