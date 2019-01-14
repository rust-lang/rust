use super::bitmask::BitMask;
use super::EMPTY;
use core::mem;

#[cfg(target_arch = "x86")]
use core::arch::x86;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86;

pub type BitMaskWord = u16;
pub const BITMASK_STRIDE: usize = 1;
pub const BITMASK_MASK: BitMaskWord = 0xffff;

/// Abstraction over a group of control bytes which can be scanned in
/// parallel.
///
/// This implementation uses a 128-bit SSE value.
#[derive(Copy, Clone)]
pub struct Group(x86::__m128i);

impl Group {
    /// Number of bytes in the group.
    pub const WIDTH: usize = mem::size_of::<Self>();

    /// Returns a full group of empty bytes, suitable for use as the initial
    /// value for an empty hash table.
    ///
    /// This is guaranteed to be aligned to the group size.
    #[inline]
    pub fn static_empty() -> &'static [u8] {
        union AlignedBytes {
            _align: Group,
            bytes: [u8; Group::WIDTH],
        };
        const ALIGNED_BYTES: AlignedBytes = AlignedBytes {
            bytes: [EMPTY; Group::WIDTH],
        };
        unsafe { &ALIGNED_BYTES.bytes }
    }

    /// Loads a group of bytes starting at the given address.
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Group {
        Group(x86::_mm_loadu_si128(ptr as *const _))
    }

    /// Loads a group of bytes starting at the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    pub unsafe fn load_aligned(ptr: *const u8) -> Group {
        debug_assert_eq!(ptr as usize & (mem::align_of::<Group>() - 1), 0);
        Group(x86::_mm_load_si128(ptr as *const _))
    }

    /// Stores the group of bytes to the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    pub unsafe fn store_aligned(&self, ptr: *mut u8) {
        debug_assert_eq!(ptr as usize & (mem::align_of::<Group>() - 1), 0);
        x86::_mm_store_si128(ptr as *mut _, self.0);
    }

    /// Returns a `BitMask` indicating all bytes in the group which have
    /// the given value.
    #[inline]
    pub fn match_byte(&self, byte: u8) -> BitMask {
        unsafe {
            let cmp = x86::_mm_cmpeq_epi8(self.0, x86::_mm_set1_epi8(byte as i8));
            BitMask(x86::_mm_movemask_epi8(cmp) as u16)
        }
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY`.
    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(EMPTY)
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY` or `DELETED`.
    #[inline]
    pub fn match_empty_or_deleted(&self) -> BitMask {
        // A byte is EMPTY or DELETED iff the high bit is set
        unsafe { BitMask(x86::_mm_movemask_epi8(self.0) as u16) }
    }

    /// Performs the following transformation on all bytes in the group:
    /// - `EMPTY => EMPTY`
    /// - `DELETED => EMPTY`
    /// - `FULL => DELETED`
    #[inline]
    pub fn convert_special_to_empty_and_full_to_deleted(&self) -> Group {
        // Map high_bit = 1 (EMPTY or DELETED) to 1111_1111
        // and high_bit = 0 (FULL) to 1000_0000
        //
        // Here's this logic expanded to concrete values:
        //   let special = 0 > byte = 1111_1111 (true) or 0000_0000 (false)
        //   1111_1111 | 1000_0000 = 1111_1111
        //   0000_0000 | 1000_0000 = 1000_0000
        unsafe {
            let zero = x86::_mm_setzero_si128();
            let special = x86::_mm_cmpgt_epi8(zero, self.0);
            Group(x86::_mm_or_si128(special, x86::_mm_set1_epi8(0x80u8 as i8)))
        }
    }
}
