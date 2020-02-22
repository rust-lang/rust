//! This is a copy of `core::hash::sip` adapted to providing 128 bit hashes.

use std::cmp;
use std::hash::Hasher;
use std::mem;
use std::ptr;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct SipHasher128 {
    k0: u64,
    k1: u64,
    length: usize, // how many bytes we've processed
    state: State,  // hash State
    tail: u64,     // unprocessed bytes le
    ntail: usize,  // how many bytes in tail are valid
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct State {
    // v0, v2 and v1, v3 show up in pairs in the algorithm,
    // and simd implementations of SipHash will use vectors
    // of v02 and v13. By placing them in this order in the struct,
    // the compiler can pick up on just a few simd optimizations by itself.
    v0: u64,
    v2: u64,
    v1: u64,
    v3: u64,
}

macro_rules! compress {
    ($state:expr) => {{ compress!($state.v0, $state.v1, $state.v2, $state.v3) }};
    ($v0:expr, $v1:expr, $v2:expr, $v3:expr) => {{
        $v0 = $v0.wrapping_add($v1);
        $v1 = $v1.rotate_left(13);
        $v1 ^= $v0;
        $v0 = $v0.rotate_left(32);
        $v2 = $v2.wrapping_add($v3);
        $v3 = $v3.rotate_left(16);
        $v3 ^= $v2;
        $v0 = $v0.wrapping_add($v3);
        $v3 = $v3.rotate_left(21);
        $v3 ^= $v0;
        $v2 = $v2.wrapping_add($v1);
        $v1 = $v1.rotate_left(17);
        $v1 ^= $v2;
        $v2 = $v2.rotate_left(32);
    }};
}

/// Loads an integer of the desired type from a byte stream, in LE order. Uses
/// `copy_nonoverlapping` to let the compiler generate the most efficient way
/// to load it from a possibly unaligned address.
///
/// Unsafe because: unchecked indexing at i..i+size_of(int_ty)
macro_rules! load_int_le {
    ($buf:expr, $i:expr, $int_ty:ident) => {{
        debug_assert!($i + mem::size_of::<$int_ty>() <= $buf.len());
        let mut data = 0 as $int_ty;
        ptr::copy_nonoverlapping(
            $buf.get_unchecked($i),
            &mut data as *mut _ as *mut u8,
            mem::size_of::<$int_ty>(),
        );
        data.to_le()
    }};
}

/// Loads a u64 using up to 7 bytes of a byte slice. It looks clumsy but the
/// `copy_nonoverlapping` calls that occur (via `load_int_le!`) all have fixed
/// sizes and avoid calling `memcpy`, which is good for speed.
///
/// Unsafe because: unchecked indexing at start..start+len
#[inline]
unsafe fn u8to64_le(buf: &[u8], start: usize, len: usize) -> u64 {
    debug_assert!(len < 8);
    let mut i = 0; // current byte index (from LSB) in the output u64
    let mut out = 0;
    if i + 3 < len {
        out = load_int_le!(buf, start + i, u32) as u64;
        i += 4;
    }
    if i + 1 < len {
        out |= (load_int_le!(buf, start + i, u16) as u64) << (i * 8);
        i += 2
    }
    if i < len {
        out |= (*buf.get_unchecked(start + i) as u64) << (i * 8);
        i += 1;
    }
    debug_assert_eq!(i, len);
    out
}

impl SipHasher128 {
    #[inline]
    pub fn new_with_keys(key0: u64, key1: u64) -> SipHasher128 {
        let mut state = SipHasher128 {
            k0: key0,
            k1: key1,
            length: 0,
            state: State { v0: 0, v1: 0, v2: 0, v3: 0 },
            tail: 0,
            ntail: 0,
        };
        state.reset();
        state
    }

    #[inline]
    fn reset(&mut self) {
        self.length = 0;
        self.state.v0 = self.k0 ^ 0x736f6d6570736575;
        self.state.v1 = self.k1 ^ 0x646f72616e646f6d;
        self.state.v2 = self.k0 ^ 0x6c7967656e657261;
        self.state.v3 = self.k1 ^ 0x7465646279746573;
        self.ntail = 0;

        // This is only done in the 128 bit version:
        self.state.v1 ^= 0xee;
    }

    // A specialized write function for values with size <= 8.
    //
    // The hashing of multi-byte integers depends on endianness. E.g.:
    // - little-endian: `write_u32(0xDDCCBBAA)` == `write([0xAA, 0xBB, 0xCC, 0xDD])`
    // - big-endian:    `write_u32(0xDDCCBBAA)` == `write([0xDD, 0xCC, 0xBB, 0xAA])`
    //
    // This function does the right thing for little-endian hardware. On
    // big-endian hardware `x` must be byte-swapped first to give the right
    // behaviour. After any byte-swapping, the input must be zero-extended to
    // 64-bits. The caller is responsible for the byte-swapping and
    // zero-extension.
    #[inline]
    fn short_write<T>(&mut self, _x: T, x: u64) {
        let size = mem::size_of::<T>();
        self.length += size;

        // The original number must be zero-extended, not sign-extended.
        debug_assert!(if size < 8 { x >> (8 * size) == 0 } else { true });

        // The number of bytes needed to fill `self.tail`.
        let needed = 8 - self.ntail;

        // SipHash parses the input stream as 8-byte little-endian integers.
        // Inputs are put into `self.tail` until 8 bytes of data have been
        // collected, and then that word is processed.
        //
        // For example, imagine that `self.tail` is 0x0000_00EE_DDCC_BBAA,
        // `self.ntail` is 5 (because 5 bytes have been put into `self.tail`),
        // and `needed` is therefore 3.
        //
        // - Scenario 1, `self.write_u8(0xFF)`: we have already zero-extended
        //   the input to 0x0000_0000_0000_00FF. We now left-shift it five
        //   bytes, giving 0x0000_FF00_0000_0000. We then bitwise-OR that value
        //   into `self.tail`, resulting in 0x0000_FFEE_DDCC_BBAA.
        //   (Zero-extension of the original input is critical in this scenario
        //   because we don't want the high two bytes of `self.tail` to be
        //   touched by the bitwise-OR.) `self.tail` is not yet full, so we
        //   return early, after updating `self.ntail` to 6.
        //
        // - Scenario 2, `self.write_u32(0xIIHH_GGFF)`: we have already
        //   zero-extended the input to 0x0000_0000_IIHH_GGFF. We now
        //   left-shift it five bytes, giving 0xHHGG_FF00_0000_0000. We then
        //   bitwise-OR that value into `self.tail`, resulting in
        //   0xHHGG_FFEE_DDCC_BBAA. `self.tail` is now full, and we can use it
        //   to update `self.state`. (As mentioned above, this assumes a
        //   little-endian machine; on a big-endian machine we would have
        //   byte-swapped 0xIIHH_GGFF in the caller, giving 0xFFGG_HHII, and we
        //   would then end up bitwise-ORing 0xGGHH_II00_0000_0000 into
        //   `self.tail`).
        //
        self.tail |= x << (8 * self.ntail);
        if size < needed {
            self.ntail += size;
            return;
        }

        // `self.tail` is full, process it.
        self.state.v3 ^= self.tail;
        Sip24Rounds::c_rounds(&mut self.state);
        self.state.v0 ^= self.tail;

        // Continuing scenario 2: we have one byte left over from the input. We
        // set `self.ntail` to 1 and `self.tail` to `0x0000_0000_IIHH_GGFF >>
        // 8*3`, which is 0x0000_0000_0000_00II. (Or on a big-endian machine
        // the prior byte-swapping would leave us with 0x0000_0000_0000_00FF.)
        //
        // The `if` is needed to avoid shifting by 64 bits, which Rust
        // complains about.
        self.ntail = size - needed;
        self.tail = if needed < 8 { x >> (8 * needed) } else { 0 };
    }

    #[inline]
    pub fn finish128(mut self) -> (u64, u64) {
        let b: u64 = ((self.length as u64 & 0xff) << 56) | self.tail;

        self.state.v3 ^= b;
        Sip24Rounds::c_rounds(&mut self.state);
        self.state.v0 ^= b;

        self.state.v2 ^= 0xee;
        Sip24Rounds::d_rounds(&mut self.state);
        let _0 = self.state.v0 ^ self.state.v1 ^ self.state.v2 ^ self.state.v3;

        self.state.v1 ^= 0xdd;
        Sip24Rounds::d_rounds(&mut self.state);
        let _1 = self.state.v0 ^ self.state.v1 ^ self.state.v2 ^ self.state.v3;
        (_0, _1)
    }
}

impl Hasher for SipHasher128 {
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.short_write(i, i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.short_write(i, i.to_le() as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.short_write(i, i.to_le() as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.short_write(i, i.to_le() as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.short_write(i, i.to_le() as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.short_write(i, i as u8 as u64);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.short_write(i, (i as u16).to_le() as u64);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.short_write(i, (i as u32).to_le() as u64);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.short_write(i, (i as u64).to_le() as u64);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.short_write(i, (i as usize).to_le() as u64);
    }

    #[inline]
    fn write(&mut self, msg: &[u8]) {
        let length = msg.len();
        self.length += length;

        let mut needed = 0;

        if self.ntail != 0 {
            needed = 8 - self.ntail;
            self.tail |= unsafe { u8to64_le(msg, 0, cmp::min(length, needed)) } << 8 * self.ntail;
            if length < needed {
                self.ntail += length;
                return;
            } else {
                self.state.v3 ^= self.tail;
                Sip24Rounds::c_rounds(&mut self.state);
                self.state.v0 ^= self.tail;
                self.ntail = 0;
            }
        }

        // Buffered tail is now flushed, process new input.
        let len = length - needed;
        let left = len & 0x7;

        let mut i = needed;
        while i < len - left {
            let mi = unsafe { load_int_le!(msg, i, u64) };

            self.state.v3 ^= mi;
            Sip24Rounds::c_rounds(&mut self.state);
            self.state.v0 ^= mi;

            i += 8;
        }

        self.tail = unsafe { u8to64_le(msg, i, left) };
        self.ntail = left;
    }

    fn finish(&self) -> u64 {
        panic!("SipHasher128 cannot provide valid 64 bit hashes")
    }
}

#[derive(Debug, Clone, Default)]
struct Sip24Rounds;

impl Sip24Rounds {
    #[inline]
    fn c_rounds(state: &mut State) {
        compress!(state);
        compress!(state);
    }

    #[inline]
    fn d_rounds(state: &mut State) {
        compress!(state);
        compress!(state);
        compress!(state);
        compress!(state);
    }
}
