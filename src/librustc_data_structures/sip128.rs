//! This is a copy of `core::hash::sip` adapted to providing 128 bit hashes.

use std::cmp;
use std::hash::Hasher;
use std::slice;
use std::ptr;
use std::mem;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct SipHasher128 {
    k0: u64,
    k1: u64,
    length: usize, // how many bytes we've processed
    state: State, // hash State
    tail: u64, // unprocessed bytes le
    ntail: usize, // how many bytes in tail are valid
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
    ($state:expr) => ({
        compress!($state.v0, $state.v1, $state.v2, $state.v3)
    });
    ($v0:expr, $v1:expr, $v2:expr, $v3:expr) =>
    ({
        $v0 = $v0.wrapping_add($v1); $v1 = $v1.rotate_left(13); $v1 ^= $v0;
        $v0 = $v0.rotate_left(32);
        $v2 = $v2.wrapping_add($v3); $v3 = $v3.rotate_left(16); $v3 ^= $v2;
        $v0 = $v0.wrapping_add($v3); $v3 = $v3.rotate_left(21); $v3 ^= $v0;
        $v2 = $v2.wrapping_add($v1); $v1 = $v1.rotate_left(17); $v1 ^= $v2;
        $v2 = $v2.rotate_left(32);
    });
}

/// Loads an integer of the desired type from a byte stream, in LE order. Uses
/// `copy_nonoverlapping` to let the compiler generate the most efficient way
/// to load it from a possibly unaligned address.
///
/// Unsafe because: unchecked indexing at i..i+size_of(int_ty)
macro_rules! load_int_le {
    ($buf:expr, $i:expr, $int_ty:ident) =>
    ({
       debug_assert!($i + mem::size_of::<$int_ty>() <= $buf.len());
       let mut data = 0 as $int_ty;
       ptr::copy_nonoverlapping($buf.get_unchecked($i),
                                &mut data as *mut _ as *mut u8,
                                mem::size_of::<$int_ty>());
       data.to_le()
    });
}

/// Loads an u64 using up to 7 bytes of a byte slice.
///
/// Unsafe because: unchecked indexing at start..start+len
#[inline]
unsafe fn u8to64_le(buf: &[u8], start: usize, len: usize) -> u64 {
    debug_assert!(len < 8);
    let mut i = 0; // current byte index (from LSB) in the output u64
    let mut out = 0;
    if i + 3 < len {
        out = u64::from(load_int_le!(buf, start + i, u32));
        i += 4;
    }
    if i + 1 < len {
        out |= u64::from(load_int_le!(buf, start + i, u16)) << (i * 8);
        i += 2
    }
    if i < len {
        out |= u64::from(*buf.get_unchecked(start + i)) << (i * 8);
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
            state: State {
                v0: 0,
                v1: 0,
                v2: 0,
                v3: 0,
            },
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

    // Specialized write function that is only valid for buffers with len <= 8.
    // It's used to force inlining of write_u8 and write_usize, those would normally be inlined
    // except for composite types (that includes slices and str hashing because of delimiter).
    // Without this extra push the compiler is very reluctant to inline delimiter writes,
    // degrading performance substantially for the most common use cases.
    #[inline]
    fn short_write(&mut self, msg: &[u8]) {
        debug_assert!(msg.len() <= 8);
        let length = msg.len();
        self.length += length;

        let needed = 8 - self.ntail;
        let fill = cmp::min(length, needed);
        if fill == 8 {
            self.tail = unsafe { load_int_le!(msg, 0, u64) };
        } else {
            self.tail |= unsafe { u8to64_le(msg, 0, fill) } << (8 * self.ntail);
            if length < needed {
                self.ntail += length;
                return;
            }
        }
        self.state.v3 ^= self.tail;
        Sip24Rounds::c_rounds(&mut self.state);
        self.state.v0 ^= self.tail;

        // Buffered tail is now flushed, process new input.
        self.ntail = length - needed;
        self.tail = unsafe { u8to64_le(msg, needed, self.ntail) };
    }

    #[inline(always)]
    fn short_write_gen<T>(&mut self, x: T) {
        let bytes = unsafe {
            slice::from_raw_parts(&x as *const T as *const u8, mem::size_of::<T>())
        };
        self.short_write(bytes);
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
        self.short_write_gen(i);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.short_write_gen(i);
    }

    #[inline]
    fn write(&mut self, msg: &[u8]) {
        let length = msg.len();
        self.length += length;

        let mut needed = 0;

        if self.ntail != 0 {
            needed = 8 - self.ntail;
            self.tail |= unsafe { u8to64_le(msg, 0, cmp::min(length, needed)) } << (8 * self.ntail);
            if length < needed {
                self.ntail += length;
                return
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
