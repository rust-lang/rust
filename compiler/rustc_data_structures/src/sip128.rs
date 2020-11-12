//! This is a copy of `core::hash::sip` adapted to providing 128 bit hashes.

use std::hash::Hasher;
use std::mem::{self, MaybeUninit};
use std::ptr;

#[cfg(test)]
mod tests;

// The SipHash algorithm operates on 8-byte chunks.
const ELEM_SIZE: usize = mem::size_of::<u64>();

// Size of the buffer in number of elements, not including the spill.
//
// The selection of this size was guided by rustc-perf benchmark comparisons of
// different buffer sizes. It should be periodically reevaluated as the compiler
// implementation and input characteristics change.
//
// Using the same-sized buffer for everything we hash is a performance versus
// complexity tradeoff. The ideal buffer size, and whether buffering should even
// be used, depends on what is being hashed. It may be worth it to size the
// buffer appropriately (perhaps by making SipHasher128 generic over the buffer
// size) or disable buffering depending on what is being hashed. But at this
// time, we use the same buffer size for everything.
const BUFFER_CAPACITY: usize = 8;

// Size of the buffer in bytes, not including the spill.
const BUFFER_SIZE: usize = BUFFER_CAPACITY * ELEM_SIZE;

// Size of the buffer in number of elements, including the spill.
const BUFFER_WITH_SPILL_CAPACITY: usize = BUFFER_CAPACITY + 1;

// Size of the buffer in bytes, including the spill.
const BUFFER_WITH_SPILL_SIZE: usize = BUFFER_WITH_SPILL_CAPACITY * ELEM_SIZE;

// Index of the spill element in the buffer.
const BUFFER_SPILL_INDEX: usize = BUFFER_WITH_SPILL_CAPACITY - 1;

#[derive(Debug, Clone)]
#[repr(C)]
pub struct SipHasher128 {
    // The access pattern during hashing consists of accesses to `nbuf` and
    // `buf` until the buffer is full, followed by accesses to `state` and
    // `processed`, and then repetition of that pattern until hashing is done.
    // This is the basis for the ordering of fields below. However, in practice
    // the cache miss-rate for data access is extremely low regardless of order.
    nbuf: usize, // how many bytes in buf are valid
    buf: [MaybeUninit<u64>; BUFFER_WITH_SPILL_CAPACITY], // unprocessed bytes le
    state: State, // hash State
    processed: usize, // how many bytes we've processed
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

// Copies up to 8 bytes from source to destination. This performs better than
// `ptr::copy_nonoverlapping` on microbenchmarks and may perform better on real
// workloads since all of the copies have fixed sizes and avoid calling memcpy.
//
// This is specifically designed for copies of up to 8 bytes, because that's the
// maximum of number bytes needed to fill an 8-byte-sized element on which
// SipHash operates. Note that for variable-sized copies which are known to be
// less than 8 bytes, this function will perform more work than necessary unless
// the compiler is able to optimize the extra work away.
#[inline]
unsafe fn copy_nonoverlapping_small(src: *const u8, dst: *mut u8, count: usize) {
    debug_assert!(count <= 8);

    if count == 8 {
        ptr::copy_nonoverlapping(src, dst, 8);
        return;
    }

    let mut i = 0;
    if i + 3 < count {
        ptr::copy_nonoverlapping(src.add(i), dst.add(i), 4);
        i += 4;
    }

    if i + 1 < count {
        ptr::copy_nonoverlapping(src.add(i), dst.add(i), 2);
        i += 2
    }

    if i < count {
        *dst.add(i) = *src.add(i);
        i += 1;
    }

    debug_assert_eq!(i, count);
}

// # Implementation
//
// This implementation uses buffering to reduce the hashing cost for inputs
// consisting of many small integers. Buffering simplifies the integration of
// integer input--the integer write function typically just appends to the
// buffer with a statically sized write, updates metadata, and returns.
//
// Buffering also prevents alternating between writes that do and do not trigger
// the hashing process. Only when the entire buffer is full do we transition
// into hashing. This allows us to keep the hash state in registers for longer,
// instead of loading and storing it before and after processing each element.
//
// When a write fills the buffer, a buffer processing function is invoked to
// hash all of the buffered input. The buffer processing functions are marked
// `#[inline(never)]` so that they aren't inlined into the append functions,
// which ensures the more frequently called append functions remain inlineable
// and don't include register pushing/popping that would only be made necessary
// by inclusion of the complex buffer processing path which uses those
// registers.
//
// The buffer includes a "spill"--an extra element at the end--which simplifies
// the integer write buffer processing path. The value that fills the buffer can
// be written with a statically sized write that may spill over into the spill.
// After the buffer is processed, the part of the value that spilled over can be
// written from the spill to the beginning of the buffer with another statically
// sized write. This write may copy more bytes than actually spilled over, but
// we maintain the metadata such that any extra copied bytes will be ignored by
// subsequent processing. Due to the static sizes, this scheme performs better
// than copying the exact number of bytes needed into the end and beginning of
// the buffer.
//
// The buffer is uninitialized, which improves performance, but may preclude
// efficient implementation of alternative approaches. The improvement is not so
// large that an alternative approach should be disregarded because it cannot be
// efficiently implemented with an uninitialized buffer. On the other hand, an
// uninitialized buffer may become more important should a larger one be used.
//
// # Platform Dependence
//
// The SipHash algorithm operates on byte sequences. It parses the input stream
// as 8-byte little-endian integers. Therefore, given the same byte sequence, it
// produces the same result on big- and little-endian hardware.
//
// However, the Hasher trait has methods which operate on multi-byte integers.
// How they are converted into byte sequences can be endian-dependent (by using
// native byte order) or independent (by consistently using either LE or BE byte
// order). It can also be `isize` and `usize` size dependent (by using the
// native size), or independent (by converting to a common size), supposing the
// values can be represented in 32 bits.
//
// In order to make `SipHasher128` consistent with `SipHasher` in libstd, we
// choose to do the integer to byte sequence conversion in the platform-
// dependent way. Clients can achieve platform-independent hashing by widening
// `isize` and `usize` integers to 64 bits on 32-bit systems and byte-swapping
// integers on big-endian systems before passing them to the writing functions.
// This causes the input byte sequence to look identical on big- and little-
// endian systems (supposing `isize` and `usize` values can be represented in 32
// bits), which ensures platform-independent results.
impl SipHasher128 {
    #[inline]
    pub fn new_with_keys(key0: u64, key1: u64) -> SipHasher128 {
        let mut hasher = SipHasher128 {
            nbuf: 0,
            buf: MaybeUninit::uninit_array(),
            state: State {
                v0: key0 ^ 0x736f6d6570736575,
                // The XOR with 0xee is only done on 128-bit algorithm version.
                v1: key1 ^ (0x646f72616e646f6d ^ 0xee),
                v2: key0 ^ 0x6c7967656e657261,
                v3: key1 ^ 0x7465646279746573,
            },
            processed: 0,
        };

        unsafe {
            // Initialize spill because we read from it in `short_write_process_buffer`.
            *hasher.buf.get_unchecked_mut(BUFFER_SPILL_INDEX) = MaybeUninit::zeroed();
        }

        hasher
    }

    // A specialized write function for values with size <= 8.
    #[inline]
    fn short_write<T>(&mut self, x: T) {
        let size = mem::size_of::<T>();
        let nbuf = self.nbuf;
        debug_assert!(size <= 8);
        debug_assert!(nbuf < BUFFER_SIZE);
        debug_assert!(nbuf + size < BUFFER_WITH_SPILL_SIZE);

        if nbuf + size < BUFFER_SIZE {
            unsafe {
                // The memcpy call is optimized away because the size is known.
                let dst = (self.buf.as_mut_ptr() as *mut u8).add(nbuf);
                ptr::copy_nonoverlapping(&x as *const _ as *const u8, dst, size);
            }

            self.nbuf = nbuf + size;

            return;
        }

        unsafe { self.short_write_process_buffer(x) }
    }

    // A specialized write function for values with size <= 8 that should only
    // be called when the write would cause the buffer to fill.
    //
    // SAFETY: the write of `x` into `self.buf` starting at byte offset
    // `self.nbuf` must cause `self.buf` to become fully initialized (and not
    // overflow) if it wasn't already.
    #[inline(never)]
    unsafe fn short_write_process_buffer<T>(&mut self, x: T) {
        let size = mem::size_of::<T>();
        let nbuf = self.nbuf;
        debug_assert!(size <= 8);
        debug_assert!(nbuf < BUFFER_SIZE);
        debug_assert!(nbuf + size >= BUFFER_SIZE);
        debug_assert!(nbuf + size < BUFFER_WITH_SPILL_SIZE);

        // Copy first part of input into end of buffer, possibly into spill
        // element. The memcpy call is optimized away because the size is known.
        let dst = (self.buf.as_mut_ptr() as *mut u8).add(nbuf);
        ptr::copy_nonoverlapping(&x as *const _ as *const u8, dst, size);

        // Process buffer.
        for i in 0..BUFFER_CAPACITY {
            let elem = self.buf.get_unchecked(i).assume_init().to_le();
            self.state.v3 ^= elem;
            Sip24Rounds::c_rounds(&mut self.state);
            self.state.v0 ^= elem;
        }

        // Copy remaining input into start of buffer by copying size - 1
        // elements from spill (at most size - 1 bytes could have overflowed
        // into the spill). The memcpy call is optimized away because the size
        // is known. And the whole copy is optimized away for size == 1.
        let src = self.buf.get_unchecked(BUFFER_SPILL_INDEX) as *const _ as *const u8;
        ptr::copy_nonoverlapping(src, self.buf.as_mut_ptr() as *mut u8, size - 1);

        // This function should only be called when the write fills the buffer.
        // Therefore, when size == 1, the new `self.nbuf` must be zero. The size
        // is statically known, so the branch is optimized away.
        self.nbuf = if size == 1 { 0 } else { nbuf + size - BUFFER_SIZE };
        self.processed += BUFFER_SIZE;
    }

    // A write function for byte slices.
    #[inline]
    fn slice_write(&mut self, msg: &[u8]) {
        let length = msg.len();
        let nbuf = self.nbuf;
        debug_assert!(nbuf < BUFFER_SIZE);

        if nbuf + length < BUFFER_SIZE {
            unsafe {
                let dst = (self.buf.as_mut_ptr() as *mut u8).add(nbuf);

                if length <= 8 {
                    copy_nonoverlapping_small(msg.as_ptr(), dst, length);
                } else {
                    // This memcpy is *not* optimized away.
                    ptr::copy_nonoverlapping(msg.as_ptr(), dst, length);
                }
            }

            self.nbuf = nbuf + length;

            return;
        }

        unsafe { self.slice_write_process_buffer(msg) }
    }

    // A write function for byte slices that should only be called when the
    // write would cause the buffer to fill.
    //
    // SAFETY: `self.buf` must be initialized up to the byte offset `self.nbuf`,
    // and `msg` must contain enough bytes to initialize the rest of the element
    // containing the byte offset `self.nbuf`.
    #[inline(never)]
    unsafe fn slice_write_process_buffer(&mut self, msg: &[u8]) {
        let length = msg.len();
        let nbuf = self.nbuf;
        debug_assert!(nbuf < BUFFER_SIZE);
        debug_assert!(nbuf + length >= BUFFER_SIZE);

        // Always copy first part of input into current element of buffer.
        // This function should only be called when the write fills the buffer,
        // so we know that there is enough input to fill the current element.
        let valid_in_elem = nbuf % ELEM_SIZE;
        let needed_in_elem = ELEM_SIZE - valid_in_elem;

        let src = msg.as_ptr();
        let dst = (self.buf.as_mut_ptr() as *mut u8).add(nbuf);
        copy_nonoverlapping_small(src, dst, needed_in_elem);

        // Process buffer.

        // Using `nbuf / ELEM_SIZE + 1` rather than `(nbuf + needed_in_elem) /
        // ELEM_SIZE` to show the compiler that this loop's upper bound is > 0.
        // We know that is true, because last step ensured we have a full
        // element in the buffer.
        let last = nbuf / ELEM_SIZE + 1;

        for i in 0..last {
            let elem = self.buf.get_unchecked(i).assume_init().to_le();
            self.state.v3 ^= elem;
            Sip24Rounds::c_rounds(&mut self.state);
            self.state.v0 ^= elem;
        }

        // Process the remaining element-sized chunks of input.
        let mut processed = needed_in_elem;
        let input_left = length - processed;
        let elems_left = input_left / ELEM_SIZE;
        let extra_bytes_left = input_left % ELEM_SIZE;

        for _ in 0..elems_left {
            let elem = (msg.as_ptr().add(processed) as *const u64).read_unaligned().to_le();
            self.state.v3 ^= elem;
            Sip24Rounds::c_rounds(&mut self.state);
            self.state.v0 ^= elem;
            processed += ELEM_SIZE;
        }

        // Copy remaining input into start of buffer.
        let src = msg.as_ptr().add(processed);
        let dst = self.buf.as_mut_ptr() as *mut u8;
        copy_nonoverlapping_small(src, dst, extra_bytes_left);

        self.nbuf = extra_bytes_left;
        self.processed += nbuf + processed;
    }

    #[inline]
    pub fn finish128(mut self) -> (u64, u64) {
        debug_assert!(self.nbuf < BUFFER_SIZE);

        // Process full elements in buffer.
        let last = self.nbuf / ELEM_SIZE;

        // Since we're consuming self, avoid updating members for a potential
        // performance gain.
        let mut state = self.state;

        for i in 0..last {
            let elem = unsafe { self.buf.get_unchecked(i).assume_init().to_le() };
            state.v3 ^= elem;
            Sip24Rounds::c_rounds(&mut state);
            state.v0 ^= elem;
        }

        // Get remaining partial element.
        let elem = if self.nbuf % ELEM_SIZE != 0 {
            unsafe {
                // Ensure element is initialized by writing zero bytes. At most
                // `ELEM_SIZE - 1` are required given the above check. It's safe
                // to write this many because we have the spill and we maintain
                // `self.nbuf` such that this write will start before the spill.
                let dst = (self.buf.as_mut_ptr() as *mut u8).add(self.nbuf);
                ptr::write_bytes(dst, 0, ELEM_SIZE - 1);
                self.buf.get_unchecked(last).assume_init().to_le()
            }
        } else {
            0
        };

        // Finalize the hash.
        let length = self.processed + self.nbuf;
        let b: u64 = ((length as u64 & 0xff) << 56) | elem;

        state.v3 ^= b;
        Sip24Rounds::c_rounds(&mut state);
        state.v0 ^= b;

        state.v2 ^= 0xee;
        Sip24Rounds::d_rounds(&mut state);
        let _0 = state.v0 ^ state.v1 ^ state.v2 ^ state.v3;

        state.v1 ^= 0xdd;
        Sip24Rounds::d_rounds(&mut state);
        let _1 = state.v0 ^ state.v1 ^ state.v2 ^ state.v3;

        (_0, _1)
    }
}

impl Hasher for SipHasher128 {
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.short_write(i);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.short_write(i);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.short_write(i);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.short_write(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.short_write(i);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.short_write(i as u8);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.short_write(i as u16);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.short_write(i as u32);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.short_write(i as u64);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.short_write(i as usize);
    }

    #[inline]
    fn write(&mut self, msg: &[u8]) {
        self.slice_write(msg);
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
