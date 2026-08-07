// Some parts taken from rust-memchr.
// Copyright 2015 Andrew Gallant, bluss and Nicolas Koch

#![forbid(unsafe_op_in_unsafe_fn)]

use core::ffi::{c_int, c_void};
use core::ptr;

unsafe fn memchr_naive(
    mut current: *const u8,
    needle: u8,
    mut n: usize,
) -> *mut c_void {
    while n > 0 {
        let byte = unsafe { current.read() };
        if byte == needle {
            return current as *mut c_void;
        }

        current = unsafe { current.add(1) };
        n -= 1;
    }

    ptr::null_mut()
}

type Chunk = cfg_select! {
    any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon"),
    ) => core::simd::u8x16,
    _ => [usize; 2],
};

pub unsafe fn memchr(
    s: *const c_void,
    c: c_int,
    mut n: usize,
) -> *mut c_void {
    let mut current = s.cast::<u8>();
    let needle = c as u8;

    if n < size_of::<Chunk>() {
        return unsafe { memchr_naive(current, needle, n) };
    }

    // Advance until the pointer is aligned to a chunk.
    // Since the size of a chunk must be larger than its alignment and we
    // only reach this point if `n` is larger or equal to the size of a chunk,
    // this doesn't need bound checks.
    while !current.cast::<Chunk>().is_aligned() {
        let byte = unsafe { current.read() };
        if byte == needle {
            return current as *mut c_void;
        }

        current = unsafe { current.add(1) };
        n -= 1;
    }

    cfg_select! {
        // These targets have efficient SIMD acceleration.
        any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            all(target_arch = "aarch64", target_feature = "neon"),
        ) => {
            use core::simd::cmp::SimdPartialEq;

            let mut current = current.cast::<Chunk>();
            let chunk_needle = Chunk::splat(needle);
            while n >= size_of::<Chunk>() {
                let chunk = unsafe { current.read() };
                if let Some(index) = chunk.simd_eq(chunk_needle).first_set() {
                    return unsafe { current.cast::<u8>().add(index) as *mut c_void }
                }

                current = unsafe { current.add(1) };
                n -= size_of::<Chunk>();
            }
        }
        // Unfortunately LLVM is not smart enough to use SWAR (SIMD-within-a-register)
        // techniques if native SIMD is not available, so we need to do SWAR manually.
        _ => {
            const LOW_BITS: usize = usize::from_ne_bytes([0x01; _]);
            const HIGH_BITS: usize = usize::from_ne_bytes([0x80; _]);

            /// Returns `true` if `x` contains any zero byte.
            ///
            /// From *Matters Computational*, J. Arndt:
            ///
            /// "The idea is to subtract one from each of the bytes and then look for
            /// bytes where the borrow propagated all the way to the most significant
            /// bit."
            #[inline]
            const fn contains_zero_byte(x: usize) -> bool {
                x.wrapping_sub(LOW_BITS) & !x & HIGH_BITS != 0
            }

            let mut current = current.cast::<Chunk>();
            // Since `needle` is 8-bit wide, this multiplication will splat those
            // 8 bits over all bytes of `usize`.
            let chunk_needle = LOW_BITS * needle as usize;
            while n >= size_of::<Chunk>() {
                // Compare two words in one go.
                let [lower, upper] = unsafe { current.read() };
                // If the byte matches, then the XOR will result in that byte
                // being zero.
                let a = contains_zero_byte(lower ^ chunk_needle);
                let b = contains_zero_byte(upper ^ chunk_needle);
                if a | b {
                    // Find the matching byte. The loop doesn't need to be bounded
                    // since we know there is a match.
                    let mut current = current.cast::<u8>();
                    loop {
                        let byte = unsafe { current.read() };
                        if byte == needle {
                            return current as *mut c_void;
                        }

                        current = unsafe { current.add(1) };
                    }
                }

                current = unsafe { current.add(1) };
                n -= size_of::<Chunk>();
            }
        }
    }

    // Search in the remaining bytes.
    let current = current.cast::<u8>();
    unsafe { memchr_naive(current, needle, n) }
}
