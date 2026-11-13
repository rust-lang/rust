//! Code for efficiently counting the number of `char`s in a UTF-8 encoded
//! string.
//!
//! Broadly, UTF-8 encodes `char`s as a "leading" byte which begins the `char`,
//! followed by some number (possibly 0) of continuation bytes.
//!
//! The leading byte can have a number of bit-patterns (with the specific
//! pattern indicating how many continuation bytes follow), but the continuation
//! bytes are always in the format `0b10XX_XXXX` (where the `X`s can take any
//! value). That is, the most significant bit is set, and the second most
//! significant bit is unset.
//!
//! To count the number of characters, we can just count the number of bytes in
//! the string which are not continuation bytes, which can be done many bytes at
//! a time fairly easily.
//!
//! Note: Because the term "leading byte" can sometimes be ambiguous (for
//! example, it could also refer to the first byte of a slice), we'll often use
//! the term "non-continuation byte" to refer to these bytes in the code.

use core::intrinsics::unlikely;

const USIZE_SIZE: usize = size_of::<usize>();
const UNROLL_INNER: usize = 4;

#[inline]
pub(super) fn count_chars(s: &str) -> usize {
    if cfg!(feature = "optimize_for_size") || s.len() < USIZE_SIZE * UNROLL_INNER {
        // Avoid entering the optimized implementation for strings where the
        // difference is not likely to matter, or where it might even be slower.
        // That said, a ton of thought was not spent on the particular threshold
        // here, beyond "this value seems to make sense".
        char_count_general_case(s.as_bytes())
    } else {
        do_count_chars(s)
    }
}

fn do_count_chars(s: &str) -> usize {
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    if s.len() >= SIMD_MIN_LEN {
        return char_count_simd(s);
    }
    // For correctness, `CHUNK_SIZE` must be:
    //
    // - Less than or equal to 255, otherwise we'll overflow bytes in `counts`.
    // - A multiple of `UNROLL_INNER`, otherwise our `break` inside the
    //   `body.chunks(CHUNK_SIZE)` loop is incorrect.
    //
    // For performance, `CHUNK_SIZE` should be:
    // - Relatively cheap to `/` against (so some simple sum of powers of two).
    // - Large enough to avoid paying for the cost of the `sum_bytes_in_usize`
    //   too often.
    const CHUNK_SIZE: usize = 192;

    // Check the properties of `CHUNK_SIZE` and `UNROLL_INNER` that are required
    // for correctness.
    const _: () = assert!(CHUNK_SIZE < 256);
    const _: () = assert!(CHUNK_SIZE.is_multiple_of(UNROLL_INNER));

    // SAFETY: transmuting `[u8]` to `[usize]` is safe except for size
    // differences which are handled by `align_to`.
    let (head, body, tail) = unsafe { s.as_bytes().align_to::<usize>() };

    // This should be quite rare, and basically exists to handle the degenerate
    // cases where align_to fails (as well as miri under symbolic alignment
    // mode).
    //
    // The `unlikely` helps discourage LLVM from inlining the body, which is
    // nice, as we would rather not mark the `char_count_general_case` function
    // as cold.
    if unlikely(body.is_empty() || head.len() > USIZE_SIZE || tail.len() > USIZE_SIZE) {
        return char_count_general_case(s.as_bytes());
    }

    let mut total = char_count_general_case(head) + char_count_general_case(tail);
    // Split `body` into `CHUNK_SIZE` chunks to reduce the frequency with which
    // we call `sum_bytes_in_usize`.
    for chunk in body.chunks(CHUNK_SIZE) {
        // We accumulate intermediate sums in `counts`, where each byte contains
        // a subset of the sum of this chunk, like a `[u8; size_of::<usize>()]`.
        let mut counts = 0;

        let (unrolled_chunks, remainder) = chunk.as_chunks::<UNROLL_INNER>();
        for unrolled in unrolled_chunks {
            for &word in unrolled {
                // Because `CHUNK_SIZE` is < 256, this addition can't cause the
                // count in any of the bytes to overflow into a subsequent byte.
                counts += contains_non_continuation_byte(word);
            }
        }

        // Sum the values in `counts` (which, again, is conceptually a `[u8;
        // size_of::<usize>()]`), and accumulate the result into `total`.
        total += sum_bytes_in_usize(counts);

        // If there's any data in `remainder`, then handle it. This will only
        // happen for the last `chunk` in `body.chunks()` (because `CHUNK_SIZE`
        // is divisible by `UNROLL_INNER`), so we explicitly break at the end
        // (which seems to help LLVM out).
        if !remainder.is_empty() {
            // Accumulate all the data in the remainder.
            let mut counts = 0;
            for &word in remainder {
                counts += contains_non_continuation_byte(word);
            }
            total += sum_bytes_in_usize(counts);
            break;
        }
    }
    total
}

// Checks each byte of `w` to see if it contains the first byte in a UTF-8
// sequence. Bytes in `w` which are continuation bytes are left as `0x00` (e.g.
// false), and bytes which are non-continuation bytes are left as `0x01` (e.g.
// true)
#[inline]
fn contains_non_continuation_byte(w: usize) -> usize {
    const LSB: usize = usize::repeat_u8(0x01);
    ((!w >> 7) | (w >> 6)) & LSB
}

// Morally equivalent to `values.to_ne_bytes().into_iter().sum::<usize>()`, but
// more efficient.
#[inline]
fn sum_bytes_in_usize(values: usize) -> usize {
    const LSB_SHORTS: usize = usize::repeat_u16(0x0001);
    const SKIP_BYTES: usize = usize::repeat_u16(0x00ff);

    let pair_sum: usize = (values & SKIP_BYTES) + ((values >> 8) & SKIP_BYTES);
    pair_sum.wrapping_mul(LSB_SHORTS) >> ((USIZE_SIZE - 2) * 8)
}

// This is the most direct implementation of the concept of "count the number of
// bytes in the string which are not continuation bytes", and is used for the
// head and tail of the input string (the first and last item in the tuple
// returned by `slice::align_to`).
fn char_count_general_case(s: &[u8]) -> usize {
    s.iter().filter(|&&byte| !super::validations::utf8_is_cont_byte(byte)).count()
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_feature = "neon")
))]
const SIMD_MIN_LEN: usize = 128;

/// Computes how many bytes of `s` are UTF-8 non-continuation bytes
/// with SIMD.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_feature = "neon")
))]
fn char_count_simd(s: &str) -> usize {
    use core::simd::cmp::SimdPartialOrd;
    use core::simd::num::SimdUint;
    use core::simd::{Select, i8x16, u8x16};
    const TAIL_CHUNK_SIZE: usize = 16;
    const CHUNK_SIZE: usize = 64;
    const BLOCK_SIZE: usize = 255;

    // The number of chunks of every block can't be over `u8::MAX`
    // to avoid overflow in loops of every block.
    const _: () = assert!(BLOCK_SIZE < 256);

    /// UTF-8 non-continuation bytes are >= -64 when interpreted as i8.
    const THRESHOLD: i8 = -64;

    let s = s.as_bytes();
    let (body, tail) = s.as_chunks::<CHUNK_SIZE>();
    let mut total = 0;

    // Group chunks into blocks to reduce the number of horizontal reductions.
    for block in body.chunks(BLOCK_SIZE) {
        let mut a = u8x16::splat(0);
        let mut b = u8x16::splat(0);
        let mut c = u8x16::splat(0);
        let mut d = u8x16::splat(0);
        let threshold = i8x16::splat(THRESHOLD);
        let one = u8x16::splat(1);
        let zero = u8x16::splat(0);

        // Computes how many bytes are UTF-8 non-continuation bytes.
        for chunk in block {
            a += u8x16::from_slice(&chunk[..16]).cast::<i8>().simd_ge(threshold).select(one, zero);
            b +=
                u8x16::from_slice(&chunk[16..32]).cast::<i8>().simd_ge(threshold).select(one, zero);
            c +=
                u8x16::from_slice(&chunk[32..48]).cast::<i8>().simd_ge(threshold).select(one, zero);
            d += u8x16::from_slice(&chunk[48..]).cast::<i8>().simd_ge(threshold).select(one, zero);
        }

        // Computes the total number of non-continuation bytes in the block.
        // Converts to `u16` to avoid overflow.
        total += a.cast::<u16>().reduce_sum() as usize
            + b.cast::<u16>().reduce_sum() as usize
            + c.cast::<u16>().reduce_sum() as usize
            + d.cast::<u16>().reduce_sum() as usize;
    }

    // Handle the tail with small chunks.
    let (chunks, rest) = tail.as_chunks::<TAIL_CHUNK_SIZE>();
    let mut acc = u8x16::splat(0);
    for chunk in chunks {
        let mask = u8x16::from_array(*chunk).cast::<i8>().simd_ge(i8x16::splat(THRESHOLD));
        acc += mask.select(u8x16::splat(1), u8x16::splat(0));
    }

    // Handle the remaining bytes when fewer than 16 are left.
    total += acc.reduce_sum() as usize + char_count_general_case(rest);

    total
}
