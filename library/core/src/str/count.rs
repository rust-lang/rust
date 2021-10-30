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

pub(super) fn count_chars(s: &str) -> usize {
    // For correctness, `CHUNK_SIZE` must be:
    // - Less than or equal to 255, otherwise we'll overflow bytes in `counts`.
    // - A multiple of `UNROLL_INNER`, otherwise our `break` inside the
    //   `body.chunks(CHUNK_SIZE)` loop.
    //
    // For performance, `CHUNK_SIZE` should be:
    // - Relatively cheap to `%` against.
    // - Large enough to avoid paying for the cost of the `sum_bytes_in_usize`
    //   too often.
    const CHUNK_SIZE: usize = 192;
    const UNROLL_INNER: usize = 4;

    // Check the properties of `CHUNK_SIZE` / `UNROLL_INNER` that are required
    // for correctness.
    const _: [(); 1] = [(); (CHUNK_SIZE < 256 && (CHUNK_SIZE % UNROLL_INNER) == 0) as usize];
    // SAFETY: transmuting `[u8]` to `[usize]` is safe except for size
    // differences which are handled by `align_to`.
    let (head, body, tail) = unsafe { s.as_bytes().align_to::<usize>() };

    let mut total = char_count_general_case(head) + char_count_general_case(tail);
    // Split `body` into `CHUNK_SIZE` chunks to reduce the frequency with which
    // we call `sum_bytes_in_usize`.
    for chunk in body.chunks(CHUNK_SIZE) {
        // We accumulate intermediate sums in `counts`, where each byte contains
        // a subset of the sum of this chunk, like a `[u8; size_of::<usize>()]`.
        let mut counts = 0;
        let unrolled_chunks = chunk.array_chunks::<UNROLL_INNER>();
        // If there's a remainder (know can only happen for the last item in
        // `chunks`, because `CHUNK_SIZE % UNROLL == 0`), then we need to
        // account for that (although we don't use it to later).
        let remainder = unrolled_chunks.remainder();
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
    let lsb = 0x0101_0101_0101_0101u64 as usize;
    ((!w >> 7) | (w >> 6)) & lsb
}

// Morally equivalent to `values.to_ne_bytes().into_iter().sum::<usize>()`, but
// more efficient.
#[inline]
fn sum_bytes_in_usize(values: usize) -> usize {
    const LSB_SHORTS: usize = 0x0001_0001_0001_0001_u64 as usize;
    const SKIP_BYTES: usize = 0x00ff_00ff_00ff_00ff_u64 as usize;

    let pair_sum: usize = (values & SKIP_BYTES) + ((values >> 8) & SKIP_BYTES);
    pair_sum.wrapping_mul(LSB_SHORTS) >> ((core::mem::size_of::<usize>() - 2) * 8)
}

// This is the most direct implementation of the concept of "count the number of
// bytes in the string which are not continuation bytes", and is used for the
// head and tail of the input string (the first and last item in the tuple
// returned by `slice::align_to`).
fn char_count_general_case(s: &[u8]) -> usize {
    const CONT_MASK_U8: u8 = 0b0011_1111;
    const TAG_CONT_U8: u8 = 0b1000_0000;
    let mut leads = 0;
    for &byte in s {
        let is_lead = (byte & !CONT_MASK_U8) != TAG_CONT_U8;
        leads += is_lead as usize;
    }
    leads
}
