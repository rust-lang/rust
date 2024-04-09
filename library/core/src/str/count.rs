//! Code for efficiently counting the number of `char`s or lines in a UTF-8
//! encoded string
//!
//! ## `char` count details
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

const USIZE_SIZE: usize = core::mem::size_of::<usize>();
const UNROLL_INNER: usize = 4;
const LSB: usize = usize::repeat_u8(0x01);

#[inline]
pub(super) fn count_chars(s: &str) -> usize {
    count::<CharCount>(s)
}

#[inline]
pub(super) fn count_lines(s: &str) -> usize {
    let newline_count = count::<NewlineCount>(s);
    // The logic for going from newline count to line count is a bit weird,
    // consider that `"foo\nbar"` is 2 lines, `"foo\nbar\n"` is also 2 lines,
    // `"\n"` is one line, and `""` is zero lines.
    let ends_with_newline = s.as_bytes().last() == Some(&b'\n');
    let is_single_newline = ends_with_newline && s.len() == 1;
    let is_special = is_single_newline || s.is_empty();
    let adjust_len_by_one = !ends_with_newline && !is_special;
    newline_count + adjust_len_by_one as usize
}

trait CountPred {
    /// Bytes in `u` which match the pred must be `0x01` in the result, bytes
    /// which fail the pred must be `0x00`.
    fn test_each_byte_in_word(u: usize) -> usize;
    /// Slow path for small inputs.
    fn count_general_case(s: &[u8]) -> usize;
}

struct CharCount;
impl CountPred for CharCount {
    #[inline]
    fn count_general_case(s: &[u8]) -> usize {
        char_count_general_case(s)
    }
    #[inline]
    fn test_each_byte_in_word(u: usize) -> usize {
        contains_non_continuation_byte(u)
    }
}
struct NewlineCount;
impl CountPred for NewlineCount {
    #[inline]
    fn count_general_case(s: &[u8]) -> usize {
        s.iter().filter(|b| **b == b'\n').count()
    }
    #[inline]
    fn test_each_byte_in_word(u: usize) -> usize {
        const NEWLINES: usize = usize::repeat_u8(b'\n');
        const NOT_MSB: usize = usize::repeat_u8(0x7f);
        // bytes of `diff` are nonzero when bytes of `u` don't contain newline
        let diff = u ^ NEWLINES;
        let res = !(((diff & NOT_MSB).wrapping_add(NOT_MSB) | diff) >> 7);
        res & LSB
    }
}

#[inline]
fn count<P: CountPred>(s: &str) -> usize {
    if s.len() < USIZE_SIZE * UNROLL_INNER {
        // Avoid entering the optimized implementation for strings where the
        // difference is not likely to matter, or where it might even be slower.
        // That said, a ton of thought was not spent on the particular threshold
        // here, beyond "this value seems to make sense".
        P::count_general_case(s.as_bytes())
    } else {
        do_count::<P>(s)
    }
}

fn do_count<P: CountPred>(s: &str) -> usize {
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
    const _: () = assert!(CHUNK_SIZE % UNROLL_INNER == 0);

    // SAFETY: transmuting `[u8]` to `[usize]` is safe except for size
    // differences which are handled by `align_to`.
    let (head, body, tail) = unsafe { s.as_bytes().align_to::<usize>() };

    // This should be quite rare, and basically exists to handle the degenerate
    // cases where align_to fails (as well as miri under symbolic alignment
    // mode).
    //
    // The `unlikely` helps discourage LLVM from inlining the body, which is
    // nice, as we would rather not mark the `P::count_general_case` function
    // as cold.
    if unlikely(body.is_empty() || head.len() > USIZE_SIZE || tail.len() > USIZE_SIZE) {
        return P::count_general_case(s.as_bytes());
    }

    let mut total = P::count_general_case(head) + P::count_general_case(tail);
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
                counts += P::test_each_byte_in_word(word);
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
                counts += P::test_each_byte_in_word(word);
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
