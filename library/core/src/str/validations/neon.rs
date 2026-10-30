//! UTF-8 lookup algorithm from Keiser and Lemire,
//! "Validating UTF-8 In Less Than One Instruction Per Byte" (2020).
//!
//! The tables check adjacent bytes. Longer sequences need two more bytes of history.
//! When a vector fails, use the scalar validator to get the exact error. Start up to three
//! bytes earlier if a character crosses the vector boundary.

use super::{Utf8Error, validate_utf8_scalar_from};
use crate::simd::num::SimdUint;
use crate::simd::{simd_swizzle, u8x16};

const VECTOR_LEN: usize = u8x16::LEN;
const BATCH_LEN: usize = 4 * VECTOR_LEN;

/// Enough input for a batch of four vectors.
pub(super) const MIN_LEN: usize = BATCH_LEN;

const TOO_SHORT: u8 = 1 << 0; // a lead byte followed by a non-continuation byte: C2 41
const TOO_LONG: u8 = 1 << 1; // an ASCII byte followed by a continuation byte: 41 80
const OVERLONG_3: u8 = 1 << 2; // E0 followed by 80..9F
const TOO_LARGE: u8 = 1 << 3; // F4..FF followed by 90..BF
const SURROGATE: u8 = 1 << 4; // ED followed by A0..BF
const OVERLONG_2: u8 = 1 << 5; // C0 or C1
// TOO_LARGE_1000 follows F5..FF, while OVERLONG_4 follows F0.
// They cannot match the same byte pair, so they share a bit.
const TOO_LARGE_1000: u8 = 1 << 6; // F5..FF followed by 80..8F
const OVERLONG_4: u8 = 1 << 6; // F0 followed by 80..8F
const TWO_CONTS: u8 = 1 << 7; // two continuation bytes in a row
const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTS;

/// Indexed by the high nibble of the previous byte.
const PREV_HIGH_NIBBLE: [u8; VECTOR_LEN] = [
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TOO_LONG,
    TWO_CONTS,
    TWO_CONTS,
    TWO_CONTS,
    TWO_CONTS,
    TOO_SHORT | OVERLONG_2,
    TOO_SHORT,
    TOO_SHORT | OVERLONG_3 | SURROGATE,
    TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4,
];

/// Indexed by the low nibble of the previous byte.
const PREV_LOW_NIBBLE: [u8; VECTOR_LEN] = [
    CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
    CARRY | OVERLONG_2,
    CARRY,
    CARRY,
    CARRY | TOO_LARGE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
];

/// Indexed by the high nibble of the current byte.
const CURRENT_HIGH_NIBBLE: [u8; VECTOR_LEN] = [
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
    TOO_SHORT,
];

/// `simd_swizzle!(prev, input, lanes_back(n))` joins the last `n` bytes of `prev`
/// with the first `VECTOR_LEN - n` bytes of `input`.
const fn lanes_back(n: usize) -> [usize; VECTOR_LEN] {
    let mut lanes = [0; VECTOR_LEN];
    let mut i = 0;
    while i < VECTOR_LEN {
        lanes[i] = VECTOR_LEN - n + i;
        i += 1;
    }
    lanes
}

fn load(v: &[u8], at: usize) -> u8x16 {
    u8x16::from_slice(&v[at..at + VECTOR_LEN])
}

/// Checks `input`, using `prev` for bytes across the vector boundary.
/// A nonzero lane flags an error found at that byte.
fn error_flags(input: u8x16, prev: u8x16) -> u8x16 {
    // For each byte of `input`, the bytes 1, 2 and 3 positions before it (`ext` on AArch64).
    let prev1 = simd_swizzle!(prev, input, lanes_back(1));
    let prev2 = simd_swizzle!(prev, input, lanes_back(2));
    let prev3 = simd_swizzle!(prev, input, lanes_back(3));

    // Indices stay below 16, so AArch64 can use one tbl instruction per lookup.
    let pair = u8x16::from_array(PREV_HIGH_NIBBLE).swizzle_dyn(prev1 >> 4)
        & u8x16::from_array(PREV_LOW_NIBBLE).swizzle_dyn(prev1 & u8x16::splat(0xF))
        & u8x16::from_array(CURRENT_HIGH_NIBBLE).swizzle_dyn(input >> 4);

    // TWO_CONTS also matches valid continuation pairs inside 3- and 4-byte characters.
    // A byte >= E0 two positions back, or >= F0 three positions back, requires such a pair.
    // The saturating subtractions detect these cases in bit 7.
    let must_be_cont = (prev2.saturating_sub(u8x16::splat(0xE0 - 0x80))
        | prev3.saturating_sub(u8x16::splat(0xF0 - 0x80)))
        & u8x16::splat(0x80);

    // A mismatch in bit 7 means a missing or extra continuation byte.
    // XOR leaves the other error bits unchanged.
    must_be_cont ^ pair
}

/// Finds the start of a possible unfinished character, or returns `bytes.len()`.
/// Invalid lead bytes are included so scalar validation can report the error.
fn incomplete_start(bytes: &[u8]) -> usize {
    let n = bytes.len();
    if n >= 1 && bytes[n - 1] >= 0xC0 {
        // A 2-, 3- or 4-byte lead has no following bytes yet.
        n - 1
    } else if n >= 2 && bytes[n - 2] >= 0xE0 {
        // A 3- or 4-byte lead has only one following byte.
        n - 2
    } else if n >= 3 && bytes[n - 3] >= 0xF0 {
        // A 4-byte lead has only two following bytes.
        n - 3
    } else {
        // No unfinished character found.
        n
    }
}

// A character may start in the previous vector. Back up to its lead byte before
// checking the error with the scalar validator.
#[cold]
fn validate_utf8_from_suspect(v: &[u8], pos: usize) -> Result<(), Utf8Error> {
    validate_utf8_scalar_from(v, incomplete_start(&v[..pos]))
}

/// Checks `v[start..]` with SIMD.
/// `v[..start]` must already be valid UTF-8, and at least `MIN_LEN` bytes must remain.
pub(super) fn validate(v: &[u8], start: usize) -> Result<(), Utf8Error> {
    let n = v.len();
    debug_assert!(start <= n && n - start >= MIN_LEN);

    let mut pos = start;
    // Keep the previous vector for characters that cross a vector boundary.
    let mut prev = u8x16::splat(0);
    let mut previous_non_ascii = false;
    // Check the first batch and any final full vectors one at a time.
    while n - pos >= VECTOR_LEN {
        let input = load(v, pos);
        let non_ascii = input.reduce_max() >= 0x80;
        let invalid = if non_ascii {
            error_flags(input, prev).reduce_max() != 0
        } else {
            // ASCII cannot finish a multibyte character left in the previous vector.
            previous_non_ascii && incomplete_start(&v[..pos]) != pos
        };
        if invalid {
            return validate_utf8_from_suspect(v, pos);
        }
        prev = input;
        previous_non_ascii = non_ascii;
        pos += VECTOR_LEN;

        // Check the first batch one vector at a time to report early errors sooner.
        if pos - start < BATCH_LEN {
            continue;
        }

        // Batch the middle of the input; the outer loop handles any leftover vectors.
        while n - pos >= BATCH_LEN {
            let block = &v[pos..][..BATCH_LEN];
            let (a, b, c, d) = (
                load(block, 0),
                load(block, VECTOR_LEN),
                load(block, 2 * VECTOR_LEN),
                load(block, 3 * VECTOR_LEN),
            );

            // All are ASCII or not.
            if ((a | b) | (c | d)).reduce_max() < 0x80 {
                // Skip this block only if no character was left unfinished before it.
                if previous_non_ascii && incomplete_start(&v[..pos]) != pos {
                    return validate_utf8_from_suspect(v, pos);
                }
                previous_non_ascii = false;
            } else {
                let errors =
                    [error_flags(a, prev), error_flags(b, a), error_flags(c, b), error_flags(d, c)];
                if ((errors[0] | errors[1]) | (errors[2] | errors[3])).reduce_max() != 0 {
                    // Find the first failing vector before scalar code reports the exact error.
                    let bad = errors.iter().position(|e| e.reduce_max() != 0).unwrap_or(0);
                    let at = pos + VECTOR_LEN * bad;
                    return validate_utf8_from_suspect(v, at);
                }
                previous_non_ascii = true;
            }
            prev = d;
            pos += BATCH_LEN;
        }
    }
    // Finish the tail, including any character split at `pos`.
    // This also catches an unfinished character when `pos == n`.
    let scalar_start = if previous_non_ascii { incomplete_start(&v[..pos]) } else { pos };
    validate_utf8_scalar_from(v, scalar_start)
}
