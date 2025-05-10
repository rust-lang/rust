//! Operations related to UTF-8 validation.

use super::Utf8Error;
use super::error::Utf8ErrorLen;
use crate::intrinsics::const_eval_select;

/// Returns the initial codepoint accumulator for the first byte.
/// The first byte is special, only want bottom 5 bits for width 2, 4 bits
/// for width 3, and 3 bits for width 4.
#[inline]
const fn utf8_first_byte(byte: u8, width: u32) -> u32 {
    (byte & (0x7F >> width)) as u32
}

/// Returns the value of `ch` updated with continuation byte `byte`.
#[inline]
const fn utf8_acc_cont_byte(ch: u32, byte: u8) -> u32 {
    (ch << 6) | (byte & CONT_MASK) as u32
}

/// Checks whether the byte is a UTF-8 continuation byte (i.e., starts with the
/// bits `10`).
#[inline]
pub(super) const fn utf8_is_cont_byte(byte: u8) -> bool {
    (byte as i8) < -64
}

/// Reads the next code point out of a byte iterator (assuming a
/// UTF-8-like encoding).
///
/// # Safety
///
/// `bytes` must produce a valid UTF-8-like (UTF-8 or WTF-8) string
#[unstable(feature = "str_internals", issue = "none")]
#[inline]
pub unsafe fn next_code_point<'a, I: Iterator<Item = &'a u8>>(bytes: &mut I) -> Option<u32> {
    // Decode UTF-8
    let x = *bytes.next()?;
    if x < 128 {
        return Some(x as u32);
    }

    // Multibyte case follows
    // Decode from a byte combination out of: [[[x y] z] w]
    // NOTE: Performance is sensitive to the exact formulation here
    let init = utf8_first_byte(x, 2);
    // SAFETY: `bytes` produces an UTF-8-like string,
    // so the iterator must produce a value here.
    let y = unsafe { *bytes.next().unwrap_unchecked() };
    let mut ch = utf8_acc_cont_byte(init, y);
    if x >= 0xE0 {
        // [[x y z] w] case
        // 5th bit in 0xE0 .. 0xEF is always clear, so `init` is still valid
        // SAFETY: `bytes` produces an UTF-8-like string,
        // so the iterator must produce a value here.
        let z = unsafe { *bytes.next().unwrap_unchecked() };
        let y_z = utf8_acc_cont_byte((y & CONT_MASK) as u32, z);
        ch = init << 12 | y_z;
        if x >= 0xF0 {
            // [x y z w] case
            // use only the lower 3 bits of `init`
            // SAFETY: `bytes` produces an UTF-8-like string,
            // so the iterator must produce a value here.
            let w = unsafe { *bytes.next().unwrap_unchecked() };
            ch = (init & 7) << 18 | utf8_acc_cont_byte(y_z, w);
        }
    }

    Some(ch)
}

/// Reads the last code point out of a byte iterator (assuming a
/// UTF-8-like encoding).
///
/// # Safety
///
/// `bytes` must produce a valid UTF-8-like (UTF-8 or WTF-8) string
#[inline]
pub(super) unsafe fn next_code_point_reverse<'a, I>(bytes: &mut I) -> Option<u32>
where
    I: DoubleEndedIterator<Item = &'a u8>,
{
    // Decode UTF-8
    let w = match *bytes.next_back()? {
        next_byte if next_byte < 128 => return Some(next_byte as u32),
        back_byte => back_byte,
    };

    // Multibyte case follows
    // Decode from a byte combination out of: [x [y [z w]]]
    let mut ch;
    // SAFETY: `bytes` produces an UTF-8-like string,
    // so the iterator must produce a value here.
    let z = unsafe { *bytes.next_back().unwrap_unchecked() };
    ch = utf8_first_byte(z, 2);
    if utf8_is_cont_byte(z) {
        // SAFETY: `bytes` produces an UTF-8-like string,
        // so the iterator must produce a value here.
        let y = unsafe { *bytes.next_back().unwrap_unchecked() };
        ch = utf8_first_byte(y, 3);
        if utf8_is_cont_byte(y) {
            // SAFETY: `bytes` produces an UTF-8-like string,
            // so the iterator must produce a value here.
            let x = unsafe { *bytes.next_back().unwrap_unchecked() };
            ch = utf8_first_byte(x, 4);
            ch = utf8_acc_cont_byte(ch, y);
        }
        ch = utf8_acc_cont_byte(ch, z);
    }
    ch = utf8_acc_cont_byte(ch, w);

    Some(ch)
}

// The shift-based DFA algorithm for UTF-8 validation.
// Ref: <https://gist.github.com/pervognsen/218ea17743e1442e59bb60d29b1aa725>
//
// In short, we encode DFA transitions in an array `TRANS_TABLE` such that:
// ```
// TRANS_TABLE[next_byte] =
//     OFFSET[target_state1] << OFFSET[source_state1] |
//     OFFSET[target_state2] << OFFSET[source_state2] |
//     ...
// ```
// Where `OFFSET[]` is a compile-time map from each state to a distinct 0..32 value.
//
// To execute the DFA:
// ```
// let state = OFFSET[initial_state];
// for byte in .. {
//     state = TRANS_TABLE[byte] >> (state & ((1 << BITS_PER_STATE) - 1));
// }
// ```
// By choosing `BITS_PER_STATE = 5` and `state: u32`, we can replace the masking by `wrapping_shr`
// and it becomes free on modern ISAs, including x86, x86_64 and ARM.
//
// ```
// // On x86-64-v3: (more instructions on ordinary x86_64 but with same cycles-per-byte)
// //   shrx state, qword ptr [TRANS_TABLE + 4 * byte], state
// // On aarch64/ARMv8:
// //   ldr temp, [TRANS_TABLE, byte, lsl 2]
// //   lsr state, temp, state
// state = TRANS_TABLE[byte].wrapping_shr(state);
// ```
//
// The DFA is directly derived from UTF-8 syntax from the RFC3629:
// <https://datatracker.ietf.org/doc/html/rfc3629#section-4>.
// We assign S0 as ERROR and S1 as ACCEPT. DFA starts at S1.
// Syntax are annotated with DFA states in angle bracket as following:
//
// UTF8-char   = <S1> (UTF8-1 / UTF8-2 / UTF8-3 / UTF8-4)
// UTF8-1      = <S1> %x00-7F
// UTF8-2      = <S1> %xC2-DF                <S2> UTF8-tail
// UTF8-3      = <S1> %xE0                   <S3> %xA0-BF <S2> UTF8-tail /
//               <S1> (%xE1-EC / %xEE-EF)    <S4> 2( UTF8-tail ) /
//               <S1> %xED                   <S5> %x80-9F <S2> UTF8-tail
// UTF8-4      = <S1> %xF0    <S6> %x90-BF   <S4> 2( UTF8-tail ) /
//               <S1> %xF1-F3 <S7> UTF8-tail <S4> 2( UTF8-tail ) /
//               <S1> %xF4    <S8> %x80-8F   <S4> 2( UTF8-tail )
// UTF8-tail   = %x80-BF   # Inlined into above usages.
//
// You may notice that encoding 9 states with 5bits per state into 32bit seems impossible,
// but we exploit overlapping bits to find a possible `OFFSET[]` and `TRANS_TABLE[]` solution.
// The SAT solver to find such (minimal) solution is in `./solve_dfa.py`.
// The solution is also appended to the end of that file and is verifiable.
const BITS_PER_STATE: u32 = 5;
const STATE_MASK: u32 = (1 << BITS_PER_STATE) - 1;
const STATE_CNT: usize = 9;
const ST_ERROR: u32 = OFFSETS[0];
const ST_ACCEPT: u32 = OFFSETS[1];
// See the end of `./solve_dfa.py`.
const OFFSETS: [u32; STATE_CNT] = [0, 6, 16, 19, 1, 25, 11, 18, 24];

// Keep the whole table in a single page.
#[repr(align(1024))]
struct TransitionTable([u32; 256]);

static TRANS_TABLE: TransitionTable = {
    let mut table = [0u32; 256];
    let mut b = 0;
    while b < 256 {
        // See the end of `./solve_dfa.py`.
        table[b] = match b as u8 {
            0x00..=0x7F => 0x180,
            0xC2..=0xDF => 0x400,
            0xE0 => 0x4C0,
            0xE1..=0xEC | 0xEE..=0xEF => 0x40,
            0xED => 0x640,
            0xF0 => 0x2C0,
            0xF1..=0xF3 => 0x480,
            0xF4 => 0x600,
            0x80..=0x8F => 0x21060020,
            0x90..=0x9F => 0x20060820,
            0xA0..=0xBF => 0x860820,
            0xC0..=0xC1 | 0xF5..=0xFF => 0x0,
        };
        b += 1;
    }
    TransitionTable(table)
};

#[inline(always)]
const fn next_state(st: u32, byte: u8) -> u32 {
    TRANS_TABLE.0[byte as usize].wrapping_shr(st)
}

/// Check if `byte` is a valid UTF-8 first byte, assuming it must be a valid first or
/// continuation byte.
#[inline(always)]
const fn is_utf8_first_byte(byte: u8) -> bool {
    byte as i8 >= 0b1100_0000u8 as i8
}

/// # Safety
/// The caller must ensure `bytes[..i]` is a valid UTF-8 prefix and `st` is the DFA state after
/// executing on `bytes[..i]`.
#[inline]
const unsafe fn resolve_error_location(st: u32, bytes: &[u8], i: usize) -> Utf8Error {
    // There are two cases:
    // 1. [valid UTF-8..] | *here
    //    The previous state must be ACCEPT for the case 1, and `valid_up_to = i`.
    // 2. [valid UTF-8..] | valid first byte, [valid continuation byte...], *here
    //    `valid_up_to` is at the latest non-continuation byte, which must exist and
    //    be in range `(i-3)..i`.
    let (valid_up_to, error_len) = if st & STATE_MASK == ST_ACCEPT {
        (i, Utf8ErrorLen::One)
    // SAFETY: UTF-8 first byte must exist if we are in an intermediate state.
    // We use pointer here because `get_unchecked` is not const fn.
    } else if is_utf8_first_byte(unsafe { bytes.as_ptr().add(i - 1).read() }) {
        (i - 1, Utf8ErrorLen::One)
    // SAFETY: Same as above.
    } else if is_utf8_first_byte(unsafe { bytes.as_ptr().add(i - 2).read() }) {
        (i - 2, Utf8ErrorLen::Two)
    } else {
        (i - 3, Utf8ErrorLen::Three)
    };
    Utf8Error { valid_up_to, error_len }
}

// The simpler but slower algorithm to run DFA with error handling.
// Returns the final state after execution on the whole slice.
//
// # Safety
// The caller must ensure `bytes[..i]` is a valid UTF-8 prefix and `st` is the DFA state after
// executing on `bytes[..i]`.
#[inline]
const unsafe fn run_with_error_handling(
    mut st: u32,
    bytes: &[u8],
    mut i: usize,
) -> Result<u32, Utf8Error> {
    while i < bytes.len() {
        let new_st = next_state(st, bytes[i]);
        if new_st & STATE_MASK == ST_ERROR {
            // SAFETY: Guaranteed by the caller.
            return Err(unsafe { resolve_error_location(st, bytes, i) });
        }
        st = new_st;
        i += 1;
    }
    Ok(st)
}

/// Walks through `v` checking that it's a valid UTF-8 sequence,
/// returning `Ok(())` in that case, or, if it is invalid, `Err(err)`.
#[cfg_attr(not(feature = "optimize_for_size"), inline)]
#[rustc_allow_const_fn_unstable(const_eval_select)] // fallback impl has same behavior
pub(super) const fn run_utf8_validation(bytes: &[u8]) -> Result<(), Utf8Error> {
    if cfg!(feature = "optimize_for_size") {
        run_utf8_validation_const(bytes)
    } else {
        const_eval_select((bytes,), run_utf8_validation_const, run_utf8_validation_rt)
    }
}

#[inline]
const fn run_utf8_validation_const(bytes: &[u8]) -> Result<(), Utf8Error> {
    // SAFETY: Start at empty string with valid state ACCEPT.
    match unsafe { run_with_error_handling(ST_ACCEPT, bytes, 0) } {
        Err(err) => Err(err),
        Ok(st) if st & STATE_MASK == ST_ACCEPT => Ok(()),
        Ok(st) => {
            // SAFETY: `st` is the last state after execution without encountering any error.
            let mut err = unsafe { resolve_error_location(st, bytes, bytes.len()) };
            err.error_len = Utf8ErrorLen::Eof;
            Err(err)
        }
    }
}

#[inline]
fn run_utf8_validation_rt(bytes: &[u8]) -> Result<(), Utf8Error> {
    const MAIN_CHUNK_SIZE: usize = 16;
    const ASCII_CHUNK_SIZE: usize = 16;
    const { assert!(ASCII_CHUNK_SIZE % MAIN_CHUNK_SIZE == 0) };

    let mut i = bytes.len() % MAIN_CHUNK_SIZE;
    // SAFETY: Start at initial state ACCEPT.
    let mut st = unsafe { run_with_error_handling(ST_ACCEPT, &bytes[..i], 0)? };

    while i < bytes.len() {
        // Fast path: if the current state is ACCEPT, we can skip to the next non-ASCII chunk.
        // We also did a quick inspection on the first byte to avoid getting into this path at all
        // when handling strings with almost no ASCII, eg. Chinese scripts.
        // SAFETY: `i` is in bound.
        if st & STATE_MASK == ST_ACCEPT && unsafe { bytes.get_unchecked(i).is_ascii() } {
            // SAFETY: `i` is in bound.
            let rest = unsafe { bytes.get_unchecked(i..) };
            let mut ascii_chunks = rest.array_chunks::<ASCII_CHUNK_SIZE>();
            let ascii_rest_chunk_cnt = ascii_chunks.len();
            let pos = ascii_chunks
                .position(|chunk| {
                    // NB. Always traverse the whole chunk instead of `.all()`, to persuade LLVM to
                    // vectorize this check.
                    // We also do not use `<[u8]>::is_ascii` which is unnecessarily complex here.
                    #[expect(clippy::unnecessary_fold)]
                    let all_ascii = chunk.iter().fold(true, |acc, b| acc && b.is_ascii());
                    !all_ascii
                })
                .unwrap_or(ascii_rest_chunk_cnt);
            i += pos * ASCII_CHUNK_SIZE;
            if i >= bytes.len() {
                break;
            }
        }

        // SAFETY: `i` and `i + MAIN_CHUNK_SIZE` are in bound by loop invariant.
        let chunk = unsafe { &*bytes.as_ptr().add(i).cast::<[u8; MAIN_CHUNK_SIZE]>() };
        let mut new_st = st;
        for &b in chunk {
            new_st = next_state(new_st, b);
        }
        if new_st & STATE_MASK == ST_ERROR {
            // SAFETY: `st` is the last state after executing `bytes[..i]` without encountering any error.
            // And we know the next chunk must fail the validation.
            return Err(unsafe { run_with_error_handling(st, bytes, i).unwrap_err_unchecked() });
        }

        st = new_st;
        i += MAIN_CHUNK_SIZE;
    }

    if st & STATE_MASK != ST_ACCEPT {
        // SAFETY: Same as above.
        let mut err = unsafe { resolve_error_location(st, bytes, bytes.len()) };
        err.error_len = Utf8ErrorLen::Eof;
        return Err(err);
    }

    Ok(())
}

// https://tools.ietf.org/html/rfc3629
const UTF8_CHAR_WIDTH: &[u8; 256] = &[
    // 1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 5
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 6
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 7
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 8
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E
    4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F
];

/// Given a first byte, determines how many bytes are in this UTF-8 character.
#[unstable(feature = "str_internals", issue = "none")]
#[must_use]
#[inline]
pub const fn utf8_char_width(b: u8) -> usize {
    UTF8_CHAR_WIDTH[b as usize] as usize
}

/// Mask of the value bits of a continuation byte.
const CONT_MASK: u8 = 0b0011_1111;
