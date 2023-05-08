//! Operations related to UTF-8 validation.

use crate::mem;

use super::Utf8Error;

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

const WORD_BYTES: usize = mem::size_of::<usize>();
/// A word-sized bitmask where every byte's MSB is set,
/// indicating a non-ASCII character.
const NONASCII_MASK: usize = usize::repeat_u8(0x80);

/// Walks through `buf` checking that it's a valid UTF-8 sequence,
/// returning `Ok(())` in that case, or, if it is invalid, `Err(err)`.
#[inline(always)]
#[rustc_const_unstable(feature = "str_internals", issue = "none")]
pub(super) const fn run_utf8_validation(buf: &[u8]) -> Result<(), Utf8Error> {
    // we check aligned blocks of up to 8 words at a time
    const ASCII_BLOCK_8X: usize = 8 * WORD_BYTES;
    const ASCII_BLOCK_4X: usize = 4 * WORD_BYTES;
    const ASCII_BLOCK_2X: usize = 2 * WORD_BYTES;

    // establish buffer extent
    let (mut curr, end) = (0, buf.len());
    let start = buf.as_ptr();
    // calculate the byte offset until the first word aligned block
    let align_offset = start.align_offset(WORD_BYTES);

    // calculate the maximum byte at which a block of size N could begin,
    // without taking alignment into account
    let block_end_8x = block_end(end, ASCII_BLOCK_8X);
    let block_end_4x = block_end(end, ASCII_BLOCK_4X);
    let block_end_2x = block_end(end, ASCII_BLOCK_2X);

    while curr < end {
        if buf[curr] < 128 {
            // `align_offset` can basically only be `usize::MAX` for ZST
            // pointers, so the first check is almost certainly optimized away
            if align_offset == usize::MAX {
                curr += 1;
                continue;
            }

            // check if `curr`'s pointer is word-aligned
            let offset = align_offset.wrapping_sub(curr) % WORD_BYTES;
            if offset == 0 {
                let len = 'block: loop {
                    macro_rules! block_loop {
                        ($N:expr) => {
                            // SAFETY: we have checked before that there are
                            // still at least `N * size_of::<usize>()` in the
                            // buffer and that the current byte is word-aligned
                            let block = unsafe { &*(start.add(curr) as *const [usize; $N]) };
                            if has_non_ascii_byte(block) {
                                break 'block Some($N);
                            }

                            curr += $N * WORD_BYTES;
                        };
                    }

                    // check 8-word blocks for non-ASCII bytes
                    while curr < block_end_8x {
                        block_loop!(8);
                    }

                    // check 4-word blocks for non-ASCII bytes
                    while curr < block_end_4x {
                        block_loop!(4);
                    }

                    // check 2-word blocks for non-ASCII bytes
                    while curr < block_end_2x {
                        block_loop!(2);
                    }

                    // `(size_of::<usize>() * 2) + (align_of::<usize> - 1)`
                    // bytes remain at most
                    break None;
                };

                // if the block loops were stopped due to a non-ascii byte
                // in some block, do another block-wise search using the last
                // used block-size for the specific byte in the previous block
                // in order to skip checking all bytes up to that one
                // individually.
                // NOTE: this operation does not auto-vectorize well, so it is
                // done only in case a non-ASCII byte is actually found
                if let Some(len) = len {
                    // SAFETY: `curr` has not changed since the last block loop,
                    // so it still points at a byte marking the beginning of a
                    // word-sized block of the given `len`
                    let block = unsafe {
                        let ptr = start.add(curr) as *const usize;
                        core::slice::from_raw_parts(ptr, len)
                    };

                    // calculate the amount of bytes that can be skipped without
                    // having to check them individually
                    let (skip, non_ascii) = non_ascii_byte_position(block);
                    curr += skip;

                    // if a non-ASCII byte was found, skip the subsequent
                    // byte-wise loop and go straight back to the main loop
                    if non_ascii {
                        continue;
                    }
                }

                // ...otherwise, fall back to byte-wise checks
                while curr < end && buf[curr] < 128 {
                    curr += 1;
                }
            } else {
                // byte is < 128 (ASCII), but pointer is not word-aligned, skip
                // until the loop reaches the next word-aligned block)
                for _ in 0..offset {
                    // no need to check alignment again for every byte, so skip
                    // up to `offset` valid ASCII bytes if possible
                    curr += 1;
                    if !(curr < end && buf[curr] < 128) {
                        break;
                    }
                }
            }
        } else {
            // non-ASCII case: validate up to 4 bytes, then advance `curr`
            // accordingly
            match validate_non_acii_bytes(buf, curr) {
                Ok(next) => curr = next,
                Err(e) => return Err(e),
            }
        }
    }

    Ok(())
}

#[inline]
const fn validate_non_acii_bytes(buf: &[u8], mut curr: usize) -> Result<usize, Utf8Error> {
    const fn subarray<const N: usize>(buf: &[u8], idx: usize) -> Option<[u8; N]> {
        if buf.len() - idx < N {
            return None;
        }

        // SAFETY: checked in previous condition
        Some(unsafe { *(buf.as_ptr().add(idx) as *const [u8; N]) })
    }

    let prev = curr;
    macro_rules! err {
        ($error_len: expr) => {
            return Err(Utf8Error { valid_up_to: prev, error_len: $error_len })
        };
    }

    let b0 = buf[curr];
    match utf8_char_width(b0) {
        2 => {
            let Some([_, b1]) = subarray(buf, curr) else {
                err!(None);
            };

            if b1 as i8 >= -64 {
                err!(Some(1));
            }

            curr += 2;
        }
        3 => {
            let Some([_, b1, b2]) = subarray(buf, curr) else {
                err!(None);
            };

            match (b0, b1) {
                (0xE0, 0xA0..=0xBF)
                | (0xE1..=0xEC, 0x80..=0xBF)
                | (0xED, 0x80..=0x9F)
                | (0xEE..=0xEF, 0x80..=0xBF) => {}
                _ => err!(Some(1)),
            }

            if b2 as i8 >= -64 {
                err!(Some(2));
            }

            curr += 3;
        }
        4 => {
            let Some([_, b1, b2, b3]) = subarray(buf, curr) else {
                err!(None);
            };

            match (b0, b1) {
                (0xF0, 0x90..=0xBF) | (0xF1..=0xF3, 0x80..=0xBF) | (0xF4, 0x80..=0x8F) => {}
                _ => err!(Some(1)),
            }

            if b2 as i8 >= -64 {
                err!(Some(2));
            }

            if b3 as i8 >= -64 {
                err!(Some(3));
            }

            curr += 4;
        }
        _ => err!(Some(1)),
    }

    Ok(curr)
}

/// Returns `true` if any one block is not a valid ASCII byte.
#[inline(always)]
const fn has_non_ascii_byte<const N: usize>(block: &[usize; N]) -> bool {
    let mut vector = [0; N];

    let mut i = 0;
    while i < N {
        vector[i] = block[i] & NONASCII_MASK;
        i += 1;
    }

    i = 0;
    while i < N {
        if vector[i] > 0 {
            return true;
        }
        i += 1;
    }

    false
}

/// Returns the number of consecutive ASCII bytes within `block` until the first
/// non-ASCII byte and `true`, if a non-ASCII byte was found.
///
/// Returns `block.len() * size_of::<usize>()` and `false`, if all bytes are
/// ASCII bytes.
#[inline(always)]
const fn non_ascii_byte_position(block: &[usize]) -> (usize, bool) {
    let mut i = 0;
    while i < block.len() {
        let mask = block[i] & NONASCII_MASK;
        let ctz = mask.trailing_zeros() as usize;
        let byte = ctz / WORD_BYTES;

        if byte != WORD_BYTES {
            return (byte + (i * WORD_BYTES), true);
        }

        i += 1;
    }

    (WORD_BYTES * block.len(), false)
}

#[inline(always)]
const fn block_end(end: usize, block_size: usize) -> usize {
    if end >= block_size { end - block_size + 1 } else { 0 }
}

/// Given a first byte, determines how many bytes are in this UTF-8 character.
#[unstable(feature = "str_internals", issue = "none")]
#[must_use]
#[inline]
pub const fn utf8_char_width(byte: u8) -> usize {
    // https://tools.ietf.org/html/rfc3629
    const UTF8_CHAR_WIDTH: [u8; 256] = [
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

    UTF8_CHAR_WIDTH[byte as usize] as usize
}

/// Mask of the value bits of a continuation byte.
const CONT_MASK: u8 = 0b0011_1111;
