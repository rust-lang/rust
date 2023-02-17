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

const NONASCII_MASK: usize = usize::repeat_u8(0x80);

/// Returns `true` if any byte in the word `x` is nonascii (>= 128).
#[inline]
const fn contains_nonascii(x: usize) -> bool {
    (x & NONASCII_MASK) != 0
}

/// Reads the first code point out of a byte slice validating whether it’s
/// valid.
///
/// This is different than [`next_code_point`] in that it doesn’t assume
/// argument is well-formed UTF-8-like string.  Together with the character its
/// encoded length is returned.
///
/// If front of the bytes slice doesn’t contain valid UTF-8 bytes sequence (that
/// includes a WTF-8 encoded surrogate) returns `None`.
///
/// ```
/// #![feature(str_internals)]
/// use core::str::try_next_code_point;
///
/// assert_eq!(Some(('f', 1)), try_next_code_point(b"foo".as_ref()));
/// assert_eq!(Some(('Ż', 2)), try_next_code_point("Żółw".as_bytes()));
/// assert_eq!(None, try_next_code_point(b"\xffoo".as_ref()));
/// ```
#[unstable(feature = "str_internals", issue = "none")]
#[inline]
pub const fn try_next_code_point(bytes: &[u8]) -> Option<(char, usize)> {
    let first = match bytes.first() {
        Some(&byte) => byte,
        None => return None,
    };
    let (value, length) = if first < 0x80 {
        (first as u32, 1)
    } else if let Ok((cp, len)) = try_finish_byte_sequence(first, bytes, 0) {
        (cp, len)
    } else {
        return None;
    };
    // SAFETY: We’ve just verified value is correct Unicode scalar value.
    // Either ASCII (first branch of the if-else-if-else) or non-ASCII Unicode
    // character (second branch).
    Some((unsafe { char::from_u32_unchecked(value) }, length))
}

/// Reads the last code point out of a byte slice validating whether it’s
/// valid.
///
/// This is different than `next_code_point_reverse` in that it doesn’t assume
/// argument is well-formed UTF-8-like string.  Together with the character its
/// encoded length is returned.
///
/// If back of the bytes slice doesn’t contain valid UTF-8 bytes sequence (that
/// includes a WTF-8 encoded surrogate) returns `None`.
///
/// ```
/// #![feature(str_internals)]
/// use core::str::try_next_code_point_reverse;
///
/// assert_eq!(Some(('o', 1)), try_next_code_point_reverse(b"foo".as_ref()));
/// assert_eq!(Some(('‽', 3)), try_next_code_point_reverse("Uh‽".as_bytes()));
/// assert_eq!(None, try_next_code_point_reverse(b"foo\xff".as_ref()));
/// ```
#[unstable(feature = "str_internals", issue = "none")]
#[inline]
pub const fn try_next_code_point_reverse(bytes: &[u8]) -> Option<(char, usize)> {
    let mut n = 1;
    let limit = bytes.len();
    let limit = if limit < 4 { limit } else { 4 }; // not .min(4) because of const
    while n <= limit && !bytes[bytes.len() - n].is_utf8_char_boundary() {
        n += 1;
    }
    if n <= limit {
        // It’s not clear to me why, but range indexing isn’t const here,
        // i.e. `&bytes[bytes.len() - n..]` doesn’t compile.  Because of that
        // I’m resorting to unsafe block with from_raw_parts.
        // SAFETY: n ≤ limit ≤ bytes.len() thus bytes.len() - n ≥ 0 and we
        // have n remaining bytes.
        let bytes = unsafe { crate::slice::from_raw_parts(bytes.as_ptr().add(bytes.len() - n), n) };
        if let Some((chr, len)) = try_next_code_point(bytes) {
            if n == len {
                return Some((chr, len));
            }
        }
    }
    None
}

/// Walks through `v` checking that it's a valid UTF-8 sequence,
/// returning `Ok(())` in that case, or, if it is invalid, `Err(err)`.
#[inline(always)]
#[rustc_const_unstable(feature = "str_internals", issue = "none")]
pub(super) const fn run_utf8_validation(v: &[u8]) -> Result<(), Utf8Error> {
    let mut index = 0;
    let len = v.len();

    let usize_bytes = mem::size_of::<usize>();
    let ascii_block_size = 2 * usize_bytes;
    let blocks_end = if len >= ascii_block_size { len - ascii_block_size + 1 } else { 0 };
    let align = v.as_ptr().align_offset(usize_bytes);

    while index < len {
        let valid_up_to = index;
        let first = v[index];
        if first >= 128 {
            match try_finish_byte_sequence(first, v, index) {
                Ok((_value, length)) => index += length,
                Err(error_len) => return Err(Utf8Error { valid_up_to, error_len }),
            }
        } else {
            // Ascii case, try to skip forward quickly.
            // When the pointer is aligned, read 2 words of data per iteration
            // until we find a word containing a non-ascii byte.
            if align != usize::MAX && align.wrapping_sub(index) % usize_bytes == 0 {
                let ptr = v.as_ptr();
                while index < blocks_end {
                    // SAFETY: since `align - index` and `ascii_block_size` are
                    // multiples of `usize_bytes`, `block = ptr.add(index)` is
                    // always aligned with a `usize` so it's safe to dereference
                    // both `block` and `block.add(1)`.
                    unsafe {
                        let block = ptr.add(index) as *const usize;
                        // break if there is a nonascii byte
                        let zu = contains_nonascii(*block);
                        let zv = contains_nonascii(*block.add(1));
                        if zu || zv {
                            break;
                        }
                    }
                    index += ascii_block_size;
                }
                // step from the point where the wordwise loop stopped
                while index < len && v[index] < 128 {
                    index += 1;
                }
            } else {
                index += 1;
            }
        }
    }

    Ok(())
}

/// Try to finish an UTF-8 byte sequence.
///
/// Assumes that `bytes[index] == first` and than `first >= 128`, i.e. that
/// `index` points at the beginning of a non-ASCII UTF-8 sequence in `bytes`.
///
/// If the byte sequence at the index is correct, returns decoded code point and
/// length of the sequence.  If it was invalid returns number of invalid bytes
/// or None if read was cut short.
#[inline(always)]
#[rustc_const_unstable(feature = "str_internals", issue = "none")]
const fn try_finish_byte_sequence(
    first: u8,
    bytes: &[u8],
    index: usize,
) -> Result<(u32, usize), Option<u8>> {
    macro_rules! get {
        (raw $offset:expr) => {
            if index + $offset < bytes.len() {
                bytes[index + $offset]
            } else {
                return Err(None)
            }
        };
        (cont $offset:expr) => {{
            let byte = get!(raw $offset);
            if !utf8_is_cont_byte(byte) {
                return Err(Some($offset as u8))
            }
            byte
        }}
    }

    // 2-byte encoding is for codepoints  \u{0080} to  \u{07ff}
    //        first  C2 80        last DF BF
    // 3-byte encoding is for codepoints  \u{0800} to  \u{ffff}
    //        first  E0 A0 80     last EF BF BF
    //   excluding surrogates codepoints  \u{d800} to  \u{dfff}
    //               ED A0 80 to       ED BF BF
    // 4-byte encoding is for codepoints \u{1000}0 to \u{10ff}ff
    //        first  F0 90 80 80  last F4 8F BF BF
    //
    // Use the UTF-8 syntax from the RFC
    //
    // https://tools.ietf.org/html/rfc3629
    // UTF8-1      = %x00-7F
    // UTF8-2      = %xC2-DF UTF8-tail
    // UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
    //               %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
    // UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
    //               %xF4 %x80-8F 2( UTF8-tail )
    match utf8_char_width(first) {
        2 => {
            let second = get!(cont 1);
            let value = utf8_first_byte(first, 3);
            let value = utf8_acc_cont_byte(value, second);
            Ok((value, 2))
        }
        3 => {
            let second = get!(raw 1);
            match (first, second) {
                (0xE0, 0xA0..=0xBF)
                | (0xE1..=0xEC, 0x80..=0xBF)
                | (0xED, 0x80..=0x9F)
                | (0xEE..=0xEF, 0x80..=0xBF) => {}
                _ => return Err(Some(1)),
            }
            let value = utf8_first_byte(first, 3);
            let value = utf8_acc_cont_byte(value, second);
            let value = utf8_acc_cont_byte(value, get!(cont 2));
            Ok((value, 3))
        }
        4 => {
            let second = get!(raw 1);
            match (first, second) {
                (0xF0, 0x90..=0xBF) | (0xF1..=0xF3, 0x80..=0xBF) | (0xF4, 0x80..=0x8F) => {}
                _ => return Err(Some(1)),
            }
            let value = utf8_first_byte(first, 4);
            let value = utf8_acc_cont_byte(value, second);
            let value = utf8_acc_cont_byte(value, get!(cont 2));
            let value = utf8_acc_cont_byte(value, get!(cont 3));
            Ok((value, 4))
        }
        _ => Err(Some(1)),
    }
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
