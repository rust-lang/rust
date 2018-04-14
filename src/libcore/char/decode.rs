// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! UTF-8 and UTF-16 decoding iterators

use fmt;
use iter::FusedIterator;
use super::from_u32_unchecked;

/// An iterator over an iterator of bytes of the characters the bytes represent
/// as UTF-8
#[unstable(feature = "decode_utf8", issue = "33906")]
#[rustc_deprecated(since = "1.27.0", reason = "Use str::from_utf8 instead:
    https://doc.rust-lang.org/nightly/std/str/struct.Utf8Error.html#examples")]
#[derive(Clone, Debug)]
#[allow(deprecated)]
pub struct DecodeUtf8<I: Iterator<Item = u8>>(::iter::Peekable<I>);

/// Decodes an `Iterator` of bytes as UTF-8.
#[unstable(feature = "decode_utf8", issue = "33906")]
#[rustc_deprecated(since = "1.27.0", reason = "Use str::from_utf8 instead:
    https://doc.rust-lang.org/nightly/std/str/struct.Utf8Error.html#examples")]
#[allow(deprecated)]
#[inline]
pub fn decode_utf8<I: IntoIterator<Item = u8>>(i: I) -> DecodeUtf8<I::IntoIter> {
    DecodeUtf8(i.into_iter().peekable())
}

/// `<DecodeUtf8 as Iterator>::next` returns this for an invalid input sequence.
#[unstable(feature = "decode_utf8", issue = "33906")]
#[rustc_deprecated(since = "1.27.0", reason = "Use str::from_utf8 instead:
    https://doc.rust-lang.org/nightly/std/str/struct.Utf8Error.html#examples")]
#[derive(PartialEq, Eq, Debug)]
#[allow(deprecated)]
pub struct InvalidSequence(());

#[unstable(feature = "decode_utf8", issue = "33906")]
#[allow(deprecated)]
impl<I: Iterator<Item = u8>> Iterator for DecodeUtf8<I> {
    type Item = Result<char, InvalidSequence>;
    #[inline]

    fn next(&mut self) -> Option<Result<char, InvalidSequence>> {
        self.0.next().map(|first_byte| {
            // Emit InvalidSequence according to
            // Unicode §5.22 Best Practice for U+FFFD Substitution
            // http://www.unicode.org/versions/Unicode9.0.0/ch05.pdf#G40630

            // Roughly: consume at least one byte,
            // then validate one byte at a time and stop before the first unexpected byte
            // (which might be the valid start of the next byte sequence).

            let mut code_point;
            macro_rules! first_byte {
                ($mask: expr) => {
                    code_point = u32::from(first_byte & $mask)
                }
            }
            macro_rules! continuation_byte {
                () => { continuation_byte!(0x80...0xBF) };
                ($range: pat) => {
                    match self.0.peek() {
                        Some(&byte @ $range) => {
                            code_point = (code_point << 6) | u32::from(byte & 0b0011_1111);
                            self.0.next();
                        }
                        _ => return Err(InvalidSequence(()))
                    }
                }
            }

            match first_byte {
                0x00...0x7F => {
                    first_byte!(0b1111_1111);
                }
                0xC2...0xDF => {
                    first_byte!(0b0001_1111);
                    continuation_byte!();
                }
                0xE0 => {
                    first_byte!(0b0000_1111);
                    continuation_byte!(0xA0...0xBF);  // 0x80...0x9F here are overlong
                    continuation_byte!();
                }
                0xE1...0xEC | 0xEE...0xEF => {
                    first_byte!(0b0000_1111);
                    continuation_byte!();
                    continuation_byte!();
                }
                0xED => {
                    first_byte!(0b0000_1111);
                    continuation_byte!(0x80...0x9F);  // 0xA0..0xBF here are surrogates
                    continuation_byte!();
                }
                0xF0 => {
                    first_byte!(0b0000_0111);
                    continuation_byte!(0x90...0xBF);  // 0x80..0x8F here are overlong
                    continuation_byte!();
                    continuation_byte!();
                }
                0xF1...0xF3 => {
                    first_byte!(0b0000_0111);
                    continuation_byte!();
                    continuation_byte!();
                    continuation_byte!();
                }
                0xF4 => {
                    first_byte!(0b0000_0111);
                    continuation_byte!(0x80...0x8F);  // 0x90..0xBF here are beyond char::MAX
                    continuation_byte!();
                    continuation_byte!();
                }
                _ => return Err(InvalidSequence(()))  // Illegal first byte, overlong, or beyond MAX
            }
            unsafe {
                Ok(from_u32_unchecked(code_point))
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.0.size_hint();

        // A code point is at most 4 bytes long.
        let min_code_points = lower / 4;

        (min_code_points, upper)
    }
}

#[unstable(feature = "decode_utf8", issue = "33906")]
#[allow(deprecated)]
impl<I: FusedIterator<Item = u8>> FusedIterator for DecodeUtf8<I> {}

/// An iterator that decodes UTF-16 encoded code points from an iterator of `u16`s.
#[stable(feature = "decode_utf16", since = "1.9.0")]
#[derive(Clone, Debug)]
pub struct DecodeUtf16<I>
    where I: Iterator<Item = u16>
{
    iter: I,
    buf: Option<u16>,
}

/// An error that can be returned when decoding UTF-16 code points.
#[stable(feature = "decode_utf16", since = "1.9.0")]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecodeUtf16Error {
    code: u16,
}

/// Create an iterator over the UTF-16 encoded code points in `iter`,
/// returning unpaired surrogates as `Err`s.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::char::decode_utf16;
///
/// fn main() {
///     // 𝄞mus<invalid>ic<invalid>
///     let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
///              0x0073, 0xDD1E, 0x0069, 0x0063,
///              0xD834];
///
///     assert_eq!(decode_utf16(v.iter().cloned())
///                            .map(|r| r.map_err(|e| e.unpaired_surrogate()))
///                            .collect::<Vec<_>>(),
///                vec![Ok('𝄞'),
///                     Ok('m'), Ok('u'), Ok('s'),
///                     Err(0xDD1E),
///                     Ok('i'), Ok('c'),
///                     Err(0xD834)]);
/// }
/// ```
///
/// A lossy decoder can be obtained by replacing `Err` results with the replacement character:
///
/// ```
/// use std::char::{decode_utf16, REPLACEMENT_CHARACTER};
///
/// fn main() {
///     // 𝄞mus<invalid>ic<invalid>
///     let v = [0xD834, 0xDD1E, 0x006d, 0x0075,
///              0x0073, 0xDD1E, 0x0069, 0x0063,
///              0xD834];
///
///     assert_eq!(decode_utf16(v.iter().cloned())
///                    .map(|r| r.unwrap_or(REPLACEMENT_CHARACTER))
///                    .collect::<String>(),
///                "𝄞mus�ic�");
/// }
/// ```
#[stable(feature = "decode_utf16", since = "1.9.0")]
#[inline]
pub fn decode_utf16<I: IntoIterator<Item = u16>>(iter: I) -> DecodeUtf16<I::IntoIter> {
    DecodeUtf16 {
        iter: iter.into_iter(),
        buf: None,
    }
}

#[stable(feature = "decode_utf16", since = "1.9.0")]
impl<I: Iterator<Item = u16>> Iterator for DecodeUtf16<I> {
    type Item = Result<char, DecodeUtf16Error>;

    fn next(&mut self) -> Option<Result<char, DecodeUtf16Error>> {
        let u = match self.buf.take() {
            Some(buf) => buf,
            None => self.iter.next()?
        };

        if u < 0xD800 || 0xDFFF < u {
            // not a surrogate
            Some(Ok(unsafe { from_u32_unchecked(u as u32) }))
        } else if u >= 0xDC00 {
            // a trailing surrogate
            Some(Err(DecodeUtf16Error { code: u }))
        } else {
            let u2 = match self.iter.next() {
                Some(u2) => u2,
                // eof
                None => return Some(Err(DecodeUtf16Error { code: u })),
            };
            if u2 < 0xDC00 || u2 > 0xDFFF {
                // not a trailing surrogate so we're not a valid
                // surrogate pair, so rewind to redecode u2 next time.
                self.buf = Some(u2);
                return Some(Err(DecodeUtf16Error { code: u }));
            }

            // all ok, so lets decode it.
            let c = (((u - 0xD800) as u32) << 10 | (u2 - 0xDC00) as u32) + 0x1_0000;
            Some(Ok(unsafe { from_u32_unchecked(c) }))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();
        // we could be entirely valid surrogates (2 elements per
        // char), or entirely non-surrogates (1 element per char)
        (low / 2, high)
    }
}

impl DecodeUtf16Error {
    /// Returns the unpaired surrogate which caused this error.
    #[stable(feature = "decode_utf16", since = "1.9.0")]
    pub fn unpaired_surrogate(&self) -> u16 {
        self.code
    }
}

#[stable(feature = "decode_utf16", since = "1.9.0")]
impl fmt::Display for DecodeUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unpaired surrogate found: {:x}", self.code)
    }
}
