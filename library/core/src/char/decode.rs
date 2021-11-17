//! UTF-8 and UTF-16 decoding iterators

use crate::fmt;

use super::{from_u32, from_u32_unchecked};

/// An iterator that decodes UTF-8 encoded code points from an iterator of `u8`s.
///
/// This `struct` is created by the [`decode_utf8`] method on [`char`]. See its
/// documentation for more.
///
/// [`decode_utf8`]: char::decode_utf8
#[derive(Clone, Debug)]
pub struct DecodeUtf8<I>
where
    I: Iterator<Item = u8>,
{
    iter: I,
    buf: DecodeUtf8Buffer,
}

/// An error that can be returned when decoding UTF-8 code points.
///
/// This `struct` is created when using the [`DecodeUtf8`] type.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecodeUtf8Error {
    code: u8,
}

/// Creates an iterator over the UTF-8 encoded code points in `iter`, returning
/// invalid bytes as `Err`s.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::char::decode_utf8;
///
/// // 🦀the<invalid>crab<invalid>
/// let v = b"\xf0\x9f\xa6\x80the\xFFcrab\x80";
///
/// assert_eq!(
///     decode_utf8(v.iter().copied())
///         .map(|r| r.map_err(|e| e.invalid_byte()))
///         .collect::<Vec<_>>(),
///     vec![
///         Ok('🦀'),
///         Ok('t'), Ok('h'), Ok('e'),
///         Err(0xFF),
///         Ok('c'), Ok('r'), Ok('a'), Ok('b'),
///         Err(0x80),
///     ]
/// );
/// ```
///
/// A lossy decoder can be obtained by replacing `Err` results with the replacement character:
///
/// ```
/// use std::char::{decode_utf8, REPLACEMENT_CHARACTER};
///
/// // 🦀the<invalid>crab<invalid>
/// let v = b"\xf0\x9f\xa6\x80the\xFFcrab\x80";
///
/// assert_eq!(
///     decode_utf8(v.iter().copied())
///        .map(|r| r.unwrap_or(REPLACEMENT_CHARACTER))
///        .collect::<String>(),
///     "🦀the�crab�"
/// );
/// ```
#[inline]
pub fn decode_utf8<I: IntoIterator<Item = u8>>(iter: I) -> DecodeUtf8<I::IntoIter> {
    DecodeUtf8 {
        iter: iter.into_iter(),
        buf: DecodeUtf8Buffer::Empty,
    }
}

#[derive(Clone, Debug)]
enum DecodeUtf8Buffer {
    Empty,
    One(u8),
    Two(u8, u8),
    Three(u8, u8, u8),
}

impl<I: Iterator<Item = u8>> Iterator for DecodeUtf8<I> {
    type Item = Result<char, DecodeUtf8Error>;

    fn next(&mut self) -> Option<Result<char, DecodeUtf8Error>> {
        use DecodeUtf8Buffer::*;

        macro_rules! valid_cont {
            ($cont:expr) => {
                (0b1000_0000..=0b1011_1111).contains(&$cont)
            };
        }

        macro_rules! err {
            ($c:expr) => {
                return Some(Err(DecodeUtf8Error { code: $c }))
            };
        }

        #[inline(always)]
        fn from_utf8x2(c1: u8, c2: u8) -> char {
            let c = (c2 as u32 & 0b0011_1111) + ((c1 as u32 & 0b0001_1111) << 6);
            // SAFETY: the number is less than 0xd800
            unsafe { from_u32_unchecked(c) }
        }

        #[inline(always)]
        fn from_utf8x3(c1: u8, c2: u8, c3: u8) -> Option<char> {
            from_u32(
                (c3 as u32 & 0b0011_1111)
                    + ((c2 as u32 & 0b0011_1111) << 6)
                    + ((c1 as u32 & 0b0000_1111) << 12),
            )
        }

        #[inline(always)]
        fn from_utf8x4(c1: u8, c2: u8, c3: u8, c4: u8) -> Option<char> {
            from_u32(
                (c4 as u32 & 0b0011_1111)
                    + ((c3 as u32 & 0b0011_1111) << 6)
                    + ((c2 as u32 & 0b0011_1111) << 12)
                    + ((c1 as u32 & 0b0000_0111) << 18),
            )
        }

        loop {
            match self.buf {
                Empty | One(_) => {
                    // Empty buffer: Test the next character for utf-8-ness
                    let c = match self.buf {
                        Empty => self.iter.next()?,
                        One(c) => {
                            self.buf = Empty;
                            c
                        }
                        _ => unreachable!(),
                    };
                    match c {
                        // ASCII
                        0..=0x7f => return Some(Ok(c as char)),
                        // Start byte
                        0b1100_0010..=0b1101_1111
                        | 0b1110_0000..=0b1110_1111
                        | 0b1111_0000..=0b1111_0111 => {
                            if let Some(cont) = self.iter.next() {
                                self.buf = Two(c, cont); // push2
                            } else {
                                err!(c);
                            }
                        }
                        // Continuation byte or Invalid byte
                        _ => err!(c),
                    }
                }
                Two(c1, c2) => {
                    // in: 2
                    // out: 0j, 1j, 3
                    match c1 {
                        // ASCII
                        0..=0x7f => {
                            self.buf = One(c2); // pop
                            return Some(Ok(c1 as char));
                        }
                        // Start byte for 2
                        0b1100_0010..=0b1101_1111 => {
                            if valid_cont!(c2) {
                                self.buf = Empty; // pop2
                                return Some(Ok(from_utf8x2(c1, c2)));
                            } else {
                                self.buf = One(c2); // pop
                                err!(c1);
                            }
                        }
                        // Start byte for 3 or 4
                        0b1110_0000..=0b1110_1111 | 0b1111_0000..=0b1111_0111 => {
                            if let Some(cont) = self.iter.next() {
                                self.buf = Three(c1, c2, cont); // push
                            } else {
                                self.buf = One(c2); // pop
                                err!(c1);
                            }
                        }
                        // Continuation byte or Invalid byte
                        _ => {
                            self.buf = One(c2);
                            err!(c1);
                        }
                    }
                }
                Three(c1, c2, c3) => {
                    // in: 3
                    // out: 0j, 1j, 2j, 3j
                    match c1 {
                        // ASCII
                        0..=0x7f => {
                            self.buf = Two(c2, c3); // pop
                            return Some(Ok(c1 as char));
                        }
                        // Start byte for 2
                        0b1100_0010..=0b1101_1111 => {
                            if valid_cont!(c2) {
                                self.buf = One(c3); // pop2
                                return Some(Ok(from_utf8x2(c1, c2)));
                            } else {
                                self.buf = Two(c2, c3); // pop
                                err!(c1);
                            }
                        }
                        // Start byte for 3
                        0b1110_0000..=0b1110_1111 => {
                            if valid_cont!(c2) && valid_cont!(c3) {
                                match from_utf8x3(c1, c2, c3) {
                                    Some(c) => {
                                        self.buf = Empty; // pop3
                                        return Some(Ok(c));
                                    }
                                    None => {
                                        // It was in the invalid range
                                        self.buf = Two(c2, c3); // pop
                                        err!(c1);
                                    }
                                }
                            } else {
                                self.buf = Two(c2, c3); // pop
                                err!(c1);
                            }
                        }
                        // Start byte for 4
                        0b1111_0000..=0b1111_0111 => {
                            if let Some(c4) = self.iter.next() {
                                // Handle inline
                                if valid_cont!(c4) {
                                    match from_utf8x4(c1, c2, c3, c4) {
                                        Some(c) => {
                                            self.buf = Empty; // pop3
                                            return Some(Ok(c));
                                        }
                                        None => {
                                            // It was in the invalid range
                                            self.buf = Three(c2, c3, c4); // push/pop
                                            err!(c1);
                                        }
                                    }
                                } else {
                                    self.buf = Three(c2, c3, c4); // push/pop
                                    err!(c1);
                                }
                            } else {
                                self.buf = Two(c2, c3); // pop
                                err!(c1);
                            }
                        }
                        // Continuation byte or Invalid byte
                        _ => {
                            self.buf = Two(c2, c3); // pop
                            err!(c1);
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();
        // we could be entirely 4-byte characters or 1-byte characters
        (low / 4, high)
    }
}

impl DecodeUtf8Error {
    /// Returns the invalid byte which caused this error.
    #[must_use]
    pub fn invalid_byte(&self) -> u8 {
        self.code
    }
}

impl fmt::Display for DecodeUtf8Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid byte found: {:x}", self.code)
    }
}

/// An iterator that decodes UTF-16 encoded code points from an iterator of `u16`s.
///
/// This `struct` is created by the [`decode_utf16`] method on [`char`]. See its
/// documentation for more.
///
/// [`decode_utf16`]: char::decode_utf16
#[stable(feature = "decode_utf16", since = "1.9.0")]
#[derive(Clone, Debug)]
pub struct DecodeUtf16<I>
where
    I: Iterator<Item = u16>,
{
    iter: I,
    buf: Option<u16>,
}

/// An error that can be returned when decoding UTF-16 code points.
///
/// This `struct` is created when using the [`DecodeUtf16`] type.
#[stable(feature = "decode_utf16", since = "1.9.0")]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecodeUtf16Error {
    code: u16,
}

/// Creates an iterator over the UTF-16 encoded code points in `iter`,
/// returning unpaired surrogates as `Err`s.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::char::decode_utf16;
///
/// // 𝄞mus<invalid>ic<invalid>
/// let v = [
///     0xD834, 0xDD1E, 0x006d, 0x0075, 0x0073, 0xDD1E, 0x0069, 0x0063, 0xD834,
/// ];
///
/// assert_eq!(
///     decode_utf16(v.iter().cloned())
///         .map(|r| r.map_err(|e| e.unpaired_surrogate()))
///         .collect::<Vec<_>>(),
///     vec![
///         Ok('𝄞'),
///         Ok('m'), Ok('u'), Ok('s'),
///         Err(0xDD1E),
///         Ok('i'), Ok('c'),
///         Err(0xD834)
///     ]
/// );
/// ```
///
/// A lossy decoder can be obtained by replacing `Err` results with the replacement character:
///
/// ```
/// use std::char::{decode_utf16, REPLACEMENT_CHARACTER};
///
/// // 𝄞mus<invalid>ic<invalid>
/// let v = [
///     0xD834, 0xDD1E, 0x006d, 0x0075, 0x0073, 0xDD1E, 0x0069, 0x0063, 0xD834,
/// ];
///
/// assert_eq!(
///     decode_utf16(v.iter().cloned())
///        .map(|r| r.unwrap_or(REPLACEMENT_CHARACTER))
///        .collect::<String>(),
///     "𝄞mus�ic�"
/// );
/// ```
#[stable(feature = "decode_utf16", since = "1.9.0")]
#[inline]
pub fn decode_utf16<I: IntoIterator<Item = u16>>(iter: I) -> DecodeUtf16<I::IntoIter> {
    DecodeUtf16 { iter: iter.into_iter(), buf: None }
}

#[stable(feature = "decode_utf16", since = "1.9.0")]
impl<I: Iterator<Item = u16>> Iterator for DecodeUtf16<I> {
    type Item = Result<char, DecodeUtf16Error>;

    fn next(&mut self) -> Option<Result<char, DecodeUtf16Error>> {
        let u = match self.buf.take() {
            Some(buf) => buf,
            None => self.iter.next()?,
        };

        if u < 0xD800 || 0xDFFF < u {
            // SAFETY: not a surrogate
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
            // SAFETY: we checked that it's a legal unicode value
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
    #[must_use]
    #[stable(feature = "decode_utf16", since = "1.9.0")]
    pub fn unpaired_surrogate(&self) -> u16 {
        self.code
    }
}

#[stable(feature = "decode_utf16", since = "1.9.0")]
impl fmt::Display for DecodeUtf16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unpaired surrogate found: {:x}", self.code)
    }
}
