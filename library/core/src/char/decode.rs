//! UTF-8 and UTF-16 decoding iterators

use crate::fmt;

use super::from_u32_unchecked;

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
