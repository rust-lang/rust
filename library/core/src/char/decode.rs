//! UTF-8 and UTF-16 decoding iterators

use crate::error::Error;
use crate::fmt;

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
/// returning unpaired surrogates as `Err`s. See [`char::decode_utf16`].
#[inline]
pub(super) fn decode_utf16<I: IntoIterator<Item = u16>>(iter: I) -> DecodeUtf16<I::IntoIter> {
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

        if !u.is_utf16_surrogate() {
            // SAFETY: not a surrogate
            Some(Ok(unsafe { char::from_u32_unchecked(u as u32) }))
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
            let c = (((u & 0x3ff) as u32) << 10 | (u2 & 0x3ff) as u32) + 0x1_0000;
            // SAFETY: we checked that it's a legal unicode value
            Some(Ok(unsafe { char::from_u32_unchecked(c) }))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();

        let (low_buf, high_buf) = match self.buf {
            // buf is empty, no additional elements from it.
            None => (0, 0),
            // `u` is a non surrogate, so it's always an additional character.
            Some(u) if !u.is_utf16_surrogate() => (1, 1),
            // `u` is a leading surrogate (it can never be a trailing surrogate and
            // it's a surrogate due to the previous branch) and `self.iter` is empty.
            //
            // `u` can't be paired, since the `self.iter` is empty,
            // so it will always become an additional element (error).
            Some(_u) if high == Some(0) => (1, 1),
            // `u` is a leading surrogate and `iter` may be non-empty.
            //
            // `u` can either pair with a trailing surrogate, in which case no additional elements
            // are produced, or it can become an error, in which case it's an additional character (error).
            Some(_u) => (0, 1),
        };

        // `self.iter` could contain entirely valid surrogates (2 elements per
        // char), or entirely non-surrogates (1 element per char).
        //
        // On odd lower bound, at least one element must stay unpaired
        // (with other elements from `self.iter`), so we round up.
        let low = low.div_ceil(2) + low_buf;
        let high = high.and_then(|h| h.checked_add(high_buf));

        (low, high)
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

#[stable(feature = "decode_utf16", since = "1.9.0")]
impl Error for DecodeUtf16Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "unpaired surrogate found"
    }
}
