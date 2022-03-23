use crate::fmt;
use crate::iter::FusedIterator;
use crate::ops::Range;
use crate::str::from_utf8_unchecked;

/// An iterator over the escaped version of a byte.
///
/// This `struct` is created by the [`escape_ascii`] method. See its
/// documentation for more.
///
/// [`escape_ascii`]: u8::escape_ascii
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone)]
#[unstable(feature = "escape_ascii_rename", issue = "93887")]
pub struct EscapeAscii {
    range: Range<u8>,
    data: [u8; 4],
}
impl EscapeAscii {
    pub(super) fn new(byte: u8) -> EscapeAscii {
        let (data, len) = match byte {
            b'\t' => ([b'\\', b't', 0, 0], 2),
            b'\r' => ([b'\\', b'r', 0, 0], 2),
            b'\n' => ([b'\\', b'n', 0, 0], 2),
            b'\\' => ([b'\\', b'\\', 0, 0], 2),
            b'\'' => ([b'\\', b'\'', 0, 0], 2),
            b'"' => ([b'\\', b'"', 0, 0], 2),
            b'\x20'..=b'\x7e' => ([byte, 0, 0, 0], 1),
            _ => {
                let hex_digits: &[u8; 16] = b"0123456789abcdef";
                (
                    [
                        b'\\',
                        b'x',
                        hex_digits[(byte >> 4) as usize],
                        hex_digits[(byte & 0xf) as usize],
                    ],
                    4,
                )
            }
        };
        EscapeAscii { range: 0..len, data }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for EscapeAscii {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        self.range.next().map(|i| self.data[i as usize])
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
    fn last(mut self) -> Option<u8> {
        self.next_back()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl DoubleEndedIterator for EscapeAscii {
    fn next_back(&mut self) -> Option<u8> {
        self.range.next_back().map(|i| self.data[i as usize])
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl ExactSizeIterator for EscapeAscii {}
#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for EscapeAscii {}

#[stable(feature = "ascii_escape_display", since = "1.39.0")]
impl fmt::Display for EscapeAscii {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: ok because `escape_ascii` created only valid utf-8 data
        f.write_str(unsafe {
            from_utf8_unchecked(&self.data[(self.range.start as usize)..(self.range.end as usize)])
        })
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for EscapeAscii {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EscapeAscii").finish_non_exhaustive()
    }
}
