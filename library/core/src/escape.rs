//! Helper code for character escaping.

use crate::fmt::Write;
use crate::marker::PhantomData;
use crate::num::NonZero;
use crate::ops::Range;
use crate::{ascii, fmt};

const HEX_DIGITS: [ascii::Char; 16] = *b"0123456789abcdef".as_ascii().unwrap();

#[inline]
const fn backslash<const N: usize>(a: ascii::Char) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 2) };

    let mut output = [ascii::Char::Null; N];

    output[0] = ascii::Char::ReverseSolidus;
    output[1] = a;

    (output, 0..2)
}

#[inline]
const fn hex_escape<const N: usize>(byte: u8) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 4) };

    let mut output = [ascii::Char::Null; N];

    let hi = HEX_DIGITS[(byte >> 4) as usize];
    let lo = HEX_DIGITS[(byte & 0xf) as usize];

    output[0] = ascii::Char::ReverseSolidus;
    output[1] = ascii::Char::SmallX;
    output[2] = hi;
    output[3] = lo;

    (output, 0..4)
}

#[inline]
const fn verbatim<const N: usize>(a: ascii::Char) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 1) };

    let mut output = [ascii::Char::Null; N];

    output[0] = a;

    (output, 0..1)
}

/// Escapes an ASCII character.
///
/// Returns a buffer and the length of the escaped representation.
const fn escape_ascii<const N: usize>(byte: u8) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 4) };

    #[cfg(feature = "optimize_for_size")]
    {
        match byte {
            b'\t' => backslash(ascii::Char::SmallT),
            b'\r' => backslash(ascii::Char::SmallR),
            b'\n' => backslash(ascii::Char::SmallN),
            b'\\' => backslash(ascii::Char::ReverseSolidus),
            b'\'' => backslash(ascii::Char::Apostrophe),
            b'"' => backslash(ascii::Char::QuotationMark),
            0x00..=0x1F | 0x7F => hex_escape(byte),
            _ => match ascii::Char::from_u8(byte) {
                Some(a) => verbatim(a),
                None => hex_escape(byte),
            },
        }
    }

    #[cfg(not(feature = "optimize_for_size"))]
    {
        /// Lookup table helps us determine how to display character.
        ///
        /// Since ASCII characters will always be 7 bits, we can exploit this to store the 8th bit to
        /// indicate whether the result is escaped or unescaped.
        ///
        /// We additionally use 0x80 (escaped NUL character) to indicate hex-escaped bytes, since
        /// escaped NUL will not occur.
        const LOOKUP: [u8; 256] = {
            let mut arr = [0; 256];
            let mut idx = 0;
            while idx <= 255 {
                arr[idx] = match idx as u8 {
                    // use 8th bit to indicate escaped
                    b'\t' => 0x80 | b't',
                    b'\r' => 0x80 | b'r',
                    b'\n' => 0x80 | b'n',
                    b'\\' => 0x80 | b'\\',
                    b'\'' => 0x80 | b'\'',
                    b'"' => 0x80 | b'"',

                    // use NUL to indicate hex-escaped
                    0x00..=0x1F | 0x7F..=0xFF => 0x80 | b'\0',

                    idx => idx,
                };
                idx += 1;
            }
            arr
        };

        let lookup = LOOKUP[byte as usize];

        // 8th bit indicates escape
        let lookup_escaped = lookup & 0x80 != 0;

        // SAFETY: We explicitly mask out the eighth bit to get a 7-bit ASCII character.
        let lookup_ascii = unsafe { ascii::Char::from_u8_unchecked(lookup & 0x7F) };

        if lookup_escaped {
            // NUL indicates hex-escaped
            if matches!(lookup_ascii, ascii::Char::Null) {
                hex_escape(byte)
            } else {
                backslash(lookup_ascii)
            }
        } else {
            verbatim(lookup_ascii)
        }
    }
}

/// Escapes a character `\u{NNNN}` representation.
///
/// Returns a buffer and the length of the escaped representation.
const fn escape_unicode<const N: usize>(c: char) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 10 && N < u8::MAX as usize) };

    let c = c as u32;

    // OR-ing `1` ensures that for `c == 0` the code computes that
    // one digit should be printed.
    let start = (c | 1).leading_zeros() as usize / 4 - 2;

    let mut output = [ascii::Char::Null; N];
    output[3] = HEX_DIGITS[((c >> 20) & 15) as usize];
    output[4] = HEX_DIGITS[((c >> 16) & 15) as usize];
    output[5] = HEX_DIGITS[((c >> 12) & 15) as usize];
    output[6] = HEX_DIGITS[((c >> 8) & 15) as usize];
    output[7] = HEX_DIGITS[((c >> 4) & 15) as usize];
    output[8] = HEX_DIGITS[((c >> 0) & 15) as usize];
    output[9] = ascii::Char::RightCurlyBracket;
    output[start + 0] = ascii::Char::ReverseSolidus;
    output[start + 1] = ascii::Char::SmallU;
    output[start + 2] = ascii::Char::LeftCurlyBracket;

    (output, (start as u8)..(N as u8))
}

#[derive(Clone, Copy)]
union MaybeEscapedCharacter<const N: usize> {
    pub escape_seq: [ascii::Char; N],
    pub literal: char,
}

/// Marker type to indicate that the character is always escaped,
/// used to optimize the iterator implementation.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub(crate) struct AlwaysEscaped;

/// Marker type to indicate that the character may be escaped,
/// used to optimize the iterator implementation.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub(crate) struct MaybeEscaped;

/// An iterator over a fixed-size array.
///
/// This is essentially equivalent to array’s `IntoIter` except that
/// indices are stored as `u8` to reduce size of the structure.
#[derive(Clone)]
pub(crate) struct EscapeIterInner<const N: usize, ESCAPING> {
    // The element type ensures this is always ASCII, and thus also valid UTF-8.
    // Invariant: `N < 128`. For non-printable characters, contains the escaped
    // representation as ASCII characters. For printable characters, contains the
    // character itself.
    data: MaybeEscapedCharacter<N>,

    // Invariant: For non-printable characters, `alive.start <= alive.end <= N`,
    // for printable characters, `128 <= alive.start <= alive.end`.
    alive: Range<u8>,

    escaping: PhantomData<ESCAPING>,
}

impl<const N: usize, ESCAPING> EscapeIterInner<N, ESCAPING> {
    #[inline]
    const fn new(data: MaybeEscapedCharacter<N>, alive: Range<u8>) -> Self {
        const { assert!(N < 128) };
        Self { data, alive, escaping: PhantomData }
    }

    pub(crate) const fn backslash(c: ascii::Char) -> Self {
        let (data, range) = backslash(c);
        Self::new(MaybeEscapedCharacter { escape_seq: data }, range)
    }

    pub(crate) const fn ascii(c: u8) -> Self {
        let (data, range) = escape_ascii(c);
        Self::new(MaybeEscapedCharacter { escape_seq: data }, range)
    }

    pub(crate) const fn unicode(c: char) -> Self {
        let (data, range) = escape_unicode(c);
        Self::new(MaybeEscapedCharacter { escape_seq: data }, range)
    }

    #[inline]
    pub(crate) const fn empty() -> Self {
        Self::new(MaybeEscapedCharacter { escape_seq: [ascii::Char::Null; N] }, 0..0)
    }

    /// # Safety
    ///
    /// The caller must ensure that `self` contains an escape sequence.
    #[inline]
    unsafe fn as_ascii(&self) -> &[ascii::Char] {
        // SAFETY: `self.data.escape_seq` contains an escape sequence, as is guaranteed
        // by the caller, and `self.alive` is guaranteed to be a valid range for
        // indexing `self.data`.
        unsafe {
            self.data
                .escape_seq
                .get_unchecked(usize::from(self.alive.start)..usize::from(self.alive.end))
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that `self` contains a literal `char`.
    #[inline]
    unsafe fn as_char(&self) -> char {
        // SAFETY: `self.data.literal` contains a literal `char`,
        // as is guaranteed by the caller.
        unsafe { self.data.literal }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        usize::from(self.alive.end - self.alive.start)
    }

    pub(crate) fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_by(n)
    }

    pub(crate) fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_back_by(n)
    }
}

impl<const N: usize> EscapeIterInner<N, AlwaysEscaped> {
    pub(crate) fn next(&mut self) -> Option<u8> {
        let i = self.alive.next()?;

        // SAFETY: The `AlwaysEscaped` marker guarantees that `self` contains an escape sequence,
        // and `i` is guaranteed to be a valid index for `self.data.escape_seq`.
        unsafe { Some(self.data.escape_seq.get_unchecked(usize::from(i)).to_u8()) }
    }

    pub(crate) fn next_back(&mut self) -> Option<u8> {
        let i = self.alive.next_back()?;

        // SAFETY: We just checked that `self` contains an escape sequence, and
        // `i` is guaranteed to be a valid index for `self.data.escape_seq`.
        unsafe { Some(self.data.escape_seq.get_unchecked(usize::from(i)).to_u8()) }
    }
}

impl<const N: usize> EscapeIterInner<N, MaybeEscaped> {
    const CHAR_FLAG: u8 = 0b1000_0000;

    // This is the only way to create an `EscapeIterInner` with a literal `char`, which
    // means the `AlwaysEscaped` marker guarantees that `self` contains an escape sequence.
    pub(crate) const fn printable(c: char) -> Self {
        Self::new(
            MaybeEscapedCharacter { literal: c },
            // Ensure `len` behaves correctly.
            Self::CHAR_FLAG..(Self::CHAR_FLAG + 1),
        )
    }

    #[inline]
    const fn is_literal(&self) -> bool {
        self.alive.start & Self::CHAR_FLAG != 0
    }

    pub(crate) fn next(&mut self) -> Option<char> {
        let i = self.alive.next()?;

        if self.is_literal() {
            // SAFETY: We just checked that `self` contains a literal `char`.
            return Some(unsafe { self.as_char() });
        }

        // SAFETY: We just checked that `self` contains an escape sequence, and
        // `i` is guaranteed to be a valid index for `self.data.escape_seq`.
        Some(char::from(unsafe { self.data.escape_seq.get_unchecked(usize::from(i)).to_u8() }))
    }
}

impl<const N: usize> fmt::Display for EscapeIterInner<N, AlwaysEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: The `AlwaysEscaped` marker guarantees that `self` contains an escape sequence.
        f.write_str(unsafe { self.as_ascii().as_str() })
    }
}

impl<const N: usize> fmt::Display for EscapeIterInner<N, MaybeEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_literal() {
            // SAFETY: We just checked that `self` contains a literal `char`.
            f.write_char(unsafe { self.as_char() })
        } else {
            // SAFETY: We just checked that `self` contains an escape sequence.
            f.write_str(unsafe { self.as_ascii().as_str() })
        }
    }
}

impl<const N: usize> fmt::Debug for EscapeIterInner<N, AlwaysEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EscapeIterInner")
            // SAFETY: The `AlwaysEscaped` marker guarantees that `self` contains an escape sequence.
            .field(unsafe { &self.as_ascii() })
            .finish()
    }
}

impl<const N: usize> fmt::Debug for EscapeIterInner<N, MaybeEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_tuple("EscapeIterInner");
        if self.is_literal() {
            // SAFETY: We just checked that `self` contains a literal `char`.
            d.field(unsafe { &self.as_char() });
        } else {
            // SAFETY: We just checked that `self` contains an escape sequence.
            d.field(unsafe { &self.as_ascii() });
        }
        d.finish()
    }
}
