//! Helper code for character escaping.

use crate::ascii;
use crate::fmt::{self, Write};
use crate::marker::PhantomData;
use crate::num::NonZero;
use crate::ops::Range;

const HEX_DIGITS: [ascii::Char; 16] = *b"0123456789abcdef".as_ascii().unwrap();

/// Escapes a character with `\x` representation.
///
/// Returns a buffer with the escaped representation and its corresponding range.
#[inline]
const fn backslash<const N: usize>(a: ascii::Char) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 2) };

    let mut output = [ascii::Char::Null; N];

    output[0] = ascii::Char::ReverseSolidus;
    output[1] = a;

    (output, 0..2)
}

/// Escapes a character with `\xNN` representation.
///
/// Returns a buffer with the escaped representation and its corresponding range.
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

/// Returns a buffer with the verbatim character and its corresponding range.
#[inline]
const fn verbatim<const N: usize>(a: ascii::Char) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 1) };

    let mut output = [ascii::Char::Null; N];

    output[0] = a;

    (output, 0..1)
}

/// Escapes an ASCII character.
///
/// Returns a buffer with the escaped representation and its corresponding range.
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

/// Escapes a character with `\u{NNNN}` representation.
///
/// Returns a buffer with the escaped representation and its corresponding range.
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

/// An iterator over a possibly escaped character.
#[derive(Clone)]
pub(crate) struct EscapeIterInner<const N: usize, ESCAPING> {
    // Invariant:
    //
    // If `alive.end <= Self::LITERAL_ESCAPE_START`, `data` must contain
    // printable ASCII characters in the `alive` range of its `escape_seq` variant.
    //
    // If `alive.end > Self::LITERAL_ESCAPE_START`, `data` must contain a
    // `char` in its `literal` variant, and the `alive` range must have a
    // length of at most `1`.
    data: MaybeEscapedCharacter<N>,
    alive: Range<u8>,
    escaping: PhantomData<ESCAPING>,
}

impl<const N: usize, ESCAPING> EscapeIterInner<N, ESCAPING> {
    const LITERAL_ESCAPE_START: u8 = 128;

    /// # Safety
    ///
    /// `data.escape_seq` must contain an escape sequence in the range given by `alive`.
    #[inline]
    const unsafe fn new(data: MaybeEscapedCharacter<N>, alive: Range<u8>) -> Self {
        // Longer escape sequences are not useful given `alive.end` is at most
        // `Self::LITERAL_ESCAPE_START`.
        const { assert!(N < Self::LITERAL_ESCAPE_START as usize) };

        // Check bounds, which implicitly also checks the invariant
        // `alive.end <= Self::LITERAL_ESCAPE_START`.
        debug_assert!(alive.end <= (N + 1) as u8);

        Self { data, alive, escaping: PhantomData }
    }

    pub(crate) const fn backslash(c: ascii::Char) -> Self {
        let (escape_seq, alive) = backslash(c);
        // SAFETY: `escape_seq` contains an escape sequence in the range given by `alive`.
        unsafe { Self::new(MaybeEscapedCharacter { escape_seq }, alive) }
    }

    pub(crate) const fn ascii(c: u8) -> Self {
        let (escape_seq, alive) = escape_ascii(c);
        // SAFETY: `escape_seq` contains an escape sequence in the range given by `alive`.
        unsafe { Self::new(MaybeEscapedCharacter { escape_seq }, alive) }
    }

    pub(crate) const fn unicode(c: char) -> Self {
        let (escape_seq, alive) = escape_unicode(c);
        // SAFETY: `escape_seq` contains an escape sequence in the range given by `alive`.
        unsafe { Self::new(MaybeEscapedCharacter { escape_seq }, alive) }
    }

    #[inline]
    pub(crate) const fn empty() -> Self {
        // SAFETY: `0..0` ensures an empty escape sequence.
        unsafe { Self::new(MaybeEscapedCharacter { escape_seq: [ascii::Char::Null; N] }, 0..0) }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        usize::from(self.alive.end - self.alive.start)
    }

    #[inline]
    pub(crate) fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_by(n)
    }

    #[inline]
    pub(crate) fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_back_by(n)
    }

    /// Returns a `char` if `self.data` contains one in its `literal` variant.
    #[inline]
    const fn to_char(&self) -> Option<char> {
        if self.alive.end > Self::LITERAL_ESCAPE_START {
            // SAFETY: We just checked that `self.data` contains a `char` in
            //         its `literal` variant.
            return Some(unsafe { self.data.literal });
        }

        None
    }

    /// Returns the printable ASCII characters in the `escape_seq` variant of `self.data`
    /// as a string.
    ///
    /// # Safety
    ///
    /// - `self.data` must contain printable ASCII characters in its `escape_seq` variant.
    /// - `self.alive` must be a valid range for `self.data.escape_seq`.
    #[inline]
    unsafe fn to_str_unchecked(&self) -> &str {
        debug_assert!(self.alive.end <= Self::LITERAL_ESCAPE_START);

        // SAFETY: The caller guarantees `self.data` contains printable ASCII
        //         characters in its `escape_seq` variant, and `self.alive` is
        //         a valid range for `self.data.escape_seq`.
        unsafe {
            self.data
                .escape_seq
                .get_unchecked(usize::from(self.alive.start)..usize::from(self.alive.end))
                .as_str()
        }
    }
}

impl<const N: usize> EscapeIterInner<N, AlwaysEscaped> {
    pub(crate) fn next(&mut self) -> Option<u8> {
        let i = self.alive.next()?;

        // SAFETY: The `AlwaysEscaped` marker guarantees that `self.data`
        //         contains printable ASCII characters in its `escape_seq`
        //         variant, and `i` is guaranteed to be a valid index for
        //         `self.data.escape_seq`.
        unsafe { Some(self.data.escape_seq.get_unchecked(usize::from(i)).to_u8()) }
    }

    pub(crate) fn next_back(&mut self) -> Option<u8> {
        let i = self.alive.next_back()?;

        // SAFETY: The `AlwaysEscaped` marker guarantees that `self.data`
        //         contains printable ASCII characters in its `escape_seq`
        //         variant, and `i` is guaranteed to be a valid index for
        //         `self.data.escape_seq`.
        unsafe { Some(self.data.escape_seq.get_unchecked(usize::from(i)).to_u8()) }
    }
}

impl<const N: usize> EscapeIterInner<N, MaybeEscaped> {
    // This is the only way to create any `EscapeIterInner` containing a `char` in
    // the `literal` variant of its `self.data`, meaning the `AlwaysEscaped` marker
    // guarantees that `self.data` contains printable ASCII characters in its
    // `escape_seq` variant.
    pub(crate) const fn printable(c: char) -> Self {
        Self {
            data: MaybeEscapedCharacter { literal: c },
            // Uphold the invariant `alive.end > Self::LITERAL_ESCAPE_START`, and ensure
            // `len` behaves correctly for iterating through one character literal.
            alive: Self::LITERAL_ESCAPE_START..(Self::LITERAL_ESCAPE_START + 1),
            escaping: PhantomData,
        }
    }

    pub(crate) fn next(&mut self) -> Option<char> {
        let i = self.alive.next()?;

        if let Some(c) = self.to_char() {
            return Some(c);
        }

        // SAFETY: At this point, `self.data` must contain printable ASCII
        //         characters in its `escape_seq` variant, and `i` is
        //         guaranteed to be a valid index for `self.data.escape_seq`.
        Some(char::from(unsafe { self.data.escape_seq.get_unchecked(usize::from(i)).to_u8() }))
    }
}

impl<const N: usize> fmt::Display for EscapeIterInner<N, AlwaysEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: The `AlwaysEscaped` marker guarantees that `self.data`
        //         contains printable ASCII chars, and `self.alive` is
        //         guaranteed to be a valid range for `self.data`.
        f.write_str(unsafe { self.to_str_unchecked() })
    }
}

impl<const N: usize> fmt::Display for EscapeIterInner<N, MaybeEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(c) = self.to_char() {
            return f.write_char(c);
        }

        // SAFETY: At this point, `self.data` must contain printable ASCII
        //         characters in its `escape_seq` variant, and `self.alive`
        //         is guaranteed to be a valid range for `self.data`.
        f.write_str(unsafe { self.to_str_unchecked() })
    }
}

impl<const N: usize> fmt::Debug for EscapeIterInner<N, AlwaysEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EscapeIterInner").field(&format_args!("'{}'", self)).finish()
    }
}

impl<const N: usize> fmt::Debug for EscapeIterInner<N, MaybeEscaped> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("EscapeIterInner").field(&format_args!("'{}'", self)).finish()
    }
}
