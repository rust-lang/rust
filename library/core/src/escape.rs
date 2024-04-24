//! Helper code for character escaping.

use crate::ascii;
use crate::num::NonZero;
use crate::ops::Range;

const HEX_DIGITS: [ascii::Char; 16] = *b"0123456789abcdef".as_ascii().unwrap();

#[inline]
const fn backslash<const N: usize>(a: ascii::Char) -> ([ascii::Char; N], u8) {
    const { assert!(N >= 2) };

    let mut output = [ascii::Char::Null; N];

    output[0] = ascii::Char::ReverseSolidus;
    output[1] = a;

    (output, 2)
}

/// Escapes an ASCII character.
///
/// Returns a buffer and the length of the escaped representation.
const fn escape_ascii<const N: usize>(byte: u8) -> ([ascii::Char; N], u8) {
    const { assert!(N >= 4) };

    match byte {
        b'\t' => backslash(ascii::Char::SmallT),
        b'\r' => backslash(ascii::Char::SmallR),
        b'\n' => backslash(ascii::Char::SmallN),
        b'\\' => backslash(ascii::Char::ReverseSolidus),
        b'\'' => backslash(ascii::Char::Apostrophe),
        b'\"' => backslash(ascii::Char::QuotationMark),
        byte => {
            let mut output = [ascii::Char::Null; N];

            if let Some(c) = byte.as_ascii()
                && !byte.is_ascii_control()
            {
                output[0] = c;
                (output, 1)
            } else {
                let hi = HEX_DIGITS[(byte >> 4) as usize];
                let lo = HEX_DIGITS[(byte & 0xf) as usize];

                output[0] = ascii::Char::ReverseSolidus;
                output[1] = ascii::Char::SmallX;
                output[2] = hi;
                output[3] = lo;

                (output, 4)
            }
        }
    }
}

/// Escapes a character `\u{NNNN}` representation.
///
/// Returns a buffer and the length of the escaped representation.
const fn escape_unicode<const N: usize>(c: char) -> ([ascii::Char; N], u8) {
    const { assert!(N >= 10) };

    let c = c as u32;

    // OR-ing `1` ensures that for `c == 0` the code computes that
    // one digit should be printed.
    let u_len = (8 - (c | 1).leading_zeros() / 4) as usize;

    let closing_paren_offset = 3 + u_len;

    let mut output = [ascii::Char::Null; N];

    output[0] = ascii::Char::ReverseSolidus;
    output[1] = ascii::Char::SmallU;
    output[2] = ascii::Char::LeftCurlyBracket;

    output[3 + u_len.saturating_sub(6)] = HEX_DIGITS[((c >> 20) & 0x0f) as usize];
    output[3 + u_len.saturating_sub(5)] = HEX_DIGITS[((c >> 16) & 0x0f) as usize];
    output[3 + u_len.saturating_sub(4)] = HEX_DIGITS[((c >> 12) & 0x0f) as usize];
    output[3 + u_len.saturating_sub(3)] = HEX_DIGITS[((c >> 8) & 0x0f) as usize];
    output[3 + u_len.saturating_sub(2)] = HEX_DIGITS[((c >> 4) & 0x0f) as usize];
    output[3 + u_len.saturating_sub(1)] = HEX_DIGITS[((c >> 0) & 0x0f) as usize];

    output[closing_paren_offset] = ascii::Char::RightCurlyBracket;

    let len = (closing_paren_offset + 1) as u8;
    (output, len)
}

/// An iterator over an fixed-size array.
///
/// This is essentially equivalent to array’s IntoIter except that indexes are
/// limited to u8 to reduce size of the structure.
#[derive(Clone, Debug)]
pub(crate) struct EscapeIterInner<const N: usize> {
    // The element type ensures this is always ASCII, and thus also valid UTF-8.
    data: [ascii::Char; N],

    // Invariant: `alive.start <= alive.end <= N`
    alive: Range<u8>,
}

impl<const N: usize> EscapeIterInner<N> {
    pub const fn backslash(c: ascii::Char) -> Self {
        let (data, len) = backslash(c);
        Self { data, alive: 0..len }
    }

    pub const fn ascii(c: u8) -> Self {
        let (data, len) = escape_ascii(c);
        Self { data, alive: 0..len }
    }

    pub const fn unicode(c: char) -> Self {
        let (data, len) = escape_unicode(c);
        Self { data, alive: 0..len }
    }

    #[inline]
    pub const fn empty() -> Self {
        Self { data: [ascii::Char::Null; N], alive: 0..0 }
    }

    pub fn as_ascii(&self) -> &[ascii::Char] {
        // SAFETY: `self.alive` is guaranteed to be a valid range for indexing `self.data`.
        unsafe {
            self.data.get_unchecked(usize::from(self.alive.start)..usize::from(self.alive.end))
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        self.as_ascii().as_str()
    }

    #[inline]
    pub fn len(&self) -> usize {
        usize::from(self.alive.end - self.alive.start)
    }

    pub fn next(&mut self) -> Option<u8> {
        let i = self.alive.next()?;

        // SAFETY: `i` is guaranteed to be a valid index for `self.data`.
        unsafe { Some(self.data.get_unchecked(usize::from(i)).to_u8()) }
    }

    pub fn next_back(&mut self) -> Option<u8> {
        let i = self.alive.next_back()?;

        // SAFETY: `i` is guaranteed to be a valid index for `self.data`.
        unsafe { Some(self.data.get_unchecked(usize::from(i)).to_u8()) }
    }

    pub fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_by(n)
    }

    pub fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_back_by(n)
    }
}
