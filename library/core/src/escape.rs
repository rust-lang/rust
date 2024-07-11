//! Helper code for character escaping.

use crate::ascii;
use crate::num::NonZero;
use crate::ops::Range;

const HEX_DIGITS: [ascii::Char; 16] = *b"0123456789abcdef".as_ascii().unwrap();

#[inline]
const fn backslash<const N: usize>(a: ascii::Char) -> ([ascii::Char; N], Range<u8>) {
    const { assert!(N >= 2) };

    let mut output = [ascii::Char::Null; N];

    output[0] = ascii::Char::ReverseSolidus;
    output[1] = a;

    (output, 0..2)
}

/// Escapes an ASCII character.
///
/// Returns a buffer and the length of the escaped representation.
const fn escape_ascii<const N: usize>(byte: u8) -> ([ascii::Char; N], Range<u8>) {
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
                (output, 0..1)
            } else {
                let hi = HEX_DIGITS[(byte >> 4) as usize];
                let lo = HEX_DIGITS[(byte & 0xf) as usize];

                output[0] = ascii::Char::ReverseSolidus;
                output[1] = ascii::Char::SmallX;
                output[2] = hi;
                output[3] = lo;

                (output, 0..4)
            }
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
        let (data, range) = backslash(c);
        Self { data, alive: range }
    }

    pub const fn ascii(c: u8) -> Self {
        let (data, range) = escape_ascii(c);
        Self { data, alive: range }
    }

    pub const fn unicode(c: char) -> Self {
        let (data, range) = escape_unicode(c);
        Self { data, alive: range }
    }

    #[inline]
    pub const fn empty() -> Self {
        Self { data: [ascii::Char::Null; N], alive: 0..0 }
    }

    #[inline]
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
