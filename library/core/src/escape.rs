//! Helper code for character escaping.

use crate::ascii;
use crate::num::NonZero;
use crate::ops::Range;

const HEX_DIGITS: [ascii::Char; 16] = *b"0123456789abcdef".as_ascii().unwrap();

/// Escapes a byte into provided buffer; returns length of escaped
/// representation.
pub(crate) fn escape_ascii_into(output: &mut [ascii::Char; 4], byte: u8) -> Range<u8> {
    #[inline]
    fn backslash(a: ascii::Char) -> ([ascii::Char; 4], u8) {
        ([ascii::Char::ReverseSolidus, a, ascii::Char::Null, ascii::Char::Null], 2)
    }

    let (data, len) = match byte {
        b'\t' => backslash(ascii::Char::SmallT),
        b'\r' => backslash(ascii::Char::SmallR),
        b'\n' => backslash(ascii::Char::SmallN),
        b'\\' => backslash(ascii::Char::ReverseSolidus),
        b'\'' => backslash(ascii::Char::Apostrophe),
        b'\"' => backslash(ascii::Char::QuotationMark),
        _ => {
            if let Some(a) = byte.as_ascii()
                && !byte.is_ascii_control()
            {
                ([a, ascii::Char::Null, ascii::Char::Null, ascii::Char::Null], 1)
            } else {
                let hi = HEX_DIGITS[usize::from(byte >> 4)];
                let lo = HEX_DIGITS[usize::from(byte & 0xf)];
                ([ascii::Char::ReverseSolidus, ascii::Char::SmallX, hi, lo], 4)
            }
        }
    };
    *output = data;
    0..len
}

/// Escapes a character into provided buffer using `\u{NNNN}` representation.
pub(crate) fn escape_unicode_into(output: &mut [ascii::Char; 10], ch: char) -> Range<u8> {
    output[9] = ascii::Char::RightCurlyBracket;

    let ch = ch as u32;
    output[3] = HEX_DIGITS[((ch >> 20) & 15) as usize];
    output[4] = HEX_DIGITS[((ch >> 16) & 15) as usize];
    output[5] = HEX_DIGITS[((ch >> 12) & 15) as usize];
    output[6] = HEX_DIGITS[((ch >> 8) & 15) as usize];
    output[7] = HEX_DIGITS[((ch >> 4) & 15) as usize];
    output[8] = HEX_DIGITS[((ch >> 0) & 15) as usize];

    // or-ing 1 ensures that for ch==0 the code computes that one digit should
    // be printed.
    let start = (ch | 1).leading_zeros() as usize / 4 - 2;
    const UNICODE_ESCAPE_PREFIX: &[ascii::Char; 3] = b"\\u{".as_ascii().unwrap();
    output[start..][..3].copy_from_slice(UNICODE_ESCAPE_PREFIX);

    (start as u8)..10
}

/// An iterator over an fixed-size array.
///
/// This is essentially equivalent to array’s IntoIter except that indexes are
/// limited to u8 to reduce size of the structure.
#[derive(Clone, Debug)]
pub(crate) struct EscapeIterInner<const N: usize> {
    // The element type ensures this is always ASCII, and thus also valid UTF-8.
    pub(crate) data: [ascii::Char; N],

    // Invariant: alive.start <= alive.end <= N.
    pub(crate) alive: Range<u8>,
}

impl<const N: usize> EscapeIterInner<N> {
    pub fn new(data: [ascii::Char; N], alive: Range<u8>) -> Self {
        const { assert!(N < 256) };
        debug_assert!(alive.start <= alive.end && usize::from(alive.end) <= N, "{alive:?}");
        Self { data, alive }
    }

    pub fn from_array<const M: usize>(array: [ascii::Char; M]) -> Self {
        const { assert!(M <= N) };

        let mut data = [ascii::Char::Null; N];
        data[..M].copy_from_slice(&array);
        Self::new(data, 0..M as u8)
    }

    pub fn as_ascii(&self) -> &[ascii::Char] {
        &self.data[usize::from(self.alive.start)..usize::from(self.alive.end)]
    }

    pub fn as_str(&self) -> &str {
        self.as_ascii().as_str()
    }

    pub fn len(&self) -> usize {
        usize::from(self.alive.end - self.alive.start)
    }

    pub fn next(&mut self) -> Option<u8> {
        self.alive.next().map(|i| self.data[usize::from(i)].to_u8())
    }

    pub fn next_back(&mut self) -> Option<u8> {
        self.alive.next_back().map(|i| self.data[usize::from(i)].to_u8())
    }

    pub fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_by(n)
    }

    pub fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_back_by(n)
    }
}
