//! Helper code for character escaping.

use crate::num::NonZeroUsize;
use crate::ops::Range;

const HEX_DIGITS: [u8; 16] = *b"0123456789abcdef";

/// Escapes a byte into provided buffer; returns length of escaped
/// representation.
pub(crate) fn escape_ascii_into(output: &mut [u8; 4], byte: u8) -> Range<u8> {
    let (data, len) = match byte {
        b'\t' => ([b'\\', b't', 0, 0], 2),
        b'\r' => ([b'\\', b'r', 0, 0], 2),
        b'\n' => ([b'\\', b'n', 0, 0], 2),
        b'\\' => ([b'\\', b'\\', 0, 0], 2),
        b'\'' => ([b'\\', b'\'', 0, 0], 2),
        b'"' => ([b'\\', b'"', 0, 0], 2),
        b'\x20'..=b'\x7e' => ([byte, 0, 0, 0], 1),
        _ => {
            let hi = HEX_DIGITS[usize::from(byte >> 4)];
            let lo = HEX_DIGITS[usize::from(byte & 0xf)];
            ([b'\\', b'x', hi, lo], 4)
        }
    };
    *output = data;
    0..(len as u8)
}

/// Escapes a character into provided buffer using `\u{NNNN}` representation.
pub(crate) fn escape_unicode_into(output: &mut [u8; 10], ch: char) -> Range<u8> {
    output[9] = b'}';

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
    output[start..start + 3].copy_from_slice(b"\\u{");

    (start as u8)..10
}

/// An iterator over an fixed-size array.
///
/// This is essentially equivalent to array’s IntoIter except that indexes are
/// limited to u8 to reduce size of the structure.
#[derive(Clone, Debug)]
pub(crate) struct EscapeIterInner<const N: usize> {
    // Invariant: data[alive] is all ASCII.
    pub(crate) data: [u8; N],

    // Invariant: alive.start <= alive.end <= N.
    pub(crate) alive: Range<u8>,
}

impl<const N: usize> EscapeIterInner<N> {
    pub fn new(data: [u8; N], alive: Range<u8>) -> Self {
        const { assert!(N < 256) };
        debug_assert!(alive.start <= alive.end && usize::from(alive.end) <= N, "{alive:?}");
        let this = Self { data, alive };
        debug_assert!(this.as_bytes().is_ascii(), "Expected ASCII, got {:?}", this.as_bytes());
        this
    }

    fn as_bytes(&self) -> &[u8] {
        &self.data[usize::from(self.alive.start)..usize::from(self.alive.end)]
    }

    pub fn as_str(&self) -> &str {
        // SAFETY: self.data[self.alive] is all ASCII characters.
        unsafe { crate::str::from_utf8_unchecked(self.as_bytes()) }
    }

    pub fn len(&self) -> usize {
        usize::from(self.alive.end - self.alive.start)
    }

    pub fn next(&mut self) -> Option<u8> {
        self.alive.next().map(|i| self.data[usize::from(i)])
    }

    pub fn next_back(&mut self) -> Option<u8> {
        self.alive.next_back().map(|i| self.data[usize::from(i)])
    }

    pub fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.alive.advance_by(n)
    }

    pub fn advance_back_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        self.alive.advance_back_by(n)
    }
}
