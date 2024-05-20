//! Helper code for character escaping.

use crate::ascii;
use crate::mem::MaybeUninit;
use crate::num::NonZero;
use crate::ops::Range;

const HEX_DIGITS: [ascii::Char; 16] = *b"0123456789abcdef".as_ascii().unwrap();

/// An iterator over an fixed-size array.
///
/// This is essentially equivalent to array’s IntoIter except that indexes are
/// limited to u8 to reduce size of the structure.
#[derive(Clone, Debug)]
pub(crate) struct EscapeIterInner<const N: usize> {
    // Invariant: all elements inside the range indexed by `alive` are initialized
    data: [MaybeUninit<ascii::Char>; N],

    // Invariant: `alive.start <= alive.end <= N`
    alive: Range<u8>,
}

impl<const N: usize> EscapeIterInner<N> {
    pub const fn backslash(c: ascii::Char) -> Self {
        const { assert!(N >= 2) };

        let mut data = [MaybeUninit::uninit(); N];

        data[0] = MaybeUninit::new(ascii::Char::ReverseSolidus);
        data[1] = MaybeUninit::new(c);

        Self { data, alive: 0..2 }
    }

    /// Escapes an ASCII character.
    pub const fn ascii(c: u8) -> Self {
        const { assert!(N >= 4) };

        match c {
            b'\t' => Self::backslash(ascii::Char::SmallT),
            b'\r' => Self::backslash(ascii::Char::SmallR),
            b'\n' => Self::backslash(ascii::Char::SmallN),
            b'\\' => Self::backslash(ascii::Char::ReverseSolidus),
            b'\'' => Self::backslash(ascii::Char::Apostrophe),
            b'\"' => Self::backslash(ascii::Char::QuotationMark),
            byte => {
                let mut data = [MaybeUninit::uninit(); N];

                if let Some(c) = byte.as_ascii()
                    && !byte.is_ascii_control()
                {
                    data[0] = MaybeUninit::new(c);
                    Self { data, alive: 0..1 }
                } else {
                    let hi = HEX_DIGITS[(byte >> 4) as usize];
                    let lo = HEX_DIGITS[(byte & 0xf) as usize];

                    data[0] = MaybeUninit::new(ascii::Char::ReverseSolidus);
                    data[1] = MaybeUninit::new(ascii::Char::SmallX);
                    data[2] = MaybeUninit::new(hi);
                    data[3] = MaybeUninit::new(lo);

                    Self { data, alive: 0..4 }
                }
            }
        }
    }

    /// Escapes a character `\u{NNNN}` representation.
    pub const fn unicode(c: char) -> Self {
        const { assert!(N >= 10 && N < u8::MAX as usize) };

        let c = c as u32;

        // OR-ing `1` ensures that for `c == 0` the code computes that
        // one digit should be printed.
        let start = (c | 1).leading_zeros() as usize / 4 - 2;

        let mut data = [MaybeUninit::uninit(); N];
        data[3] = MaybeUninit::new(HEX_DIGITS[((c >> 20) & 15) as usize]);
        data[4] = MaybeUninit::new(HEX_DIGITS[((c >> 16) & 15) as usize]);
        data[5] = MaybeUninit::new(HEX_DIGITS[((c >> 12) & 15) as usize]);
        data[6] = MaybeUninit::new(HEX_DIGITS[((c >> 8) & 15) as usize]);
        data[7] = MaybeUninit::new(HEX_DIGITS[((c >> 4) & 15) as usize]);
        data[8] = MaybeUninit::new(HEX_DIGITS[((c >> 0) & 15) as usize]);
        data[9] = MaybeUninit::new(ascii::Char::RightCurlyBracket);
        data[start + 0] = MaybeUninit::new(ascii::Char::ReverseSolidus);
        data[start + 1] = MaybeUninit::new(ascii::Char::SmallU);
        data[start + 2] = MaybeUninit::new(ascii::Char::LeftCurlyBracket);

        Self { data, alive: start as u8..10 }
    }

    #[inline]
    pub const fn empty() -> Self {
        Self { data: [MaybeUninit::uninit(); N], alive: 0..0 }
    }

    #[inline]
    pub fn as_ascii(&self) -> &[ascii::Char] {
        // SAFETY: the range indexed by `self.alive` is guaranteed to contain valid data.
        unsafe {
            let data = self.data.get_unchecked(self.alive.start as usize..self.alive.end as usize);
            MaybeUninit::slice_assume_init_ref(data)
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        self.as_ascii().as_str()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.alive.len()
    }

    #[inline]
    pub fn next(&mut self) -> Option<u8> {
        let i = self.alive.next()?;

        // SAFETY: the range indexed by `self.alive` is guaranteed to contain initialized data.
        unsafe { Some(MaybeUninit::assume_init_ref(self.data.get_unchecked(i as usize)).to_u8()) }
    }

    #[inline]
    pub fn next_back(&mut self) -> Option<u8> {
        let i = self.alive.next_back()?;

        // SAFETY: the range indexed by `self.alive` is guaranteed to contain initialized data.
        unsafe { Some(MaybeUninit::assume_init_ref(self.data.get_unchecked(i as usize)).to_u8()) }
    }

    #[inline]
    pub fn advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_by(n)
    }

    #[inline]
    pub fn advance_back_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
        self.alive.advance_back_by(n)
    }
}
