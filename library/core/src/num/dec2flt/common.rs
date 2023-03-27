//! Common utilities, for internal use only.

use crate::ptr;

/// Helper methods to process immutable bytes.
pub(crate) trait ByteSlice: AsRef<[u8]> {
    unsafe fn first_unchecked(&self) -> u8 {
        debug_assert!(!self.is_empty());
        // SAFETY: safe as long as self is not empty
        unsafe { *self.as_ref().get_unchecked(0) }
    }

    /// Get if the slice contains no elements.
    fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    /// Check if the slice at least `n` length.
    fn check_len(&self, n: usize) -> bool {
        n <= self.as_ref().len()
    }

    /// Check if the first character in the slice is equal to c.
    fn first_is(&self, c: u8) -> bool {
        self.as_ref().first() == Some(&c)
    }

    /// Check if the first character in the slice is equal to c1 or c2.
    fn first_is2(&self, c1: u8, c2: u8) -> bool {
        if let Some(&c) = self.as_ref().first() { c == c1 || c == c2 } else { false }
    }

    /// Bounds-checked test if the first character in the slice is a digit.
    fn first_isdigit(&self) -> bool {
        if let Some(&c) = self.as_ref().first() { c.is_ascii_digit() } else { false }
    }

    /// Check if self starts with u with a case-insensitive comparison.
    fn starts_with_ignore_case(&self, u: &[u8]) -> bool {
        debug_assert!(self.as_ref().len() >= u.len());
        let iter = self.as_ref().iter().zip(u.iter());
        let d = iter.fold(0, |i, (&x, &y)| i | (x ^ y));
        d == 0 || d == 32
    }

    /// Get the remaining slice after the first N elements.
    fn advance(&self, n: usize) -> &[u8] {
        &self.as_ref()[n..]
    }

    /// Get the slice after skipping all leading characters equal c.
    fn skip_chars(&self, c: u8) -> &[u8] {
        let mut s = self.as_ref();
        while s.first_is(c) {
            s = s.advance(1);
        }
        s
    }

    /// Get the slice after skipping all leading characters equal c1 or c2.
    fn skip_chars2(&self, c1: u8, c2: u8) -> &[u8] {
        let mut s = self.as_ref();
        while s.first_is2(c1, c2) {
            s = s.advance(1);
        }
        s
    }

    /// Read 8 bytes as a 64-bit integer in little-endian order.
    unsafe fn read_u64_unchecked(&self) -> u64 {
        debug_assert!(self.check_len(8));
        let src = self.as_ref().as_ptr() as *const u64;
        // SAFETY: safe as long as self is at least 8 bytes
        u64::from_le(unsafe { ptr::read_unaligned(src) })
    }

    /// Try to read the next 8 bytes from the slice.
    fn read_u64(&self) -> Option<u64> {
        if self.check_len(8) {
            // SAFETY: self must be at least 8 bytes.
            Some(unsafe { self.read_u64_unchecked() })
        } else {
            None
        }
    }

    /// Calculate the offset of slice from another.
    fn offset_from(&self, other: &Self) -> isize {
        other.as_ref().len() as isize - self.as_ref().len() as isize
    }
}

impl ByteSlice for [u8] {}

/// Helper methods to process mutable bytes.
pub(crate) trait ByteSliceMut: AsMut<[u8]> {
    /// Write a 64-bit integer as 8 bytes in little-endian order.
    unsafe fn write_u64_unchecked(&mut self, value: u64) {
        debug_assert!(self.as_mut().len() >= 8);
        let dst = self.as_mut().as_mut_ptr() as *mut u64;
        // NOTE: we must use `write_unaligned`, since dst is not
        // guaranteed to be properly aligned. Miri will warn us
        // if we use `write` instead of `write_unaligned`, as expected.
        // SAFETY: safe as long as self is at least 8 bytes
        unsafe {
            ptr::write_unaligned(dst, u64::to_le(value));
        }
    }
}

impl ByteSliceMut for [u8] {}

/// Bytes wrapper with specialized methods for ASCII characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AsciiStr<'a> {
    slc: &'a [u8],
}

impl<'a> AsciiStr<'a> {
    pub fn new(slc: &'a [u8]) -> Self {
        Self { slc }
    }

    /// Advance the view by n, advancing it in-place to (n..).
    pub unsafe fn step_by(&mut self, n: usize) -> &mut Self {
        // SAFETY: safe as long n is less than the buffer length
        self.slc = unsafe { self.slc.get_unchecked(n..) };
        self
    }

    /// Advance the view by n, advancing it in-place to (1..).
    pub unsafe fn step(&mut self) -> &mut Self {
        // SAFETY: safe as long as self is not empty
        unsafe { self.step_by(1) }
    }

    /// Iteratively parse and consume digits from bytes.
    pub fn parse_digits(&mut self, mut func: impl FnMut(u8)) {
        while let Some(&c) = self.as_ref().first() {
            let c = c.wrapping_sub(b'0');
            if c < 10 {
                func(c);
                // SAFETY: self cannot be empty
                unsafe {
                    self.step();
                }
            } else {
                break;
            }
        }
    }
}

impl<'a> AsRef<[u8]> for AsciiStr<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.slc
    }
}

impl<'a> ByteSlice for AsciiStr<'a> {}

/// Determine if 8 bytes are all decimal digits.
/// This does not care about the order in which the bytes were loaded.
pub(crate) fn is_8digits(v: u64) -> bool {
    let a = v.wrapping_add(0x4646_4646_4646_4646);
    let b = v.wrapping_sub(0x3030_3030_3030_3030);
    (a | b) & 0x8080_8080_8080_8080 == 0
}

/// Iteratively parse and consume digits from bytes.
pub(crate) fn parse_digits(s: &mut &[u8], mut f: impl FnMut(u8)) {
    while let Some(&c) = s.get(0) {
        let c = c.wrapping_sub(b'0');
        if c < 10 {
            f(c);
            *s = s.advance(1);
        } else {
            break;
        }
    }
}

/// A custom 64-bit floating point type, representing `f * 2^e`.
/// e is biased, so it be directly shifted into the exponent bits.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct BiasedFp {
    /// The significant digits.
    pub f: u64,
    /// The biased, binary exponent.
    pub e: i32,
}

impl BiasedFp {
    #[inline]
    pub const fn zero_pow2(e: i32) -> Self {
        Self { f: 0, e }
    }
}
