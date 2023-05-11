use crate::ascii;

#[cfg(not(test))]
impl<const N: usize> [u8; N] {
    /// Converts this array of bytes into a array of ASCII characters,
    /// or returns `None` if any of the characters is non-ASCII.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char)]
    /// #![feature(const_option)]
    ///
    /// const HEX_DIGITS: [std::ascii::Char; 16] =
    ///     *b"0123456789abcdef".as_ascii().unwrap();
    ///
    /// assert_eq!(HEX_DIGITS[1].as_str(), "1");
    /// assert_eq!(HEX_DIGITS[10].as_str(), "a");
    /// ```
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[must_use]
    #[inline]
    pub const fn as_ascii(&self) -> Option<&[ascii::Char; N]> {
        if self.is_ascii() {
            // SAFETY: Just checked that it's ASCII
            Some(unsafe { self.as_ascii_unchecked() })
        } else {
            None
        }
    }

    /// Converts this array of bytes into a array of ASCII characters,
    /// without checking whether they're valid.
    ///
    /// # Safety
    ///
    /// Every byte in the array must be in `0..=127`, or else this is UB.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[must_use]
    #[inline]
    pub const unsafe fn as_ascii_unchecked(&self) -> &[ascii::Char; N] {
        let byte_ptr: *const [u8; N] = self;
        let ascii_ptr = byte_ptr as *const [ascii::Char; N];
        // SAFETY: The caller promised all the bytes are ASCII
        unsafe { &*ascii_ptr }
    }
}
