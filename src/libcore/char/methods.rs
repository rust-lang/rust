//! impl char {}

use crate::slice;
use crate::str::from_utf8_unchecked_mut;
use crate::unicode::printable::is_printable;
use crate::unicode::tables::{conversions, derived_property, general_category, property};

use super::*;

#[lang = "char"]
impl char {
    /// Checks if a `char` is a digit in the given radix.
    ///
    /// A 'radix' here is sometimes also called a 'base'. A radix of two
    /// indicates a binary number, a radix of ten, decimal, and a radix of
    /// sixteen, hexadecimal, to give some common values. Arbitrary
    /// radices are supported.
    ///
    /// Compared to `is_numeric()`, this function only recognizes the characters
    /// `0-9`, `a-z` and `A-Z`.
    ///
    /// 'Digit' is defined to be only the following characters:
    ///
    /// * `0-9`
    /// * `a-z`
    /// * `A-Z`
    ///
    /// For a more comprehensive understanding of 'digit', see [`is_numeric`][is_numeric].
    ///
    /// [is_numeric]: #method.is_numeric
    ///
    /// # Panics
    ///
    /// Panics if given a radix larger than 36.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!('1'.is_digit(10));
    /// assert!('f'.is_digit(16));
    /// assert!(!'f'.is_digit(10));
    /// ```
    ///
    /// Passing a large radix, causing a panic:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let result = thread::spawn(|| {
    ///     // this panics
    ///     '1'.is_digit(37);
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_digit(self, radix: u32) -> bool {
        self.to_digit(radix).is_some()
    }

    /// Converts a `char` to a digit in the given radix.
    ///
    /// A 'radix' here is sometimes also called a 'base'. A radix of two
    /// indicates a binary number, a radix of ten, decimal, and a radix of
    /// sixteen, hexadecimal, to give some common values. Arbitrary
    /// radices are supported.
    ///
    /// 'Digit' is defined to be only the following characters:
    ///
    /// * `0-9`
    /// * `a-z`
    /// * `A-Z`
    ///
    /// # Errors
    ///
    /// Returns `None` if the `char` does not refer to a digit in the given radix.
    ///
    /// # Panics
    ///
    /// Panics if given a radix larger than 36.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert_eq!('1'.to_digit(10), Some(1));
    /// assert_eq!('f'.to_digit(16), Some(15));
    /// ```
    ///
    /// Passing a non-digit results in failure:
    ///
    /// ```
    /// assert_eq!('f'.to_digit(10), None);
    /// assert_eq!('z'.to_digit(16), None);
    /// ```
    ///
    /// Passing a large radix, causing a panic:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let result = thread::spawn(|| {
    ///     '1'.to_digit(37);
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_digit(self, radix: u32) -> Option<u32> {
        assert!(radix <= 36, "to_digit: radix is too high (maximum 36)");

        // the code is split up here to improve execution speed for cases where
        // the `radix` is constant and 10 or smaller
        let val = if radix <= 10  {
            match self {
                '0' ..= '9' => self as u32 - '0' as u32,
                _ => return None,
            }
        } else {
            match self {
                '0'..='9' => self as u32 - '0' as u32,
                'a'..='z' => self as u32 - 'a' as u32 + 10,
                'A'..='Z' => self as u32 - 'A' as u32 + 10,
                _ => return None,
            }
        };

        if val < radix { Some(val) }
        else { None }
    }

    /// Returns an iterator that yields the hexadecimal Unicode escape of a
    /// character as `char`s.
    ///
    /// This will escape characters with the Rust syntax of the form
    /// `\u{NNNNNN}` where `NNNNNN` is a hexadecimal representation.
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in '❤'.escape_unicode() {
    ///     print!("{}", c);
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", '❤'.escape_unicode());
    /// ```
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("\\u{{2764}}");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!('❤'.escape_unicode().to_string(), "\\u{2764}");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn escape_unicode(self) -> EscapeUnicode {
        let c = self as u32;

        // or-ing 1 ensures that for c==0 the code computes that one
        // digit should be printed and (which is the same) avoids the
        // (31 - 32) underflow
        let msb = 31 - (c | 1).leading_zeros();

        // the index of the most significant hex digit
        let ms_hex_digit = msb / 4;
        EscapeUnicode {
            c: self,
            state: EscapeUnicodeState::Backslash,
            hex_digit_idx: ms_hex_digit as usize,
        }
    }

    /// An extended version of `escape_debug` that optionally permits escaping
    /// Extended Grapheme codepoints. This allows us to format characters like
    /// nonspacing marks better when they're at the start of a string.
    #[inline]
    pub(crate) fn escape_debug_ext(self, escape_grapheme_extended: bool) -> EscapeDebug {
        let init_state = match self {
            '\t' => EscapeDefaultState::Backslash('t'),
            '\r' => EscapeDefaultState::Backslash('r'),
            '\n' => EscapeDefaultState::Backslash('n'),
            '\\' | '\'' | '"' => EscapeDefaultState::Backslash(self),
            _ if escape_grapheme_extended && self.is_grapheme_extended() => {
                EscapeDefaultState::Unicode(self.escape_unicode())
            }
            _ if is_printable(self) => EscapeDefaultState::Char(self),
            _ => EscapeDefaultState::Unicode(self.escape_unicode()),
        };
        EscapeDebug(EscapeDefault { state: init_state })
    }

    /// Returns an iterator that yields the literal escape code of a character
    /// as `char`s.
    ///
    /// This will escape the characters similar to the `Debug` implementations
    /// of `str` or `char`.
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in '\n'.escape_debug() {
    ///     print!("{}", c);
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", '\n'.escape_debug());
    /// ```
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("\\n");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!('\n'.escape_debug().to_string(), "\\n");
    /// ```
    #[stable(feature = "char_escape_debug", since = "1.20.0")]
    #[inline]
    pub fn escape_debug(self) -> EscapeDebug {
        self.escape_debug_ext(true)
    }

    /// Returns an iterator that yields the literal escape code of a character
    /// as `char`s.
    ///
    /// The default is chosen with a bias toward producing literals that are
    /// legal in a variety of languages, including C++11 and similar C-family
    /// languages. The exact rules are:
    ///
    /// * Tab is escaped as `\t`.
    /// * Carriage return is escaped as `\r`.
    /// * Line feed is escaped as `\n`.
    /// * Single quote is escaped as `\'`.
    /// * Double quote is escaped as `\"`.
    /// * Backslash is escaped as `\\`.
    /// * Any character in the 'printable ASCII' range `0x20` .. `0x7e`
    ///   inclusive is not escaped.
    /// * All other characters are given hexadecimal Unicode escapes; see
    ///   [`escape_unicode`][escape_unicode].
    ///
    /// [escape_unicode]: #method.escape_unicode
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in '"'.escape_default() {
    ///     print!("{}", c);
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", '"'.escape_default());
    /// ```
    ///
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("\\\"");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!('"'.escape_default().to_string(), "\\\"");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn escape_default(self) -> EscapeDefault {
        let init_state = match self {
            '\t' => EscapeDefaultState::Backslash('t'),
            '\r' => EscapeDefaultState::Backslash('r'),
            '\n' => EscapeDefaultState::Backslash('n'),
            '\\' | '\'' | '"' => EscapeDefaultState::Backslash(self),
            '\x20' ..= '\x7e' => EscapeDefaultState::Char(self),
            _ => EscapeDefaultState::Unicode(self.escape_unicode())
        };
        EscapeDefault { state: init_state }
    }

    /// Returns the number of bytes this `char` would need if encoded in UTF-8.
    ///
    /// That number of bytes is always between 1 and 4, inclusive.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let len = 'A'.len_utf8();
    /// assert_eq!(len, 1);
    ///
    /// let len = 'ß'.len_utf8();
    /// assert_eq!(len, 2);
    ///
    /// let len = 'ℝ'.len_utf8();
    /// assert_eq!(len, 3);
    ///
    /// let len = '💣'.len_utf8();
    /// assert_eq!(len, 4);
    /// ```
    ///
    /// The `&str` type guarantees that its contents are UTF-8, and so we can compare the length it
    /// would take if each code point was represented as a `char` vs in the `&str` itself:
    ///
    /// ```
    /// // as chars
    /// let eastern = '東';
    /// let capital = '京';
    ///
    /// // both can be represented as three bytes
    /// assert_eq!(3, eastern.len_utf8());
    /// assert_eq!(3, capital.len_utf8());
    ///
    /// // as a &str, these two are encoded in UTF-8
    /// let tokyo = "東京";
    ///
    /// let len = eastern.len_utf8() + capital.len_utf8();
    ///
    /// // we can see that they take six bytes total...
    /// assert_eq!(6, tokyo.len());
    ///
    /// // ... just like the &str
    /// assert_eq!(len, tokyo.len());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn len_utf8(self) -> usize {
        let code = self as u32;
        if code < MAX_ONE_B {
            1
        } else if code < MAX_TWO_B {
            2
        } else if code < MAX_THREE_B {
            3
        } else {
            4
        }
    }

    /// Returns the number of 16-bit code units this `char` would need if
    /// encoded in UTF-16.
    ///
    /// See the documentation for [`len_utf8`] for more explanation of this
    /// concept. This function is a mirror, but for UTF-16 instead of UTF-8.
    ///
    /// [`len_utf8`]: #method.len_utf8
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let n = 'ß'.len_utf16();
    /// assert_eq!(n, 1);
    ///
    /// let len = '💣'.len_utf16();
    /// assert_eq!(len, 2);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn len_utf16(self) -> usize {
        let ch = self as u32;
        if (ch & 0xFFFF) == ch { 1 } else { 2 }
    }

    /// Encodes this character as UTF-8 into the provided byte buffer,
    /// and then returns the subslice of the buffer that contains the encoded character.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not large enough.
    /// A buffer of length four is large enough to encode any `char`.
    ///
    /// # Examples
    ///
    /// In both of these examples, 'ß' takes two bytes to encode.
    ///
    /// ```
    /// let mut b = [0; 2];
    ///
    /// let result = 'ß'.encode_utf8(&mut b);
    ///
    /// assert_eq!(result, "ß");
    ///
    /// assert_eq!(result.len(), 2);
    /// ```
    ///
    /// A buffer that's too small:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let result = thread::spawn(|| {
    ///     let mut b = [0; 1];
    ///
    ///     // this panics
    ///    'ß'.encode_utf8(&mut b);
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "unicode_encode_char", since = "1.15.0")]
    #[inline]
    pub fn encode_utf8(self, dst: &mut [u8]) -> &mut str {
        let code = self as u32;
        unsafe {
            let len =
            if code < MAX_ONE_B && !dst.is_empty() {
                *dst.get_unchecked_mut(0) = code as u8;
                1
            } else if code < MAX_TWO_B && dst.len() >= 2 {
                *dst.get_unchecked_mut(0) = (code >> 6 & 0x1F) as u8 | TAG_TWO_B;
                *dst.get_unchecked_mut(1) = (code & 0x3F) as u8 | TAG_CONT;
                2
            } else if code < MAX_THREE_B && dst.len() >= 3  {
                *dst.get_unchecked_mut(0) = (code >> 12 & 0x0F) as u8 | TAG_THREE_B;
                *dst.get_unchecked_mut(1) = (code >>  6 & 0x3F) as u8 | TAG_CONT;
                *dst.get_unchecked_mut(2) = (code & 0x3F) as u8 | TAG_CONT;
                3
            } else if dst.len() >= 4 {
                *dst.get_unchecked_mut(0) = (code >> 18 & 0x07) as u8 | TAG_FOUR_B;
                *dst.get_unchecked_mut(1) = (code >> 12 & 0x3F) as u8 | TAG_CONT;
                *dst.get_unchecked_mut(2) = (code >>  6 & 0x3F) as u8 | TAG_CONT;
                *dst.get_unchecked_mut(3) = (code & 0x3F) as u8 | TAG_CONT;
                4
            } else {
                panic!("encode_utf8: need {} bytes to encode U+{:X}, but the buffer has {}",
                    from_u32_unchecked(code).len_utf8(),
                    code,
                    dst.len())
            };
            from_utf8_unchecked_mut(dst.get_unchecked_mut(..len))
        }
    }

    /// Encodes this character as UTF-16 into the provided `u16` buffer,
    /// and then returns the subslice of the buffer that contains the encoded character.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not large enough.
    /// A buffer of length 2 is large enough to encode any `char`.
    ///
    /// # Examples
    ///
    /// In both of these examples, '𝕊' takes two `u16`s to encode.
    ///
    /// ```
    /// let mut b = [0; 2];
    ///
    /// let result = '𝕊'.encode_utf16(&mut b);
    ///
    /// assert_eq!(result.len(), 2);
    /// ```
    ///
    /// A buffer that's too small:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let result = thread::spawn(|| {
    ///     let mut b = [0; 1];
    ///
    ///     // this panics
    ///     '𝕊'.encode_utf16(&mut b);
    /// }).join();
    ///
    /// assert!(result.is_err());
    /// ```
    #[stable(feature = "unicode_encode_char", since = "1.15.0")]
    #[inline]
    pub fn encode_utf16(self, dst: &mut [u16]) -> &mut [u16] {
        let mut code = self as u32;
        unsafe {
            if (code & 0xFFFF) == code && !dst.is_empty() {
                // The BMP falls through (assuming non-surrogate, as it should)
                *dst.get_unchecked_mut(0) = code as u16;
                slice::from_raw_parts_mut(dst.as_mut_ptr(), 1)
            } else if dst.len() >= 2 {
                // Supplementary planes break into surrogates.
                code -= 0x1_0000;
                *dst.get_unchecked_mut(0) = 0xD800 | ((code >> 10) as u16);
                *dst.get_unchecked_mut(1) = 0xDC00 | ((code as u16) & 0x3FF);
                slice::from_raw_parts_mut(dst.as_mut_ptr(), 2)
            } else {
                panic!("encode_utf16: need {} units to encode U+{:X}, but the buffer has {}",
                    from_u32_unchecked(code).len_utf16(),
                    code,
                    dst.len())
            }
        }
    }

    /// Returns `true` if this `char` is an alphabetic code point, and false if not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!('a'.is_alphabetic());
    /// assert!('京'.is_alphabetic());
    ///
    /// let c = '💝';
    /// // love is many things, but it is not alphabetic
    /// assert!(!c.is_alphabetic());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_alphabetic(self) -> bool {
        match self {
            'a'..='z' | 'A'..='Z' => true,
            c if c > '\x7f' => derived_property::Alphabetic(c),
            _ => false,
        }
    }

    /// Returns `true` if this `char` satisfies the 'XID_Start' Unicode property, and false
    /// otherwise.
    ///
    /// 'XID_Start' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to `ID_Start` but modified for closure under `NFKx`.
    #[unstable(feature = "rustc_private",
               reason = "mainly needed for compiler internals",
               issue = "27812")]
    #[inline]
    pub fn is_xid_start(self) -> bool {
        derived_property::XID_Start(self)
    }

    /// Returns `true` if this `char` satisfies the 'XID_Continue' Unicode property, and false
    /// otherwise.
    ///
    /// 'XID_Continue' is a Unicode Derived Property specified in
    /// [UAX #31](http://unicode.org/reports/tr31/#NFKC_Modifications),
    /// mostly similar to 'ID_Continue' but modified for closure under NFKx.
    #[unstable(feature = "rustc_private",
               reason = "mainly needed for compiler internals",
               issue = "27812")]
    #[inline]
    pub fn is_xid_continue(self) -> bool {
        derived_property::XID_Continue(self)
    }

    /// Returns `true` if this `char` is lowercase.
    ///
    /// 'Lowercase' is defined according to the terms of the Unicode Derived Core
    /// Property `Lowercase`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!('a'.is_lowercase());
    /// assert!('δ'.is_lowercase());
    /// assert!(!'A'.is_lowercase());
    /// assert!(!'Δ'.is_lowercase());
    ///
    /// // The various Chinese scripts do not have case, and so:
    /// assert!(!'中'.is_lowercase());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_lowercase(self) -> bool {
        match self {
            'a'..='z' => true,
            c if c > '\x7f' => derived_property::Lowercase(c),
            _ => false,
        }
    }

    /// Returns `true` if this `char` is uppercase.
    ///
    /// 'Uppercase' is defined according to the terms of the Unicode Derived Core
    /// Property `Uppercase`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!(!'a'.is_uppercase());
    /// assert!(!'δ'.is_uppercase());
    /// assert!('A'.is_uppercase());
    /// assert!('Δ'.is_uppercase());
    ///
    /// // The various Chinese scripts do not have case, and so:
    /// assert!(!'中'.is_uppercase());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_uppercase(self) -> bool {
        match self {
            'A'..='Z' => true,
            c if c > '\x7f' => derived_property::Uppercase(c),
            _ => false,
        }
    }

    /// Returns `true` if this `char` is whitespace.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived Core
    /// Property `White_Space`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!(' '.is_whitespace());
    ///
    /// // a non-breaking space
    /// assert!('\u{A0}'.is_whitespace());
    ///
    /// assert!(!'越'.is_whitespace());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_whitespace(self) -> bool {
        match self {
            ' ' | '\x09'..='\x0d' => true,
            c if c > '\x7f' => property::White_Space(c),
            _ => false,
        }
    }

    /// Returns `true` if this `char` is alphanumeric.
    ///
    /// 'Alphanumeric'-ness is defined in terms of the Unicode General Categories
    /// 'Nd', 'Nl', 'No' and the Derived Core Property 'Alphabetic'.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!('٣'.is_alphanumeric());
    /// assert!('7'.is_alphanumeric());
    /// assert!('৬'.is_alphanumeric());
    /// assert!('¾'.is_alphanumeric());
    /// assert!('①'.is_alphanumeric());
    /// assert!('K'.is_alphanumeric());
    /// assert!('و'.is_alphanumeric());
    /// assert!('藏'.is_alphanumeric());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_alphanumeric(self) -> bool {
        self.is_alphabetic() || self.is_numeric()
    }

    /// Returns `true` if this `char` is a control code point.
    ///
    /// 'Control code point' is defined in terms of the Unicode General
    /// Category `Cc`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // U+009C, STRING TERMINATOR
    /// assert!(''.is_control());
    /// assert!(!'q'.is_control());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_control(self) -> bool {
        general_category::Cc(self)
    }

    /// Returns `true` if this `char` is an extended grapheme character.
    ///
    /// 'Extended grapheme character' is defined in terms of the Unicode Shaping and Rendering
    /// Category `Grapheme_Extend`.
    #[inline]
    pub(crate) fn is_grapheme_extended(self) -> bool {
        derived_property::Grapheme_Extend(self)
    }

    /// Returns `true` if this `char` is numeric.
    ///
    /// 'Numeric'-ness is defined in terms of the Unicode General Categories
    /// 'Nd', 'Nl', 'No'.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!('٣'.is_numeric());
    /// assert!('7'.is_numeric());
    /// assert!('৬'.is_numeric());
    /// assert!('¾'.is_numeric());
    /// assert!('①'.is_numeric());
    /// assert!(!'K'.is_numeric());
    /// assert!(!'و'.is_numeric());
    /// assert!(!'藏'.is_numeric());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_numeric(self) -> bool {
        match self {
            '0'..='9' => true,
            c if c > '\x7f' => general_category::N(c),
            _ => false,
        }
    }

    /// Returns an iterator that yields the lowercase equivalent of a `char`
    /// as one or more `char`s.
    ///
    /// If a character does not have a lowercase equivalent, the same character
    /// will be returned back by the iterator.
    ///
    /// This performs complex unconditional mappings with no tailoring: it maps
    /// one Unicode character to its lowercase equivalent according to the
    /// [Unicode database] and the additional complex mappings
    /// [`SpecialCasing.txt`]. Conditional mappings (based on context or
    /// language) are not considered here.
    ///
    /// For a full reference, see [here][reference].
    ///
    /// [Unicode database]: ftp://ftp.unicode.org/Public/UNIDATA/UnicodeData.txt
    ///
    /// [`SpecialCasing.txt`]: ftp://ftp.unicode.org/Public/UNIDATA/SpecialCasing.txt
    ///
    /// [reference]: http://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in 'İ'.to_lowercase() {
    ///     print!("{}", c);
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", 'İ'.to_lowercase());
    /// ```
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("i\u{307}");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!('C'.to_lowercase().to_string(), "c");
    ///
    /// // Sometimes the result is more than one character:
    /// assert_eq!('İ'.to_lowercase().to_string(), "i\u{307}");
    ///
    /// // Characters that do not have both uppercase and lowercase
    /// // convert into themselves.
    /// assert_eq!('山'.to_lowercase().to_string(), "山");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_lowercase(self) -> ToLowercase {
        ToLowercase(CaseMappingIter::new(conversions::to_lower(self)))
    }

    /// Returns an iterator that yields the uppercase equivalent of a `char`
    /// as one or more `char`s.
    ///
    /// If a character does not have an uppercase equivalent, the same character
    /// will be returned back by the iterator.
    ///
    /// This performs complex unconditional mappings with no tailoring: it maps
    /// one Unicode character to its uppercase equivalent according to the
    /// [Unicode database] and the additional complex mappings
    /// [`SpecialCasing.txt`]. Conditional mappings (based on context or
    /// language) are not considered here.
    ///
    /// For a full reference, see [here][reference].
    ///
    /// [Unicode database]: ftp://ftp.unicode.org/Public/UNIDATA/UnicodeData.txt
    ///
    /// [`SpecialCasing.txt`]: ftp://ftp.unicode.org/Public/UNIDATA/SpecialCasing.txt
    ///
    /// [reference]: http://www.unicode.org/versions/Unicode7.0.0/ch03.pdf#G33992
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in 'ß'.to_uppercase() {
    ///     print!("{}", c);
    /// }
    /// println!();
    /// ```
    ///
    /// Using `println!` directly:
    ///
    /// ```
    /// println!("{}", 'ß'.to_uppercase());
    /// ```
    ///
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("SS");
    /// ```
    ///
    /// Using `to_string`:
    ///
    /// ```
    /// assert_eq!('c'.to_uppercase().to_string(), "C");
    ///
    /// // Sometimes the result is more than one character:
    /// assert_eq!('ß'.to_uppercase().to_string(), "SS");
    ///
    /// // Characters that do not have both uppercase and lowercase
    /// // convert into themselves.
    /// assert_eq!('山'.to_uppercase().to_string(), "山");
    /// ```
    ///
    /// # Note on locale
    ///
    /// In Turkish, the equivalent of 'i' in Latin has five forms instead of two:
    ///
    /// * 'Dotless': I / ı, sometimes written ï
    /// * 'Dotted': İ / i
    ///
    /// Note that the lowercase dotted 'i' is the same as the Latin. Therefore:
    ///
    /// ```
    /// let upper_i = 'i'.to_uppercase().to_string();
    /// ```
    ///
    /// The value of `upper_i` here relies on the language of the text: if we're
    /// in `en-US`, it should be `"I"`, but if we're in `tr_TR`, it should
    /// be `"İ"`. `to_uppercase()` does not take this into account, and so:
    ///
    /// ```
    /// let upper_i = 'i'.to_uppercase().to_string();
    ///
    /// assert_eq!(upper_i, "I");
    /// ```
    ///
    /// holds across languages.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_uppercase(self) -> ToUppercase {
        ToUppercase(CaseMappingIter::new(conversions::to_upper(self)))
    }

    /// Checks if the value is within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii = 'a';
    /// let non_ascii = '❤';
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub const fn is_ascii(&self) -> bool {
        *self as u32 <= 0x7F
    }

    /// Makes a copy of the value in its ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// To uppercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii = 'a';
    /// let non_ascii = '❤';
    ///
    /// assert_eq!('A', ascii.to_ascii_uppercase());
    /// assert_eq!('❤', non_ascii.to_ascii_uppercase());
    /// ```
    ///
    /// [`make_ascii_uppercase`]: #method.make_ascii_uppercase
    /// [`to_uppercase`]: #method.to_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn to_ascii_uppercase(&self) -> char {
        if self.is_ascii() {
            (*self as u8).to_ascii_uppercase() as char
        } else {
            *self
        }
    }

    /// Makes a copy of the value in its ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// To lowercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii = 'A';
    /// let non_ascii = '❤';
    ///
    /// assert_eq!('a', ascii.to_ascii_lowercase());
    /// assert_eq!('❤', non_ascii.to_ascii_lowercase());
    /// ```
    ///
    /// [`make_ascii_lowercase`]: #method.make_ascii_lowercase
    /// [`to_lowercase`]: #method.to_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn to_ascii_lowercase(&self) -> char {
        if self.is_ascii() {
            (*self as u8).to_ascii_lowercase() as char
        } else {
            *self
        }
    }

    /// Checks that two values are an ASCII case-insensitive match.
    ///
    /// Equivalent to `to_ascii_lowercase(a) == to_ascii_lowercase(b)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let upper_a = 'A';
    /// let lower_a = 'a';
    /// let lower_z = 'z';
    ///
    /// assert!(upper_a.eq_ignore_ascii_case(&lower_a));
    /// assert!(upper_a.eq_ignore_ascii_case(&upper_a));
    /// assert!(!upper_a.eq_ignore_ascii_case(&lower_z));
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &char) -> bool {
        self.to_ascii_lowercase() == other.to_ascii_lowercase()
    }

    /// Converts this type to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut ascii = 'a';
    ///
    /// ascii.make_ascii_uppercase();
    ///
    /// assert_eq!('A', ascii);
    /// ```
    ///
    /// [`to_ascii_uppercase`]: #method.to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        *self = self.to_ascii_uppercase();
    }

    /// Converts this type to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut ascii = 'A';
    ///
    /// ascii.make_ascii_lowercase();
    ///
    /// assert_eq!('a', ascii);
    /// ```
    ///
    /// [`to_ascii_lowercase`]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        *self = self.to_ascii_lowercase();
    }

    /// Checks if the value is an ASCII alphabetic character:
    ///
    /// - U+0041 'A' ..= U+005A 'Z', or
    /// - U+0061 'a' ..= U+007A 'z'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(uppercase_a.is_ascii_alphabetic());
    /// assert!(uppercase_g.is_ascii_alphabetic());
    /// assert!(a.is_ascii_alphabetic());
    /// assert!(g.is_ascii_alphabetic());
    /// assert!(!zero.is_ascii_alphabetic());
    /// assert!(!percent.is_ascii_alphabetic());
    /// assert!(!space.is_ascii_alphabetic());
    /// assert!(!lf.is_ascii_alphabetic());
    /// assert!(!esc.is_ascii_alphabetic());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_alphabetic(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_alphabetic()
    }

    /// Checks if the value is an ASCII uppercase character:
    /// U+0041 'A' ..= U+005A 'Z'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(uppercase_a.is_ascii_uppercase());
    /// assert!(uppercase_g.is_ascii_uppercase());
    /// assert!(!a.is_ascii_uppercase());
    /// assert!(!g.is_ascii_uppercase());
    /// assert!(!zero.is_ascii_uppercase());
    /// assert!(!percent.is_ascii_uppercase());
    /// assert!(!space.is_ascii_uppercase());
    /// assert!(!lf.is_ascii_uppercase());
    /// assert!(!esc.is_ascii_uppercase());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_uppercase(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_uppercase()
    }

    /// Checks if the value is an ASCII lowercase character:
    /// U+0061 'a' ..= U+007A 'z'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(!uppercase_a.is_ascii_lowercase());
    /// assert!(!uppercase_g.is_ascii_lowercase());
    /// assert!(a.is_ascii_lowercase());
    /// assert!(g.is_ascii_lowercase());
    /// assert!(!zero.is_ascii_lowercase());
    /// assert!(!percent.is_ascii_lowercase());
    /// assert!(!space.is_ascii_lowercase());
    /// assert!(!lf.is_ascii_lowercase());
    /// assert!(!esc.is_ascii_lowercase());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_lowercase(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_lowercase()
    }

    /// Checks if the value is an ASCII alphanumeric character:
    ///
    /// - U+0041 'A' ..= U+005A 'Z', or
    /// - U+0061 'a' ..= U+007A 'z', or
    /// - U+0030 '0' ..= U+0039 '9'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(uppercase_a.is_ascii_alphanumeric());
    /// assert!(uppercase_g.is_ascii_alphanumeric());
    /// assert!(a.is_ascii_alphanumeric());
    /// assert!(g.is_ascii_alphanumeric());
    /// assert!(zero.is_ascii_alphanumeric());
    /// assert!(!percent.is_ascii_alphanumeric());
    /// assert!(!space.is_ascii_alphanumeric());
    /// assert!(!lf.is_ascii_alphanumeric());
    /// assert!(!esc.is_ascii_alphanumeric());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_alphanumeric(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_alphanumeric()
    }

    /// Checks if the value is an ASCII decimal digit:
    /// U+0030 '0' ..= U+0039 '9'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(!uppercase_a.is_ascii_digit());
    /// assert!(!uppercase_g.is_ascii_digit());
    /// assert!(!a.is_ascii_digit());
    /// assert!(!g.is_ascii_digit());
    /// assert!(zero.is_ascii_digit());
    /// assert!(!percent.is_ascii_digit());
    /// assert!(!space.is_ascii_digit());
    /// assert!(!lf.is_ascii_digit());
    /// assert!(!esc.is_ascii_digit());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_digit(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_digit()
    }

    /// Checks if the value is an ASCII hexadecimal digit:
    ///
    /// - U+0030 '0' ..= U+0039 '9', or
    /// - U+0041 'A' ..= U+0046 'F', or
    /// - U+0061 'a' ..= U+0066 'f'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(uppercase_a.is_ascii_hexdigit());
    /// assert!(!uppercase_g.is_ascii_hexdigit());
    /// assert!(a.is_ascii_hexdigit());
    /// assert!(!g.is_ascii_hexdigit());
    /// assert!(zero.is_ascii_hexdigit());
    /// assert!(!percent.is_ascii_hexdigit());
    /// assert!(!space.is_ascii_hexdigit());
    /// assert!(!lf.is_ascii_hexdigit());
    /// assert!(!esc.is_ascii_hexdigit());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_hexdigit(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_hexdigit()
    }

    /// Checks if the value is an ASCII punctuation character:
    ///
    /// - U+0021 ..= U+002F `! " # $ % & ' ( ) * + , - . /`, or
    /// - U+003A ..= U+0040 `: ; < = > ? @`, or
    /// - U+005B ..= U+0060 ``[ \ ] ^ _ ` ``, or
    /// - U+007B ..= U+007E `{ | } ~`
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(!uppercase_a.is_ascii_punctuation());
    /// assert!(!uppercase_g.is_ascii_punctuation());
    /// assert!(!a.is_ascii_punctuation());
    /// assert!(!g.is_ascii_punctuation());
    /// assert!(!zero.is_ascii_punctuation());
    /// assert!(percent.is_ascii_punctuation());
    /// assert!(!space.is_ascii_punctuation());
    /// assert!(!lf.is_ascii_punctuation());
    /// assert!(!esc.is_ascii_punctuation());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_punctuation(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_punctuation()
    }

    /// Checks if the value is an ASCII graphic character:
    /// U+0021 '!' ..= U+007E '~'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(uppercase_a.is_ascii_graphic());
    /// assert!(uppercase_g.is_ascii_graphic());
    /// assert!(a.is_ascii_graphic());
    /// assert!(g.is_ascii_graphic());
    /// assert!(zero.is_ascii_graphic());
    /// assert!(percent.is_ascii_graphic());
    /// assert!(!space.is_ascii_graphic());
    /// assert!(!lf.is_ascii_graphic());
    /// assert!(!esc.is_ascii_graphic());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_graphic(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_graphic()
    }

    /// Checks if the value is an ASCII whitespace character:
    /// U+0020 SPACE, U+0009 HORIZONTAL TAB, U+000A LINE FEED,
    /// U+000C FORM FEED, or U+000D CARRIAGE RETURN.
    ///
    /// Rust uses the WhatWG Infra Standard's [definition of ASCII
    /// whitespace][infra-aw]. There are several other definitions in
    /// wide use. For instance, [the POSIX locale][pct] includes
    /// U+000B VERTICAL TAB as well as all the above characters,
    /// but—from the very same specification—[the default rule for
    /// "field splitting" in the Bourne shell][bfs] considers *only*
    /// SPACE, HORIZONTAL TAB, and LINE FEED as whitespace.
    ///
    /// If you are writing a program that will process an existing
    /// file format, check what that format's definition of whitespace is
    /// before using this function.
    ///
    /// [infra-aw]: https://infra.spec.whatwg.org/#ascii-whitespace
    /// [pct]: http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap07.html#tag_07_03_01
    /// [bfs]: http://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_05
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(!uppercase_a.is_ascii_whitespace());
    /// assert!(!uppercase_g.is_ascii_whitespace());
    /// assert!(!a.is_ascii_whitespace());
    /// assert!(!g.is_ascii_whitespace());
    /// assert!(!zero.is_ascii_whitespace());
    /// assert!(!percent.is_ascii_whitespace());
    /// assert!(space.is_ascii_whitespace());
    /// assert!(lf.is_ascii_whitespace());
    /// assert!(!esc.is_ascii_whitespace());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_whitespace(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_whitespace()
    }

    /// Checks if the value is an ASCII control character:
    /// U+0000 NUL ..= U+001F UNIT SEPARATOR, or U+007F DELETE.
    /// Note that most ASCII whitespace characters are control
    /// characters, but SPACE is not.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 'A';
    /// let uppercase_g = 'G';
    /// let a = 'a';
    /// let g = 'g';
    /// let zero = '0';
    /// let percent = '%';
    /// let space = ' ';
    /// let lf = '\n';
    /// let esc: char = 0x1b_u8.into();
    ///
    /// assert!(!uppercase_a.is_ascii_control());
    /// assert!(!uppercase_g.is_ascii_control());
    /// assert!(!a.is_ascii_control());
    /// assert!(!g.is_ascii_control());
    /// assert!(!zero.is_ascii_control());
    /// assert!(!percent.is_ascii_control());
    /// assert!(!space.is_ascii_control());
    /// assert!(lf.is_ascii_control());
    /// assert!(esc.is_ascii_control());
    /// ```
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[inline]
    pub fn is_ascii_control(&self) -> bool {
        self.is_ascii() && (*self as u8).is_ascii_control()
    }
}
