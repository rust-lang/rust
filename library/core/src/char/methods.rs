//! impl char {}

use super::*;
use crate::panic::const_panic;
use crate::slice;
use crate::str::from_utf8_unchecked_mut;
use crate::unicode::printable::is_printable;
use crate::unicode::{self, conversions};

impl char {
    /// The lowest valid code point a `char` can have, `'\0'`.
    ///
    /// Unlike integer types, `char` actually has a gap in the middle,
    /// meaning that the range of possible `char`s is smaller than you
    /// might expect. Ranges of `char` will automatically hop this gap
    /// for you:
    ///
    /// ```
    /// let dist = u32::from(char::MAX) - u32::from(char::MIN);
    /// let size = (char::MIN..=char::MAX).count() as u32;
    /// assert!(size < dist);
    /// ```
    ///
    /// Despite this gap, the `MIN` and [`MAX`] values can be used as bounds for
    /// all `char` values.
    ///
    /// [`MAX`]: char::MAX
    ///
    /// # Examples
    ///
    /// ```
    /// # fn something_which_returns_char() -> char { 'a' }
    /// let c: char = something_which_returns_char();
    /// assert!(char::MIN <= c);
    ///
    /// let value_at_min = u32::from(char::MIN);
    /// assert_eq!(char::from_u32(value_at_min), Some('\0'));
    /// ```
    #[stable(feature = "char_min", since = "1.83.0")]
    pub const MIN: char = '\0';

    /// The highest valid code point a `char` can have, `'\u{10FFFF}'`.
    ///
    /// Unlike integer types, `char` actually has a gap in the middle,
    /// meaning that the range of possible `char`s is smaller than you
    /// might expect. Ranges of `char` will automatically hop this gap
    /// for you:
    ///
    /// ```
    /// let dist = u32::from(char::MAX) - u32::from(char::MIN);
    /// let size = (char::MIN..=char::MAX).count() as u32;
    /// assert!(size < dist);
    /// ```
    ///
    /// Despite this gap, the [`MIN`] and `MAX` values can be used as bounds for
    /// all `char` values.
    ///
    /// [`MIN`]: char::MIN
    ///
    /// # Examples
    ///
    /// ```
    /// # fn something_which_returns_char() -> char { 'a' }
    /// let c: char = something_which_returns_char();
    /// assert!(c <= char::MAX);
    ///
    /// let value_at_max = u32::from(char::MAX);
    /// assert_eq!(char::from_u32(value_at_max), Some('\u{10FFFF}'));
    /// assert_eq!(char::from_u32(value_at_max + 1), None);
    /// ```
    #[stable(feature = "assoc_char_consts", since = "1.52.0")]
    pub const MAX: char = '\u{10FFFF}';

    /// `U+FFFD REPLACEMENT CHARACTER` (�) is used in Unicode to represent a
    /// decoding error.
    ///
    /// It can occur, for example, when giving ill-formed UTF-8 bytes to
    /// [`String::from_utf8_lossy`](../std/string/struct.String.html#method.from_utf8_lossy).
    #[stable(feature = "assoc_char_consts", since = "1.52.0")]
    pub const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

    /// The version of [Unicode](https://www.unicode.org/) that the Unicode parts of
    /// `char` and `str` methods are based on.
    ///
    /// New versions of Unicode are released regularly and subsequently all methods
    /// in the standard library depending on Unicode are updated. Therefore the
    /// behavior of some `char` and `str` methods and the value of this constant
    /// changes over time. This is *not* considered to be a breaking change.
    ///
    /// The version numbering scheme is explained in
    /// [Unicode 11.0 or later, Section 3.1 Versions of the Unicode Standard](https://www.unicode.org/versions/Unicode11.0.0/ch03.pdf#page=4).
    #[stable(feature = "assoc_char_consts", since = "1.52.0")]
    pub const UNICODE_VERSION: (u8, u8, u8) = crate::unicode::UNICODE_VERSION;

    /// Creates an iterator over the UTF-16 encoded code points in `iter`,
    /// returning unpaired surrogates as `Err`s.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // 𝄞mus<invalid>ic<invalid>
    /// let v = [
    ///     0xD834, 0xDD1E, 0x006d, 0x0075, 0x0073, 0xDD1E, 0x0069, 0x0063, 0xD834,
    /// ];
    ///
    /// assert_eq!(
    ///     char::decode_utf16(v)
    ///         .map(|r| r.map_err(|e| e.unpaired_surrogate()))
    ///         .collect::<Vec<_>>(),
    ///     vec![
    ///         Ok('𝄞'),
    ///         Ok('m'), Ok('u'), Ok('s'),
    ///         Err(0xDD1E),
    ///         Ok('i'), Ok('c'),
    ///         Err(0xD834)
    ///     ]
    /// );
    /// ```
    ///
    /// A lossy decoder can be obtained by replacing `Err` results with the replacement character:
    ///
    /// ```
    /// // 𝄞mus<invalid>ic<invalid>
    /// let v = [
    ///     0xD834, 0xDD1E, 0x006d, 0x0075, 0x0073, 0xDD1E, 0x0069, 0x0063, 0xD834,
    /// ];
    ///
    /// assert_eq!(
    ///     char::decode_utf16(v)
    ///        .map(|r| r.unwrap_or(char::REPLACEMENT_CHARACTER))
    ///        .collect::<String>(),
    ///     "𝄞mus�ic�"
    /// );
    /// ```
    #[stable(feature = "assoc_char_funcs", since = "1.52.0")]
    #[inline]
    pub fn decode_utf16<I: IntoIterator<Item = u16>>(iter: I) -> DecodeUtf16<I::IntoIter> {
        super::decode::decode_utf16(iter)
    }

    /// Converts a `u32` to a `char`.
    ///
    /// Note that all `char`s are valid [`u32`]s, and can be cast to one with
    /// [`as`](../std/keyword.as.html):
    ///
    /// ```
    /// let c = '💯';
    /// let i = c as u32;
    ///
    /// assert_eq!(128175, i);
    /// ```
    ///
    /// However, the reverse is not true: not all valid [`u32`]s are valid
    /// `char`s. `from_u32()` will return `None` if the input is not a valid value
    /// for a `char`.
    ///
    /// For an unsafe version of this function which ignores these checks, see
    /// [`from_u32_unchecked`].
    ///
    /// [`from_u32_unchecked`]: #method.from_u32_unchecked
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = char::from_u32(0x2764);
    ///
    /// assert_eq!(Some('❤'), c);
    /// ```
    ///
    /// Returning `None` when the input is not a valid `char`:
    ///
    /// ```
    /// let c = char::from_u32(0x110000);
    ///
    /// assert_eq!(None, c);
    /// ```
    #[stable(feature = "assoc_char_funcs", since = "1.52.0")]
    #[rustc_const_stable(feature = "const_char_convert", since = "1.67.0")]
    #[must_use]
    #[inline]
    pub const fn from_u32(i: u32) -> Option<char> {
        super::convert::from_u32(i)
    }

    /// Converts a `u32` to a `char`, ignoring validity.
    ///
    /// Note that all `char`s are valid [`u32`]s, and can be cast to one with
    /// `as`:
    ///
    /// ```
    /// let c = '💯';
    /// let i = c as u32;
    ///
    /// assert_eq!(128175, i);
    /// ```
    ///
    /// However, the reverse is not true: not all valid [`u32`]s are valid
    /// `char`s. `from_u32_unchecked()` will ignore this, and blindly cast to
    /// `char`, possibly creating an invalid one.
    ///
    /// # Safety
    ///
    /// This function is unsafe, as it may construct invalid `char` values.
    ///
    /// For a safe version of this function, see the [`from_u32`] function.
    ///
    /// [`from_u32`]: #method.from_u32
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let c = unsafe { char::from_u32_unchecked(0x2764) };
    ///
    /// assert_eq!('❤', c);
    /// ```
    #[stable(feature = "assoc_char_funcs", since = "1.52.0")]
    #[rustc_const_stable(feature = "const_char_from_u32_unchecked", since = "1.81.0")]
    #[must_use]
    #[inline]
    pub const unsafe fn from_u32_unchecked(i: u32) -> char {
        // SAFETY: the safety contract must be upheld by the caller.
        unsafe { super::convert::from_u32_unchecked(i) }
    }

    /// Converts a digit in the given radix to a `char`.
    ///
    /// A 'radix' here is sometimes also called a 'base'. A radix of two
    /// indicates a binary number, a radix of ten, decimal, and a radix of
    /// sixteen, hexadecimal, to give some common values. Arbitrary
    /// radices are supported.
    ///
    /// `from_digit()` will return `None` if the input is not a digit in
    /// the given radix.
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
    /// let c = char::from_digit(4, 10);
    ///
    /// assert_eq!(Some('4'), c);
    ///
    /// // Decimal 11 is a single digit in base 16
    /// let c = char::from_digit(11, 16);
    ///
    /// assert_eq!(Some('b'), c);
    /// ```
    ///
    /// Returning `None` when the input is not a digit:
    ///
    /// ```
    /// let c = char::from_digit(20, 10);
    ///
    /// assert_eq!(None, c);
    /// ```
    ///
    /// Passing a large radix, causing a panic:
    ///
    /// ```should_panic
    /// // this panics
    /// let _c = char::from_digit(1, 37);
    /// ```
    #[stable(feature = "assoc_char_funcs", since = "1.52.0")]
    #[rustc_const_stable(feature = "const_char_convert", since = "1.67.0")]
    #[must_use]
    #[inline]
    pub const fn from_digit(num: u32, radix: u32) -> Option<char> {
        super::convert::from_digit(num, radix)
    }

    /// Checks if a `char` is a digit in the given radix.
    ///
    /// A 'radix' here is sometimes also called a 'base'. A radix of two
    /// indicates a binary number, a radix of ten, decimal, and a radix of
    /// sixteen, hexadecimal, to give some common values. Arbitrary
    /// radices are supported.
    ///
    /// Compared to [`is_numeric()`], this function only recognizes the characters
    /// `0-9`, `a-z` and `A-Z`.
    ///
    /// 'Digit' is defined to be only the following characters:
    ///
    /// * `0-9`
    /// * `a-z`
    /// * `A-Z`
    ///
    /// For a more comprehensive understanding of 'digit', see [`is_numeric()`].
    ///
    /// [`is_numeric()`]: #method.is_numeric
    ///
    /// # Panics
    ///
    /// Panics if given a radix smaller than 2 or larger than 36.
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
    /// ```should_panic
    /// // this panics
    /// '1'.is_digit(37);
    /// ```
    ///
    /// Passing a small radix, causing a panic:
    ///
    /// ```should_panic
    /// // this panics
    /// '1'.is_digit(1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_char_classify", issue = "132241")]
    #[inline]
    pub const fn is_digit(self, radix: u32) -> bool {
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
    /// Panics if given a radix smaller than 2 or larger than 36.
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
    /// ```should_panic
    /// // this panics
    /// let _ = '1'.to_digit(37);
    /// ```
    /// Passing a small radix, causing a panic:
    ///
    /// ```should_panic
    /// // this panics
    /// let _ = '1'.to_digit(1);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_char_convert", since = "1.67.0")]
    #[must_use = "this returns the result of the operation, \
                  without modifying the original"]
    #[inline]
    pub const fn to_digit(self, radix: u32) -> Option<u32> {
        assert!(
            radix >= 2 && radix <= 36,
            "to_digit: invalid radix -- radix must be in the range 2 to 36 inclusive"
        );
        // check radix to remove letter handling code when radix is a known constant
        let value = if self > '9' && radix > 10 {
            // convert ASCII letters to lowercase
            let lower = self as u32 | 0x20;
            // convert an ASCII letter to the corresponding value,
            // non-letters convert to values > 36
            lower.wrapping_sub('a' as u32) as u64 + 10
        } else {
            // convert digit to value, non-digits wrap to values > 36
            (self as u32).wrapping_sub('0' as u32) as u64
        };
        // FIXME(const-hack): once then_some is const fn, use it here
        if value < radix as u64 { Some(value as u32) } else { None }
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
    ///     print!("{c}");
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
    /// Using [`to_string`](../std/string/trait.ToString.html#tymethod.to_string):
    ///
    /// ```
    /// assert_eq!('❤'.escape_unicode().to_string(), "\\u{2764}");
    /// ```
    #[must_use = "this returns the escaped char as an iterator, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn escape_unicode(self) -> EscapeUnicode {
        EscapeUnicode::new(self)
    }

    /// An extended version of `escape_debug` that optionally permits escaping
    /// Extended Grapheme codepoints, single quotes, and double quotes. This
    /// allows us to format characters like nonspacing marks better when they're
    /// at the start of a string, and allows escaping single quotes in
    /// characters, and double quotes in strings.
    #[inline]
    pub(crate) fn escape_debug_ext(self, args: EscapeDebugExtArgs) -> EscapeDebug {
        match self {
            '\0' => EscapeDebug::backslash(ascii::Char::Digit0),
            '\t' => EscapeDebug::backslash(ascii::Char::SmallT),
            '\r' => EscapeDebug::backslash(ascii::Char::SmallR),
            '\n' => EscapeDebug::backslash(ascii::Char::SmallN),
            '\\' => EscapeDebug::backslash(ascii::Char::ReverseSolidus),
            '\"' if args.escape_double_quote => EscapeDebug::backslash(ascii::Char::QuotationMark),
            '\'' if args.escape_single_quote => EscapeDebug::backslash(ascii::Char::Apostrophe),
            _ if args.escape_grapheme_extended && self.is_grapheme_extended() => {
                EscapeDebug::unicode(self)
            }
            _ if is_printable(self) => EscapeDebug::printable(self),
            _ => EscapeDebug::unicode(self),
        }
    }

    /// Returns an iterator that yields the literal escape code of a character
    /// as `char`s.
    ///
    /// This will escape the characters similar to the [`Debug`](core::fmt::Debug) implementations
    /// of `str` or `char`.
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in '\n'.escape_debug() {
    ///     print!("{c}");
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
    /// Using [`to_string`](../std/string/trait.ToString.html#tymethod.to_string):
    ///
    /// ```
    /// assert_eq!('\n'.escape_debug().to_string(), "\\n");
    /// ```
    #[must_use = "this returns the escaped char as an iterator, \
                  without modifying the original"]
    #[stable(feature = "char_escape_debug", since = "1.20.0")]
    #[inline]
    pub fn escape_debug(self) -> EscapeDebug {
        self.escape_debug_ext(EscapeDebugExtArgs::ESCAPE_ALL)
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
    ///   [`escape_unicode`].
    ///
    /// [`escape_unicode`]: #method.escape_unicode
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in '"'.escape_default() {
    ///     print!("{c}");
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
    /// Both are equivalent to:
    ///
    /// ```
    /// println!("\\\"");
    /// ```
    ///
    /// Using [`to_string`](../std/string/trait.ToString.html#tymethod.to_string):
    ///
    /// ```
    /// assert_eq!('"'.escape_default().to_string(), "\\\"");
    /// ```
    #[must_use = "this returns the escaped char as an iterator, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn escape_default(self) -> EscapeDefault {
        match self {
            '\t' => EscapeDefault::backslash(ascii::Char::SmallT),
            '\r' => EscapeDefault::backslash(ascii::Char::SmallR),
            '\n' => EscapeDefault::backslash(ascii::Char::SmallN),
            '\\' | '\'' | '\"' => EscapeDefault::backslash(self.as_ascii().unwrap()),
            '\x20'..='\x7e' => EscapeDefault::printable(self.as_ascii().unwrap()),
            _ => EscapeDefault::unicode(self),
        }
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
    #[rustc_const_stable(feature = "const_char_len_utf", since = "1.52.0")]
    #[inline]
    #[must_use]
    pub const fn len_utf8(self) -> usize {
        len_utf8(self as u32)
    }

    /// Returns the number of 16-bit code units this `char` would need if
    /// encoded in UTF-16.
    ///
    /// That number of code units is always either 1 or 2, for unicode scalar values in
    /// the [basic multilingual plane] or [supplementary planes] respectively.
    ///
    /// See the documentation for [`len_utf8()`] for more explanation of this
    /// concept. This function is a mirror, but for UTF-16 instead of UTF-8.
    ///
    /// [basic multilingual plane]: http://www.unicode.org/glossary/#basic_multilingual_plane
    /// [supplementary planes]: http://www.unicode.org/glossary/#supplementary_planes
    /// [`len_utf8()`]: #method.len_utf8
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
    #[rustc_const_stable(feature = "const_char_len_utf", since = "1.52.0")]
    #[inline]
    #[must_use]
    pub const fn len_utf16(self) -> usize {
        len_utf16(self as u32)
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
    /// ```should_panic
    /// let mut b = [0; 1];
    ///
    /// // this panics
    /// 'ß'.encode_utf8(&mut b);
    /// ```
    #[stable(feature = "unicode_encode_char", since = "1.15.0")]
    #[rustc_const_stable(feature = "const_char_encode_utf8", since = "1.83.0")]
    #[inline]
    pub const fn encode_utf8(self, dst: &mut [u8]) -> &mut str {
        // SAFETY: `char` is not a surrogate, so this is valid UTF-8.
        unsafe { from_utf8_unchecked_mut(encode_utf8_raw(self as u32, dst)) }
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
    /// ```should_panic
    /// let mut b = [0; 1];
    ///
    /// // this panics
    /// '𝕊'.encode_utf16(&mut b);
    /// ```
    #[stable(feature = "unicode_encode_char", since = "1.15.0")]
    #[rustc_const_stable(feature = "const_char_encode_utf16", since = "1.84.0")]
    #[inline]
    pub const fn encode_utf16(self, dst: &mut [u16]) -> &mut [u16] {
        encode_utf16_raw(self as u32, dst)
    }

    /// Returns `true` if this `char` has the `Alphabetic` property.
    ///
    /// `Alphabetic` is described in Chapter 4 (Character Properties) of the [Unicode Standard] and
    /// specified in the [Unicode Character Database][ucd] [`DerivedCoreProperties.txt`].
    ///
    /// [Unicode Standard]: https://www.unicode.org/versions/latest/
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`DerivedCoreProperties.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_alphabetic(self) -> bool {
        match self {
            'a'..='z' | 'A'..='Z' => true,
            c => c > '\x7f' && unicode::Alphabetic(c),
        }
    }

    /// Returns `true` if this `char` has the `Lowercase` property.
    ///
    /// `Lowercase` is described in Chapter 4 (Character Properties) of the [Unicode Standard] and
    /// specified in the [Unicode Character Database][ucd] [`DerivedCoreProperties.txt`].
    ///
    /// [Unicode Standard]: https://www.unicode.org/versions/latest/
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`DerivedCoreProperties.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
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
    /// // The various Chinese scripts and punctuation do not have case, and so:
    /// assert!(!'中'.is_lowercase());
    /// assert!(!' '.is_lowercase());
    /// ```
    ///
    /// In a const context:
    ///
    /// ```
    /// const CAPITAL_DELTA_IS_LOWERCASE: bool = 'Δ'.is_lowercase();
    /// assert!(!CAPITAL_DELTA_IS_LOWERCASE);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_unicode_case_lookup", since = "1.84.0")]
    #[inline]
    pub const fn is_lowercase(self) -> bool {
        match self {
            'a'..='z' => true,
            c => c > '\x7f' && unicode::Lowercase(c),
        }
    }

    /// Returns `true` if this `char` has the `Uppercase` property.
    ///
    /// `Uppercase` is described in Chapter 4 (Character Properties) of the [Unicode Standard] and
    /// specified in the [Unicode Character Database][ucd] [`DerivedCoreProperties.txt`].
    ///
    /// [Unicode Standard]: https://www.unicode.org/versions/latest/
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`DerivedCoreProperties.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
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
    /// // The various Chinese scripts and punctuation do not have case, and so:
    /// assert!(!'中'.is_uppercase());
    /// assert!(!' '.is_uppercase());
    /// ```
    ///
    /// In a const context:
    ///
    /// ```
    /// const CAPITAL_DELTA_IS_UPPERCASE: bool = 'Δ'.is_uppercase();
    /// assert!(CAPITAL_DELTA_IS_UPPERCASE);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_unicode_case_lookup", since = "1.84.0")]
    #[inline]
    pub const fn is_uppercase(self) -> bool {
        match self {
            'A'..='Z' => true,
            c => c > '\x7f' && unicode::Uppercase(c),
        }
    }

    /// Returns `true` if this `char` has the `White_Space` property.
    ///
    /// `White_Space` is specified in the [Unicode Character Database][ucd] [`PropList.txt`].
    ///
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`PropList.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert!(' '.is_whitespace());
    ///
    /// // line break
    /// assert!('\n'.is_whitespace());
    ///
    /// // a non-breaking space
    /// assert!('\u{A0}'.is_whitespace());
    ///
    /// assert!(!'越'.is_whitespace());
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_char_classify", issue = "132241")]
    #[inline]
    pub const fn is_whitespace(self) -> bool {
        match self {
            ' ' | '\x09'..='\x0d' => true,
            c => c > '\x7f' && unicode::White_Space(c),
        }
    }

    /// Returns `true` if this `char` satisfies either [`is_alphabetic()`] or [`is_numeric()`].
    ///
    /// [`is_alphabetic()`]: #method.is_alphabetic
    /// [`is_numeric()`]: #method.is_numeric
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_alphanumeric(self) -> bool {
        self.is_alphabetic() || self.is_numeric()
    }

    /// Returns `true` if this `char` has the general category for control codes.
    ///
    /// Control codes (code points with the general category of `Cc`) are described in Chapter 4
    /// (Character Properties) of the [Unicode Standard] and specified in the [Unicode Character
    /// Database][ucd] [`UnicodeData.txt`].
    ///
    /// [Unicode Standard]: https://www.unicode.org/versions/latest/
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`UnicodeData.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
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
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_control(self) -> bool {
        unicode::Cc(self)
    }

    /// Returns `true` if this `char` has the `Grapheme_Extend` property.
    ///
    /// `Grapheme_Extend` is described in [Unicode Standard Annex #29 (Unicode Text
    /// Segmentation)][uax29] and specified in the [Unicode Character Database][ucd]
    /// [`DerivedCoreProperties.txt`].
    ///
    /// [uax29]: https://www.unicode.org/reports/tr29/
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`DerivedCoreProperties.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/DerivedCoreProperties.txt
    #[must_use]
    #[inline]
    pub(crate) fn is_grapheme_extended(self) -> bool {
        unicode::Grapheme_Extend(self)
    }

    /// Returns `true` if this `char` has one of the general categories for numbers.
    ///
    /// The general categories for numbers (`Nd` for decimal digits, `Nl` for letter-like numeric
    /// characters, and `No` for other numeric characters) are specified in the [Unicode Character
    /// Database][ucd] [`UnicodeData.txt`].
    ///
    /// This method doesn't cover everything that could be considered a number, e.g. ideographic numbers like '三'.
    /// If you want everything including characters with overlapping purposes then you might want to use
    /// a unicode or language-processing library that exposes the appropriate character properties instead
    /// of looking at the unicode categories.
    ///
    /// If you want to parse ASCII decimal digits (0-9) or ASCII base-N, use
    /// `is_ascii_digit` or `is_digit` instead.
    ///
    /// [Unicode Standard]: https://www.unicode.org/versions/latest/
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`UnicodeData.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
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
    /// assert!(!'三'.is_numeric());
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is_numeric(self) -> bool {
        match self {
            '0'..='9' => true,
            c => c > '\x7f' && unicode::N(c),
        }
    }

    /// Returns an iterator that yields the lowercase mapping of this `char` as one or more
    /// `char`s.
    ///
    /// If this `char` does not have a lowercase mapping, the iterator yields the same `char`.
    ///
    /// If this `char` has a one-to-one lowercase mapping given by the [Unicode Character
    /// Database][ucd] [`UnicodeData.txt`], the iterator yields that `char`.
    ///
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`UnicodeData.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
    ///
    /// If this `char` requires special considerations (e.g. multiple `char`s) the iterator yields
    /// the `char`(s) given by [`SpecialCasing.txt`].
    ///
    /// [`SpecialCasing.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/SpecialCasing.txt
    ///
    /// This operation performs an unconditional mapping without tailoring. That is, the conversion
    /// is independent of context and language.
    ///
    /// In the [Unicode Standard], Chapter 4 (Character Properties) discusses case mapping in
    /// general and Chapter 3 (Conformance) discusses the default algorithm for case conversion.
    ///
    /// [Unicode Standard]: https://www.unicode.org/versions/latest/
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in 'İ'.to_lowercase() {
    ///     print!("{c}");
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
    /// Using [`to_string`](../std/string/trait.ToString.html#tymethod.to_string):
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
    #[must_use = "this returns the lowercase character as a new iterator, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn to_lowercase(self) -> ToLowercase {
        ToLowercase(CaseMappingIter::new(conversions::to_lower(self)))
    }

    /// Returns an iterator that yields the uppercase mapping of this `char` as one or more
    /// `char`s.
    ///
    /// If this `char` does not have an uppercase mapping, the iterator yields the same `char`.
    ///
    /// If this `char` has a one-to-one uppercase mapping given by the [Unicode Character
    /// Database][ucd] [`UnicodeData.txt`], the iterator yields that `char`.
    ///
    /// [ucd]: https://www.unicode.org/reports/tr44/
    /// [`UnicodeData.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
    ///
    /// If this `char` requires special considerations (e.g. multiple `char`s) the iterator yields
    /// the `char`(s) given by [`SpecialCasing.txt`].
    ///
    /// [`SpecialCasing.txt`]: https://www.unicode.org/Public/UCD/latest/ucd/SpecialCasing.txt
    ///
    /// This operation performs an unconditional mapping without tailoring. That is, the conversion
    /// is independent of context and language.
    ///
    /// In the [Unicode Standard], Chapter 4 (Character Properties) discusses case mapping in
    /// general and Chapter 3 (Conformance) discusses the default algorithm for case conversion.
    ///
    /// [Unicode Standard]: https://www.unicode.org/versions/latest/
    ///
    /// # Examples
    ///
    /// As an iterator:
    ///
    /// ```
    /// for c in 'ß'.to_uppercase() {
    ///     print!("{c}");
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
    /// Using [`to_string`](../std/string/trait.ToString.html#tymethod.to_string):
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
    #[must_use = "this returns the uppercase character as a new iterator, \
                  without modifying the original"]
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
    #[must_use]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_char_is_ascii", since = "1.32.0")]
    #[inline]
    pub const fn is_ascii(&self) -> bool {
        *self as u32 <= 0x7F
    }

    /// Returns `Some` if the value is within the ASCII range,
    /// or `None` if it's not.
    ///
    /// This is preferred to [`Self::is_ascii`] when you're passing the value
    /// along to something else that can take [`ascii::Char`] rather than
    /// needing to check again for itself whether the value is in ASCII.
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn as_ascii(&self) -> Option<ascii::Char> {
        if self.is_ascii() {
            // SAFETY: Just checked that this is ASCII.
            Some(unsafe { ascii::Char::from_u8_unchecked(*self as u8) })
        } else {
            None
        }
    }

    /// Makes a copy of the value in its ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase()`].
    ///
    /// To uppercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_uppercase()`].
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
    /// [`make_ascii_uppercase()`]: #method.make_ascii_uppercase
    /// [`to_uppercase()`]: #method.to_uppercase
    #[must_use = "to uppercase the value in-place, use `make_ascii_uppercase()`"]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.52.0")]
    #[inline]
    pub const fn to_ascii_uppercase(&self) -> char {
        if self.is_ascii_lowercase() {
            (*self as u8).ascii_change_case_unchecked() as char
        } else {
            *self
        }
    }

    /// Makes a copy of the value in its ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase()`].
    ///
    /// To lowercase ASCII characters in addition to non-ASCII characters, use
    /// [`to_lowercase()`].
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
    /// [`make_ascii_lowercase()`]: #method.make_ascii_lowercase
    /// [`to_lowercase()`]: #method.to_lowercase
    #[must_use = "to lowercase the value in-place, use `make_ascii_lowercase()`"]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.52.0")]
    #[inline]
    pub const fn to_ascii_lowercase(&self) -> char {
        if self.is_ascii_uppercase() {
            (*self as u8).ascii_change_case_unchecked() as char
        } else {
            *self
        }
    }

    /// Checks that two values are an ASCII case-insensitive match.
    ///
    /// Equivalent to <code>[to_ascii_lowercase]\(a) == [to_ascii_lowercase]\(b)</code>.
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
    ///
    /// [to_ascii_lowercase]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.52.0")]
    #[inline]
    pub const fn eq_ignore_ascii_case(&self, other: &char) -> bool {
        self.to_ascii_lowercase() == other.to_ascii_lowercase()
    }

    /// Converts this type to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase()`].
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
    /// [`to_ascii_uppercase()`]: #method.to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_make_ascii", since = "1.84.0")]
    #[inline]
    pub const fn make_ascii_uppercase(&mut self) {
        *self = self.to_ascii_uppercase();
    }

    /// Converts this type to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase()`].
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
    /// [`to_ascii_lowercase()`]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_make_ascii", since = "1.84.0")]
    #[inline]
    pub const fn make_ascii_lowercase(&mut self) {
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_alphabetic(&self) -> bool {
        matches!(*self, 'A'..='Z' | 'a'..='z')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_uppercase(&self) -> bool {
        matches!(*self, 'A'..='Z')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_lowercase(&self) -> bool {
        matches!(*self, 'a'..='z')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_alphanumeric(&self) -> bool {
        matches!(*self, '0'..='9') | matches!(*self, 'A'..='Z') | matches!(*self, 'a'..='z')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_digit(&self) -> bool {
        matches!(*self, '0'..='9')
    }

    /// Checks if the value is an ASCII octal digit:
    /// U+0030 '0' ..= U+0037 '7'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(is_ascii_octdigit)]
    ///
    /// let uppercase_a = 'A';
    /// let a = 'a';
    /// let zero = '0';
    /// let seven = '7';
    /// let nine = '9';
    /// let percent = '%';
    /// let lf = '\n';
    ///
    /// assert!(!uppercase_a.is_ascii_octdigit());
    /// assert!(!a.is_ascii_octdigit());
    /// assert!(zero.is_ascii_octdigit());
    /// assert!(seven.is_ascii_octdigit());
    /// assert!(!nine.is_ascii_octdigit());
    /// assert!(!percent.is_ascii_octdigit());
    /// assert!(!lf.is_ascii_octdigit());
    /// ```
    #[must_use]
    #[unstable(feature = "is_ascii_octdigit", issue = "101288")]
    #[inline]
    pub const fn is_ascii_octdigit(&self) -> bool {
        matches!(*self, '0'..='7')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_hexdigit(&self) -> bool {
        matches!(*self, '0'..='9') | matches!(*self, 'A'..='F') | matches!(*self, 'a'..='f')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_punctuation(&self) -> bool {
        matches!(*self, '!'..='/')
            | matches!(*self, ':'..='@')
            | matches!(*self, '['..='`')
            | matches!(*self, '{'..='~')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_graphic(&self) -> bool {
        matches!(*self, '!'..='~')
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
    /// [pct]: https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap07.html#tag_07_03_01
    /// [bfs]: https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_05
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_whitespace(&self) -> bool {
        matches!(*self, '\t' | '\n' | '\x0C' | '\r' | ' ')
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
    /// let esc = '\x1b';
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
    #[must_use]
    #[stable(feature = "ascii_ctype_on_intrinsics", since = "1.24.0")]
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_control(&self) -> bool {
        matches!(*self, '\0'..='\x1F' | '\x7F')
    }
}

pub(crate) struct EscapeDebugExtArgs {
    /// Escape Extended Grapheme codepoints?
    pub(crate) escape_grapheme_extended: bool,

    /// Escape single quotes?
    pub(crate) escape_single_quote: bool,

    /// Escape double quotes?
    pub(crate) escape_double_quote: bool,
}

impl EscapeDebugExtArgs {
    pub(crate) const ESCAPE_ALL: Self = Self {
        escape_grapheme_extended: true,
        escape_single_quote: true,
        escape_double_quote: true,
    };
}

#[inline]
#[must_use]
const fn len_utf8(code: u32) -> usize {
    match code {
        ..MAX_ONE_B => 1,
        ..MAX_TWO_B => 2,
        ..MAX_THREE_B => 3,
        _ => 4,
    }
}

#[inline]
#[must_use]
const fn len_utf16(code: u32) -> usize {
    if (code & 0xFFFF) == code { 1 } else { 2 }
}

/// Encodes a raw `u32` value as UTF-8 into the provided byte buffer,
/// and then returns the subslice of the buffer that contains the encoded character.
///
/// Unlike `char::encode_utf8`, this method also handles codepoints in the surrogate range.
/// (Creating a `char` in the surrogate range is UB.)
/// The result is valid [generalized UTF-8] but not valid UTF-8.
///
/// [generalized UTF-8]: https://simonsapin.github.io/wtf-8/#generalized-utf8
///
/// # Panics
///
/// Panics if the buffer is not large enough.
/// A buffer of length four is large enough to encode any `char`.
#[unstable(feature = "char_internals", reason = "exposed only for libstd", issue = "none")]
#[doc(hidden)]
#[inline]
pub const fn encode_utf8_raw(code: u32, dst: &mut [u8]) -> &mut [u8] {
    let len = len_utf8(code);
    match (len, &mut *dst) {
        (1, [a, ..]) => {
            *a = code as u8;
        }
        (2, [a, b, ..]) => {
            *a = (code >> 6 & 0x1F) as u8 | TAG_TWO_B;
            *b = (code & 0x3F) as u8 | TAG_CONT;
        }
        (3, [a, b, c, ..]) => {
            *a = (code >> 12 & 0x0F) as u8 | TAG_THREE_B;
            *b = (code >> 6 & 0x3F) as u8 | TAG_CONT;
            *c = (code & 0x3F) as u8 | TAG_CONT;
        }
        (4, [a, b, c, d, ..]) => {
            *a = (code >> 18 & 0x07) as u8 | TAG_FOUR_B;
            *b = (code >> 12 & 0x3F) as u8 | TAG_CONT;
            *c = (code >> 6 & 0x3F) as u8 | TAG_CONT;
            *d = (code & 0x3F) as u8 | TAG_CONT;
        }
        _ => {
            const_panic!(
                "encode_utf8: buffer does not have enough bytes to encode code point",
                "encode_utf8: need {len} bytes to encode U+{code:04X} but buffer has just {dst_len}",
                code: u32 = code,
                len: usize = len,
                dst_len: usize = dst.len(),
            )
        }
    };
    // SAFETY: `<&mut [u8]>::as_mut_ptr` is guaranteed to return a valid pointer and `len` has been tested to be within bounds.
    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr(), len) }
}

/// Encodes a raw `u32` value as UTF-16 into the provided `u16` buffer,
/// and then returns the subslice of the buffer that contains the encoded character.
///
/// Unlike `char::encode_utf16`, this method also handles codepoints in the surrogate range.
/// (Creating a `char` in the surrogate range is UB.)
///
/// # Panics
///
/// Panics if the buffer is not large enough.
/// A buffer of length 2 is large enough to encode any `char`.
#[unstable(feature = "char_internals", reason = "exposed only for libstd", issue = "none")]
#[doc(hidden)]
#[inline]
pub const fn encode_utf16_raw(mut code: u32, dst: &mut [u16]) -> &mut [u16] {
    let len = len_utf16(code);
    match (len, &mut *dst) {
        (1, [a, ..]) => {
            *a = code as u16;
        }
        (2, [a, b, ..]) => {
            code -= 0x1_0000;
            *a = (code >> 10) as u16 | 0xD800;
            *b = (code & 0x3FF) as u16 | 0xDC00;
        }
        _ => {
            const_panic!(
                "encode_utf16: buffer does not have enough bytes to encode code point",
                "encode_utf16: need {len} bytes to encode U+{code:04X} but buffer has just {dst_len}",
                code: u32 = code,
                len: usize = len,
                dst_len: usize = dst.len(),
            )
        }
    };
    // SAFETY: `<&mut [u16]>::as_mut_ptr` is guaranteed to return a valid pointer and `len` has been tested to be within bounds.
    unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr(), len) }
}
