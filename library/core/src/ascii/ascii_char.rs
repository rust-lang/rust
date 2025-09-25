//! This uses the name `AsciiChar`, even though it's not exposed that way right now,
//! because it avoids a whole bunch of "are you sure you didn't mean `char`?"
//! suggestions from rustc if you get anything slightly wrong in here, and overall
//! helps with clarity as we're also referring to `char` intentionally in here.

use crate::mem::transmute;
use crate::{assert_unsafe_precondition, fmt};

/// One of the 128 Unicode characters from U+0000 through U+007F,
/// often known as the [ASCII] subset.
///
/// Officially, this is the first [block] in Unicode, _Basic Latin_.
/// For details, see the [*C0 Controls and Basic Latin*][chart] code chart.
///
/// This block was based on older 7-bit character code standards such as
/// ANSI X3.4-1977, ISO 646-1973, and [NIST FIPS 1-2].
///
/// # When to use this
///
/// The main advantage of this subset is that it's always valid UTF-8.  As such,
/// the `&[ascii::Char]` -> `&str` conversion function (as well as other related
/// ones) are O(1): *no* runtime checks are needed.
///
/// If you're consuming strings, you should usually handle Unicode and thus
/// accept `str`s, not limit yourself to `ascii::Char`s.
///
/// However, certain formats are intentionally designed to produce ASCII-only
/// output in order to be 8-bit-clean.  In those cases, it can be simpler and
/// faster to generate `ascii::Char`s instead of dealing with the variable width
/// properties of general UTF-8 encoded strings, while still allowing the result
/// to be used freely with other Rust things that deal in general `str`s.
///
/// For example, a UUID library might offer a way to produce the string
/// representation of a UUID as an `[ascii::Char; 36]` to avoid memory
/// allocation yet still allow it to be used as UTF-8 via `as_str` without
/// paying for validation (or needing `unsafe` code) the way it would if it
/// were provided as a `[u8; 36]`.
///
/// # Layout
///
/// This type is guaranteed to have a size and alignment of 1 byte.
///
/// # Names
///
/// The variants on this type are [Unicode names][NamesList] of the characters
/// in upper camel case, with a few tweaks:
/// - For `<control>` characters, the primary alias name is used.
/// - `LATIN` is dropped, as this block has no non-latin letters.
/// - `LETTER` is dropped, as `CAPITAL`/`SMALL` suffices in this block.
/// - `DIGIT`s use a single digit rather than writing out `ZERO`, `ONE`, etc.
///
/// [ASCII]: https://www.unicode.org/glossary/index.html#ASCII
/// [block]: https://www.unicode.org/glossary/index.html#block
/// [chart]: https://www.unicode.org/charts/PDF/U0000.pdf
/// [NIST FIPS 1-2]: https://nvlpubs.nist.gov/nistpubs/Legacy/FIPS/fipspub1-2-1977.pdf
/// [NamesList]: https://www.unicode.org/Public/15.0.0/ucd/NamesList.txt
#[derive(Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
#[unstable(feature = "ascii_char", issue = "110998")]
#[repr(u8)]
pub enum AsciiChar {
    /// U+0000 (The default variant)
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Null = 0,
    /// U+0001
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    StartOfHeading = 1,
    /// U+0002
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    StartOfText = 2,
    /// U+0003
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    EndOfText = 3,
    /// U+0004
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    EndOfTransmission = 4,
    /// U+0005
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Enquiry = 5,
    /// U+0006
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Acknowledge = 6,
    /// U+0007
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Bell = 7,
    /// U+0008
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Backspace = 8,
    /// U+0009
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CharacterTabulation = 9,
    /// U+000A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    LineFeed = 10,
    /// U+000B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    LineTabulation = 11,
    /// U+000C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    FormFeed = 12,
    /// U+000D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CarriageReturn = 13,
    /// U+000E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    ShiftOut = 14,
    /// U+000F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    ShiftIn = 15,
    /// U+0010
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    DataLinkEscape = 16,
    /// U+0011
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    DeviceControlOne = 17,
    /// U+0012
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    DeviceControlTwo = 18,
    /// U+0013
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    DeviceControlThree = 19,
    /// U+0014
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    DeviceControlFour = 20,
    /// U+0015
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    NegativeAcknowledge = 21,
    /// U+0016
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SynchronousIdle = 22,
    /// U+0017
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    EndOfTransmissionBlock = 23,
    /// U+0018
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Cancel = 24,
    /// U+0019
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    EndOfMedium = 25,
    /// U+001A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Substitute = 26,
    /// U+001B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Escape = 27,
    /// U+001C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    InformationSeparatorFour = 28,
    /// U+001D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    InformationSeparatorThree = 29,
    /// U+001E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    InformationSeparatorTwo = 30,
    /// U+001F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    InformationSeparatorOne = 31,
    /// U+0020
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Space = 32,
    /// U+0021
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    ExclamationMark = 33,
    /// U+0022
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    QuotationMark = 34,
    /// U+0023
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    NumberSign = 35,
    /// U+0024
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    DollarSign = 36,
    /// U+0025
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    PercentSign = 37,
    /// U+0026
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Ampersand = 38,
    /// U+0027
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Apostrophe = 39,
    /// U+0028
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    LeftParenthesis = 40,
    /// U+0029
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    RightParenthesis = 41,
    /// U+002A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Asterisk = 42,
    /// U+002B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    PlusSign = 43,
    /// U+002C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Comma = 44,
    /// U+002D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    HyphenMinus = 45,
    /// U+002E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    FullStop = 46,
    /// U+002F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Solidus = 47,
    /// U+0030
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit0 = 48,
    /// U+0031
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit1 = 49,
    /// U+0032
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit2 = 50,
    /// U+0033
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit3 = 51,
    /// U+0034
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit4 = 52,
    /// U+0035
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit5 = 53,
    /// U+0036
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit6 = 54,
    /// U+0037
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit7 = 55,
    /// U+0038
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit8 = 56,
    /// U+0039
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Digit9 = 57,
    /// U+003A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Colon = 58,
    /// U+003B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Semicolon = 59,
    /// U+003C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    LessThanSign = 60,
    /// U+003D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    EqualsSign = 61,
    /// U+003E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    GreaterThanSign = 62,
    /// U+003F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    QuestionMark = 63,
    /// U+0040
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CommercialAt = 64,
    /// U+0041
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalA = 65,
    /// U+0042
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalB = 66,
    /// U+0043
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalC = 67,
    /// U+0044
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalD = 68,
    /// U+0045
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalE = 69,
    /// U+0046
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalF = 70,
    /// U+0047
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalG = 71,
    /// U+0048
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalH = 72,
    /// U+0049
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalI = 73,
    /// U+004A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalJ = 74,
    /// U+004B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalK = 75,
    /// U+004C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalL = 76,
    /// U+004D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalM = 77,
    /// U+004E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalN = 78,
    /// U+004F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalO = 79,
    /// U+0050
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalP = 80,
    /// U+0051
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalQ = 81,
    /// U+0052
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalR = 82,
    /// U+0053
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalS = 83,
    /// U+0054
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalT = 84,
    /// U+0055
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalU = 85,
    /// U+0056
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalV = 86,
    /// U+0057
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalW = 87,
    /// U+0058
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalX = 88,
    /// U+0059
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalY = 89,
    /// U+005A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CapitalZ = 90,
    /// U+005B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    LeftSquareBracket = 91,
    /// U+005C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    ReverseSolidus = 92,
    /// U+005D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    RightSquareBracket = 93,
    /// U+005E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    CircumflexAccent = 94,
    /// U+005F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    LowLine = 95,
    /// U+0060
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    GraveAccent = 96,
    /// U+0061
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallA = 97,
    /// U+0062
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallB = 98,
    /// U+0063
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallC = 99,
    /// U+0064
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallD = 100,
    /// U+0065
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallE = 101,
    /// U+0066
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallF = 102,
    /// U+0067
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallG = 103,
    /// U+0068
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallH = 104,
    /// U+0069
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallI = 105,
    /// U+006A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallJ = 106,
    /// U+006B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallK = 107,
    /// U+006C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallL = 108,
    /// U+006D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallM = 109,
    /// U+006E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallN = 110,
    /// U+006F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallO = 111,
    /// U+0070
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallP = 112,
    /// U+0071
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallQ = 113,
    /// U+0072
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallR = 114,
    /// U+0073
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallS = 115,
    /// U+0074
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallT = 116,
    /// U+0075
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallU = 117,
    /// U+0076
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallV = 118,
    /// U+0077
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallW = 119,
    /// U+0078
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallX = 120,
    /// U+0079
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallY = 121,
    /// U+007A
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    SmallZ = 122,
    /// U+007B
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    LeftCurlyBracket = 123,
    /// U+007C
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    VerticalLine = 124,
    /// U+007D
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    RightCurlyBracket = 125,
    /// U+007E
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Tilde = 126,
    /// U+007F
    #[unstable(feature = "ascii_char_variants", issue = "110998")]
    Delete = 127,
}

impl AsciiChar {
    /// The character with the lowest ASCII code.
    #[unstable(feature = "ascii_char", issue = "110998")]
    pub const MIN: Self = Self::Null;

    /// The character with the highest ASCII code.
    #[unstable(feature = "ascii_char", issue = "110998")]
    pub const MAX: Self = Self::Delete;

    /// Creates an ASCII character from the byte `b`,
    /// or returns `None` if it's too large.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn from_u8(b: u8) -> Option<Self> {
        if b <= 127 {
            // SAFETY: Just checked that `b` is in-range
            Some(unsafe { Self::from_u8_unchecked(b) })
        } else {
            None
        }
    }

    /// Creates an ASCII character from the byte `b`,
    /// without checking whether it's valid.
    ///
    /// # Safety
    ///
    /// `b` must be in `0..=127`, or else this is UB.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const unsafe fn from_u8_unchecked(b: u8) -> Self {
        // SAFETY: Our safety precondition is that `b` is in-range.
        unsafe { transmute(b) }
    }

    /// When passed the *number* `0`, `1`, …, `9`, returns the *character*
    /// `'0'`, `'1'`, …, `'9'` respectively.
    ///
    /// If `d >= 10`, returns `None`.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn digit(d: u8) -> Option<Self> {
        if d < 10 {
            // SAFETY: Just checked it's in-range.
            Some(unsafe { Self::digit_unchecked(d) })
        } else {
            None
        }
    }

    /// When passed the *number* `0`, `1`, …, `9`, returns the *character*
    /// `'0'`, `'1'`, …, `'9'` respectively, without checking that it's in-range.
    ///
    /// # Safety
    ///
    /// This is immediate UB if called with `d > 64`.
    ///
    /// If `d >= 10` and `d <= 64`, this is allowed to return any value or panic.
    /// Notably, it should not be expected to return hex digits, or any other
    /// reasonable extension of the decimal digits.
    ///
    /// (This loose safety condition is intended to simplify soundness proofs
    /// when writing code using this method, since the implementation doesn't
    /// need something really specific, not to make those other arguments do
    /// something useful. It might be tightened before stabilization.)
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    #[track_caller]
    pub const unsafe fn digit_unchecked(d: u8) -> Self {
        assert_unsafe_precondition!(
            check_library_ub,
            "`ascii::Char::digit_unchecked` input cannot exceed 9.",
            (d: u8 = d) => d < 10
        );

        // SAFETY: `'0'` through `'9'` are U+00030 through U+0039,
        // so because `d` must be 64 or less the addition can return at most
        // 112 (0x70), which doesn't overflow and is within the ASCII range.
        unsafe {
            let byte = b'0'.unchecked_add(d);
            Self::from_u8_unchecked(byte)
        }
    }

    /// Gets this ASCII character as a byte.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Gets this ASCII character as a `char` Unicode Scalar Value.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn to_char(self) -> char {
        self as u8 as char
    }

    /// Views this ASCII character as a one-code-unit UTF-8 `str`.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn as_str(&self) -> &str {
        crate::slice::from_ref(self).as_str()
    }

    /// Makes a copy of the value in its upper case equivalent.
    ///
    /// Letters 'a' to 'z' are mapped to 'A' to 'Z'.
    ///
    /// To uppercase the value in-place, use [`make_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let lowercase_a = ascii::Char::SmallA;
    ///
    /// assert_eq!(
    ///     ascii::Char::CapitalA,
    ///     lowercase_a.to_uppercase(),
    /// );
    /// ```
    ///
    /// [`make_uppercase`]: Self::make_uppercase
    #[must_use = "to uppercase the value in-place, use `make_uppercase()`"]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn to_uppercase(self) -> Self {
        let uppercase_byte = self.to_u8().to_ascii_uppercase();
        // SAFETY: Toggling the 6th bit won't convert ASCII to non-ASCII.
        unsafe { Self::from_u8_unchecked(uppercase_byte) }
    }

    /// Makes a copy of the value in its lower case equivalent.
    ///
    /// Letters 'A' to 'Z' are mapped to 'a' to 'z'.
    ///
    /// To lowercase the value in-place, use [`make_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    ///
    /// assert_eq!(
    ///     ascii::Char::SmallA,
    ///     uppercase_a.to_lowercase(),
    /// );
    /// ```
    ///
    /// [`make_lowercase`]: Self::make_lowercase
    #[must_use = "to lowercase the value in-place, use `make_lowercase()`"]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn to_lowercase(self) -> Self {
        let lowercase_byte = self.to_u8().to_ascii_lowercase();
        // SAFETY: Setting the 6th bit won't convert ASCII to non-ASCII.
        unsafe { Self::from_u8_unchecked(lowercase_byte) }
    }

    /// Checks that two values are a case-insensitive match.
    ///
    /// This is equivalent to `to_lowercase(a) == to_lowercase(b)`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let lowercase_a = ascii::Char::SmallA;
    /// let uppercase_a = ascii::Char::CapitalA;
    ///
    /// assert!(lowercase_a.eq_ignore_case(uppercase_a));
    /// ```
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn eq_ignore_case(self, other: Self) -> bool {
        // FIXME(const-hack) `arg.to_u8().to_ascii_lowercase()` -> `arg.to_lowercase()`
        // once `PartialEq` is const for `Self`.
        self.to_u8().to_ascii_lowercase() == other.to_u8().to_ascii_lowercase()
    }

    /// Converts this value to its upper case equivalent in-place.
    ///
    /// Letters 'a' to 'z' are mapped to 'A' to 'Z'.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let mut letter_a = ascii::Char::SmallA;
    ///
    /// letter_a.make_uppercase();
    ///
    /// assert_eq!(ascii::Char::CapitalA, letter_a);
    /// ```
    ///
    /// [`to_uppercase`]: Self::to_uppercase
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn make_uppercase(&mut self) {
        *self = self.to_uppercase();
    }

    /// Converts this value to its lower case equivalent in-place.
    ///
    /// Letters 'A' to 'Z' are mapped to 'a' to 'z'.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let mut letter_a = ascii::Char::CapitalA;
    ///
    /// letter_a.make_lowercase();
    ///
    /// assert_eq!(ascii::Char::SmallA, letter_a);
    /// ```
    ///
    /// [`to_lowercase`]: Self::to_lowercase
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn make_lowercase(&mut self) {
        *self = self.to_lowercase();
    }

    /// Checks if the value is an alphabetic character:
    ///
    /// - 0x41 'A' ..= 0x5A 'Z', or
    /// - 0x61 'a' ..= 0x7A 'z'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(uppercase_a.is_alphabetic());
    /// assert!(uppercase_g.is_alphabetic());
    /// assert!(a.is_alphabetic());
    /// assert!(g.is_alphabetic());
    /// assert!(!zero.is_alphabetic());
    /// assert!(!percent.is_alphabetic());
    /// assert!(!space.is_alphabetic());
    /// assert!(!lf.is_alphabetic());
    /// assert!(!esc.is_alphabetic());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_alphabetic(self) -> bool {
        self.to_u8().is_ascii_alphabetic()
    }

    /// Checks if the value is an uppercase character:
    /// 0x41 'A' ..= 0x5A 'Z'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(uppercase_a.is_uppercase());
    /// assert!(uppercase_g.is_uppercase());
    /// assert!(!a.is_uppercase());
    /// assert!(!g.is_uppercase());
    /// assert!(!zero.is_uppercase());
    /// assert!(!percent.is_uppercase());
    /// assert!(!space.is_uppercase());
    /// assert!(!lf.is_uppercase());
    /// assert!(!esc.is_uppercase());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_uppercase(self) -> bool {
        self.to_u8().is_ascii_uppercase()
    }

    /// Checks if the value is a lowercase character:
    /// 0x61 'a' ..= 0x7A 'z'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(!uppercase_a.is_lowercase());
    /// assert!(!uppercase_g.is_lowercase());
    /// assert!(a.is_lowercase());
    /// assert!(g.is_lowercase());
    /// assert!(!zero.is_lowercase());
    /// assert!(!percent.is_lowercase());
    /// assert!(!space.is_lowercase());
    /// assert!(!lf.is_lowercase());
    /// assert!(!esc.is_lowercase());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_lowercase(self) -> bool {
        self.to_u8().is_ascii_lowercase()
    }

    /// Checks if the value is an alphanumeric character:
    ///
    /// - 0x41 'A' ..= 0x5A 'Z', or
    /// - 0x61 'a' ..= 0x7A 'z', or
    /// - 0x30 '0' ..= 0x39 '9'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(uppercase_a.is_alphanumeric());
    /// assert!(uppercase_g.is_alphanumeric());
    /// assert!(a.is_alphanumeric());
    /// assert!(g.is_alphanumeric());
    /// assert!(zero.is_alphanumeric());
    /// assert!(!percent.is_alphanumeric());
    /// assert!(!space.is_alphanumeric());
    /// assert!(!lf.is_alphanumeric());
    /// assert!(!esc.is_alphanumeric());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_alphanumeric(self) -> bool {
        self.to_u8().is_ascii_alphanumeric()
    }

    /// Checks if the value is a decimal digit:
    /// 0x30 '0' ..= 0x39 '9'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(!uppercase_a.is_digit());
    /// assert!(!uppercase_g.is_digit());
    /// assert!(!a.is_digit());
    /// assert!(!g.is_digit());
    /// assert!(zero.is_digit());
    /// assert!(!percent.is_digit());
    /// assert!(!space.is_digit());
    /// assert!(!lf.is_digit());
    /// assert!(!esc.is_digit());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_digit(self) -> bool {
        self.to_u8().is_ascii_digit()
    }

    /// Checks if the value is an octal digit:
    /// 0x30 '0' ..= 0x37 '7'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants, is_ascii_octdigit)]
    ///
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let a = ascii::Char::SmallA;
    /// let zero = ascii::Char::Digit0;
    /// let seven = ascii::Char::Digit7;
    /// let eight = ascii::Char::Digit8;
    /// let percent = ascii::Char::PercentSign;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(!uppercase_a.is_octdigit());
    /// assert!(!a.is_octdigit());
    /// assert!(zero.is_octdigit());
    /// assert!(seven.is_octdigit());
    /// assert!(!eight.is_octdigit());
    /// assert!(!percent.is_octdigit());
    /// assert!(!lf.is_octdigit());
    /// assert!(!esc.is_octdigit());
    /// ```
    #[must_use]
    // This is blocked on two unstable features. Please ensure both are
    // stabilized before marking this method as stable.
    #[unstable(feature = "ascii_char", issue = "110998")]
    // #[unstable(feature = "is_ascii_octdigit", issue = "101288")]
    #[inline]
    pub const fn is_octdigit(self) -> bool {
        self.to_u8().is_ascii_octdigit()
    }

    /// Checks if the value is a hexadecimal digit:
    ///
    /// - 0x30 '0' ..= 0x39 '9', or
    /// - 0x41 'A' ..= 0x46 'F', or
    /// - 0x61 'a' ..= 0x66 'f'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(uppercase_a.is_hexdigit());
    /// assert!(!uppercase_g.is_hexdigit());
    /// assert!(a.is_hexdigit());
    /// assert!(!g.is_hexdigit());
    /// assert!(zero.is_hexdigit());
    /// assert!(!percent.is_hexdigit());
    /// assert!(!space.is_hexdigit());
    /// assert!(!lf.is_hexdigit());
    /// assert!(!esc.is_hexdigit());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_hexdigit(self) -> bool {
        self.to_u8().is_ascii_hexdigit()
    }

    /// Checks if the value is a punctuation character:
    ///
    /// - 0x21 ..= 0x2F `! " # $ % & ' ( ) * + , - . /`, or
    /// - 0x3A ..= 0x40 `: ; < = > ? @`, or
    /// - 0x5B ..= 0x60 `` [ \ ] ^ _ ` ``, or
    /// - 0x7B ..= 0x7E `{ | } ~`
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(!uppercase_a.is_punctuation());
    /// assert!(!uppercase_g.is_punctuation());
    /// assert!(!a.is_punctuation());
    /// assert!(!g.is_punctuation());
    /// assert!(!zero.is_punctuation());
    /// assert!(percent.is_punctuation());
    /// assert!(!space.is_punctuation());
    /// assert!(!lf.is_punctuation());
    /// assert!(!esc.is_punctuation());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_punctuation(self) -> bool {
        self.to_u8().is_ascii_punctuation()
    }

    /// Checks if the value is a graphic character:
    /// 0x21 '!' ..= 0x7E '~'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(uppercase_a.is_graphic());
    /// assert!(uppercase_g.is_graphic());
    /// assert!(a.is_graphic());
    /// assert!(g.is_graphic());
    /// assert!(zero.is_graphic());
    /// assert!(percent.is_graphic());
    /// assert!(!space.is_graphic());
    /// assert!(!lf.is_graphic());
    /// assert!(!esc.is_graphic());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_graphic(self) -> bool {
        self.to_u8().is_ascii_graphic()
    }

    /// Checks if the value is a whitespace character:
    /// 0x20 SPACE, 0x09 HORIZONTAL TAB, 0x0A LINE FEED,
    /// 0x0C FORM FEED, or 0x0D CARRIAGE RETURN.
    ///
    /// Rust uses the WhatWG Infra Standard's [definition of ASCII
    /// whitespace][infra-aw]. There are several other definitions in
    /// wide use. For instance, [the POSIX locale][pct] includes
    /// 0x0B VERTICAL TAB as well as all the above characters,
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
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(!uppercase_a.is_whitespace());
    /// assert!(!uppercase_g.is_whitespace());
    /// assert!(!a.is_whitespace());
    /// assert!(!g.is_whitespace());
    /// assert!(!zero.is_whitespace());
    /// assert!(!percent.is_whitespace());
    /// assert!(space.is_whitespace());
    /// assert!(lf.is_whitespace());
    /// assert!(!esc.is_whitespace());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_whitespace(self) -> bool {
        self.to_u8().is_ascii_whitespace()
    }

    /// Checks if the value is a control character:
    /// 0x00 NUL ..= 0x1F UNIT SEPARATOR, or 0x7F DELETE.
    /// Note that most whitespace characters are control
    /// characters, but SPACE is not.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let uppercase_a = ascii::Char::CapitalA;
    /// let uppercase_g = ascii::Char::CapitalG;
    /// let a = ascii::Char::SmallA;
    /// let g = ascii::Char::SmallG;
    /// let zero = ascii::Char::Digit0;
    /// let percent = ascii::Char::PercentSign;
    /// let space = ascii::Char::Space;
    /// let lf = ascii::Char::LineFeed;
    /// let esc = ascii::Char::Escape;
    ///
    /// assert!(!uppercase_a.is_control());
    /// assert!(!uppercase_g.is_control());
    /// assert!(!a.is_control());
    /// assert!(!g.is_control());
    /// assert!(!zero.is_control());
    /// assert!(!percent.is_control());
    /// assert!(!space.is_control());
    /// assert!(lf.is_control());
    /// assert!(esc.is_control());
    /// ```
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn is_control(self) -> bool {
        self.to_u8().is_ascii_control()
    }

    /// Returns an iterator that produces an escaped version of a
    /// character.
    ///
    /// The behavior is identical to
    /// [`ascii::escape_default`](crate::ascii::escape_default).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(ascii_char, ascii_char_variants)]
    /// use std::ascii;
    ///
    /// let zero = ascii::Char::Digit0;
    /// let tab = ascii::Char::CharacterTabulation;
    /// let cr = ascii::Char::CarriageReturn;
    /// let lf = ascii::Char::LineFeed;
    /// let apostrophe = ascii::Char::Apostrophe;
    /// let double_quote = ascii::Char::QuotationMark;
    /// let backslash = ascii::Char::ReverseSolidus;
    ///
    /// assert_eq!("0", zero.escape_ascii().to_string());
    /// assert_eq!("\\t", tab.escape_ascii().to_string());
    /// assert_eq!("\\r", cr.escape_ascii().to_string());
    /// assert_eq!("\\n", lf.escape_ascii().to_string());
    /// assert_eq!("\\'", apostrophe.escape_ascii().to_string());
    /// assert_eq!("\\\"", double_quote.escape_ascii().to_string());
    /// assert_eq!("\\\\", backslash.escape_ascii().to_string());
    /// ```
    #[must_use = "this returns the escaped character as an iterator, \
                  without modifying the original"]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub fn escape_ascii(self) -> super::EscapeDefault {
        super::escape_default(self.to_u8())
    }
}

macro_rules! into_int_impl {
    ($($ty:ty)*) => {
        $(
            #[unstable(feature = "ascii_char", issue = "110998")]
            #[rustc_const_unstable(feature = "const_convert", issue = "143773")]
            impl const From<AsciiChar> for $ty {
                #[inline]
                fn from(chr: AsciiChar) -> $ty {
                    chr as u8 as $ty
                }
            }
        )*
    }
}

into_int_impl!(u8 u16 u32 u64 u128 char);

impl [AsciiChar] {
    /// Views this slice of ASCII characters as a UTF-8 `str`.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn as_str(&self) -> &str {
        let ascii_ptr: *const Self = self;
        let str_ptr = ascii_ptr as *const str;
        // SAFETY: Each ASCII codepoint in UTF-8 is encoded as one single-byte
        // code unit having the same value as the ASCII byte.
        unsafe { &*str_ptr }
    }

    /// Views this slice of ASCII characters as a slice of `u8` bytes.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn as_bytes(&self) -> &[u8] {
        self.as_str().as_bytes()
    }
}

#[unstable(feature = "ascii_char", issue = "110998")]
impl fmt::Display for AsciiChar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <str as fmt::Display>::fmt(self.as_str(), f)
    }
}

#[unstable(feature = "ascii_char", issue = "110998")]
impl fmt::Debug for AsciiChar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use AsciiChar::{Apostrophe, Null, ReverseSolidus as Backslash};

        fn backslash(a: AsciiChar) -> ([AsciiChar; 6], usize) {
            ([Apostrophe, Backslash, a, Apostrophe, Null, Null], 4)
        }

        let (buf, len) = match self {
            AsciiChar::Null => backslash(AsciiChar::Digit0),
            AsciiChar::CharacterTabulation => backslash(AsciiChar::SmallT),
            AsciiChar::CarriageReturn => backslash(AsciiChar::SmallR),
            AsciiChar::LineFeed => backslash(AsciiChar::SmallN),
            AsciiChar::ReverseSolidus => backslash(AsciiChar::ReverseSolidus),
            AsciiChar::Apostrophe => backslash(AsciiChar::Apostrophe),
            _ if self.to_u8().is_ascii_control() => {
                const HEX_DIGITS: [AsciiChar; 16] = *b"0123456789abcdef".as_ascii().unwrap();

                let byte = self.to_u8();
                let hi = HEX_DIGITS[usize::from(byte >> 4)];
                let lo = HEX_DIGITS[usize::from(byte & 0xf)];
                ([Apostrophe, Backslash, AsciiChar::SmallX, hi, lo, Apostrophe], 6)
            }
            _ => ([Apostrophe, *self, Apostrophe, Null, Null, Null], 3),
        };

        f.write_str(buf[..len].as_str())
    }
}
