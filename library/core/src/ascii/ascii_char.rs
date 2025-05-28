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
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
    /// Creates an ascii character from the byte `b`,
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
            check_language_ub,
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
}

macro_rules! into_int_impl {
    ($($ty:ty)*) => {
        $(
            #[unstable(feature = "ascii_char", issue = "110998")]
            impl From<AsciiChar> for $ty {
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
