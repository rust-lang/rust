//! Numeric traits and functions for the built-in numeric types.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::ascii;
use crate::intrinsics;
use crate::mem;
use crate::str::FromStr;

// Used because the `?` operator is not allowed in a const context.
macro_rules! try_opt {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => return None,
        }
    };
}

#[allow_internal_unstable(const_likely)]
macro_rules! unlikely {
    ($e: expr) => {
        intrinsics::unlikely($e)
    };
}

// All these modules are technically private and only exposed for coretests:
#[cfg(not(no_fp_fmt_parse))]
pub mod bignum;
#[cfg(not(no_fp_fmt_parse))]
pub mod dec2flt;
#[cfg(not(no_fp_fmt_parse))]
pub mod diy_float;
#[cfg(not(no_fp_fmt_parse))]
pub mod flt2dec;
pub mod fmt;

#[macro_use]
mod int_macros; // import int_impl!
#[macro_use]
mod uint_macros; // import uint_impl!

mod error;
mod int_log10;
mod nonzero;
#[unstable(feature = "saturating_int_impl", issue = "87920")]
mod saturating;
mod wrapping;

#[unstable(feature = "saturating_int_impl", issue = "87920")]
pub use saturating::Saturating;
#[stable(feature = "rust1", since = "1.0.0")]
pub use wrapping::Wrapping;

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(not(no_fp_fmt_parse))]
pub use dec2flt::ParseFloatError;

#[stable(feature = "rust1", since = "1.0.0")]
pub use error::ParseIntError;

#[stable(feature = "nonzero", since = "1.28.0")]
pub use nonzero::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize};

#[stable(feature = "signed_nonzero", since = "1.34.0")]
pub use nonzero::{NonZeroI128, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI8, NonZeroIsize};

#[stable(feature = "try_from", since = "1.34.0")]
pub use error::TryFromIntError;

#[stable(feature = "int_error_matching", since = "1.55.0")]
pub use error::IntErrorKind;

macro_rules! usize_isize_to_xe_bytes_doc {
    () => {
        "

**Note**: This function returns an array of length 2, 4 or 8 bytes
depending on the target pointer size.

"
    };
}

macro_rules! usize_isize_from_xe_bytes_doc {
    () => {
        "

**Note**: This function takes an array of length 2, 4 or 8 bytes
depending on the target pointer size.

"
    };
}

#[lang = "i8"]
impl i8 {
    int_impl! { i8, i8, u8, 8, 7, -128, 127, 2, "-0x7e", "0xa", "0x12", "0x12", "0x48",
    "[0x12]", "[0x12]", "", "" }
}

#[lang = "i16"]
impl i16 {
    int_impl! { i16, i16, u16, 16, 15, -32768, 32767, 4, "-0x5ffd", "0x3a", "0x1234", "0x3412",
    "0x2c48", "[0x34, 0x12]", "[0x12, 0x34]", "", "" }
}

#[lang = "i32"]
impl i32 {
    int_impl! { i32, i32, u32, 32, 31, -2147483648, 2147483647, 8, "0x10000b3", "0xb301",
    "0x12345678", "0x78563412", "0x1e6a2c48", "[0x78, 0x56, 0x34, 0x12]",
    "[0x12, 0x34, 0x56, 0x78]", "", "" }
}

#[lang = "i64"]
impl i64 {
    int_impl! { i64, i64, u64, 64, 63, -9223372036854775808, 9223372036854775807, 12,
    "0xaa00000000006e1", "0x6e10aa", "0x1234567890123456", "0x5634129078563412",
    "0x6a2c48091e6a2c48", "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
    "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]", "", "" }
}

#[lang = "i128"]
impl i128 {
    int_impl! { i128, i128, u128, 128, 127, -170141183460469231731687303715884105728,
    170141183460469231731687303715884105727, 16,
    "0x13f40000000000000000000000004f76", "0x4f7613f4", "0x12345678901234567890123456789012",
    "0x12907856341290785634129078563412", "0x48091e6a2c48091e6a2c48091e6a2c48",
    "[0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, \
      0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
    "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, \
      0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12]", "", "" }
}

#[cfg(target_pointer_width = "16")]
#[lang = "isize"]
impl isize {
    int_impl! { isize, i16, usize, 16, 15, -32768, 32767, 4, "-0x5ffd", "0x3a", "0x1234",
    "0x3412", "0x2c48", "[0x34, 0x12]", "[0x12, 0x34]",
    usize_isize_to_xe_bytes_doc!(), usize_isize_from_xe_bytes_doc!() }
}

#[cfg(target_pointer_width = "32")]
#[lang = "isize"]
impl isize {
    int_impl! { isize, i32, usize, 32, 31, -2147483648, 2147483647, 8, "0x10000b3", "0xb301",
    "0x12345678", "0x78563412", "0x1e6a2c48", "[0x78, 0x56, 0x34, 0x12]",
    "[0x12, 0x34, 0x56, 0x78]",
    usize_isize_to_xe_bytes_doc!(), usize_isize_from_xe_bytes_doc!() }
}

#[cfg(target_pointer_width = "64")]
#[lang = "isize"]
impl isize {
    int_impl! { isize, i64, usize, 64, 63, -9223372036854775808, 9223372036854775807,
    12, "0xaa00000000006e1", "0x6e10aa",  "0x1234567890123456", "0x5634129078563412",
     "0x6a2c48091e6a2c48", "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
     "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]",
     usize_isize_to_xe_bytes_doc!(), usize_isize_from_xe_bytes_doc!() }
}

/// If 6th bit set ascii is upper case.
const ASCII_CASE_MASK: u8 = 0b0010_0000;

#[lang = "u8"]
impl u8 {
    uint_impl! { u8, u8, 8, 255, 2, "0x82", "0xa", "0x12", "0x12", "0x48", "[0x12]",
    "[0x12]", "", "" }

    /// Checks if the value is within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii = 97u8;
    /// let non_ascii = 150u8;
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.43.0")]
    #[inline]
    pub const fn is_ascii(&self) -> bool {
        *self & 128 == 0
    }

    /// Makes a copy of the value in its ASCII upper case equivalent.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To uppercase the value in-place, use [`make_ascii_uppercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let lowercase_a = 97u8;
    ///
    /// assert_eq!(65, lowercase_a.to_ascii_uppercase());
    /// ```
    ///
    /// [`make_ascii_uppercase`]: Self::make_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.52.0")]
    #[inline]
    pub const fn to_ascii_uppercase(&self) -> u8 {
        // Unset the fifth bit if this is a lowercase letter
        *self & !((self.is_ascii_lowercase() as u8) * ASCII_CASE_MASK)
    }

    /// Makes a copy of the value in its ASCII lower case equivalent.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To lowercase the value in-place, use [`make_ascii_lowercase`].
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = 65u8;
    ///
    /// assert_eq!(97, uppercase_a.to_ascii_lowercase());
    /// ```
    ///
    /// [`make_ascii_lowercase`]: Self::make_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.52.0")]
    #[inline]
    pub const fn to_ascii_lowercase(&self) -> u8 {
        // Set the fifth bit if this is an uppercase letter
        *self | (self.is_ascii_uppercase() as u8 * ASCII_CASE_MASK)
    }

    /// Assumes self is ascii
    #[inline]
    pub(crate) const fn ascii_change_case_unchecked(&self) -> u8 {
        *self ^ ASCII_CASE_MASK
    }

    /// Checks that two values are an ASCII case-insensitive match.
    ///
    /// This is equivalent to `to_ascii_lowercase(a) == to_ascii_lowercase(b)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let lowercase_a = 97u8;
    /// let uppercase_a = 65u8;
    ///
    /// assert!(lowercase_a.eq_ignore_ascii_case(&uppercase_a));
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.52.0")]
    #[inline]
    pub const fn eq_ignore_ascii_case(&self, other: &u8) -> bool {
        self.to_ascii_lowercase() == other.to_ascii_lowercase()
    }

    /// Converts this value to its ASCII upper case equivalent in-place.
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
    /// let mut byte = b'a';
    ///
    /// byte.make_ascii_uppercase();
    ///
    /// assert_eq!(b'A', byte);
    /// ```
    ///
    /// [`to_ascii_uppercase`]: Self::to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        *self = self.to_ascii_uppercase();
    }

    /// Converts this value to its ASCII lower case equivalent in-place.
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
    /// let mut byte = b'A';
    ///
    /// byte.make_ascii_lowercase();
    ///
    /// assert_eq!(b'a', byte);
    /// ```
    ///
    /// [`to_ascii_lowercase`]: Self::to_ascii_lowercase
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
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_alphabetic(&self) -> bool {
        matches!(*self, b'A'..=b'Z' | b'a'..=b'z')
    }

    /// Checks if the value is an ASCII uppercase character:
    /// U+0041 'A' ..= U+005A 'Z'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_uppercase(&self) -> bool {
        matches!(*self, b'A'..=b'Z')
    }

    /// Checks if the value is an ASCII lowercase character:
    /// U+0061 'a' ..= U+007A 'z'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_lowercase(&self) -> bool {
        matches!(*self, b'a'..=b'z')
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
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_alphanumeric(&self) -> bool {
        matches!(*self, b'0'..=b'9' | b'A'..=b'Z' | b'a'..=b'z')
    }

    /// Checks if the value is an ASCII decimal digit:
    /// U+0030 '0' ..= U+0039 '9'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_digit(&self) -> bool {
        matches!(*self, b'0'..=b'9')
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
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_hexdigit(&self) -> bool {
        matches!(*self, b'0'..=b'9' | b'A'..=b'F' | b'a'..=b'f')
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
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_punctuation(&self) -> bool {
        matches!(*self, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~')
    }

    /// Checks if the value is an ASCII graphic character:
    /// U+0021 '!' ..= U+007E '~'.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_graphic(&self) -> bool {
        matches!(*self, b'!'..=b'~')
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
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_whitespace(&self) -> bool {
        matches!(*self, b'\t' | b'\n' | b'\x0C' | b'\r' | b' ')
    }

    /// Checks if the value is an ASCII control character:
    /// U+0000 NUL ..= U+001F UNIT SEPARATOR, or U+007F DELETE.
    /// Note that most ASCII whitespace characters are control
    /// characters, but SPACE is not.
    ///
    /// # Examples
    ///
    /// ```
    /// let uppercase_a = b'A';
    /// let uppercase_g = b'G';
    /// let a = b'a';
    /// let g = b'g';
    /// let zero = b'0';
    /// let percent = b'%';
    /// let space = b' ';
    /// let lf = b'\n';
    /// let esc = 0x1b_u8;
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
    #[rustc_const_stable(feature = "const_ascii_ctype_on_intrinsics", since = "1.47.0")]
    #[inline]
    pub const fn is_ascii_control(&self) -> bool {
        matches!(*self, b'\0'..=b'\x1F' | b'\x7F')
    }

    /// Returns an iterator that produces an escaped version of a `u8`,
    /// treating it as an ASCII character.
    ///
    /// The behavior is identical to [`ascii::escape_default`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(inherent_ascii_escape)]
    ///
    /// assert_eq!("0", b'0'.escape_ascii().to_string());
    /// assert_eq!("\\t", b'\t'.escape_ascii().to_string());
    /// assert_eq!("\\r", b'\r'.escape_ascii().to_string());
    /// assert_eq!("\\n", b'\n'.escape_ascii().to_string());
    /// assert_eq!("\\'", b'\''.escape_ascii().to_string());
    /// assert_eq!("\\\"", b'"'.escape_ascii().to_string());
    /// assert_eq!("\\\\", b'\\'.escape_ascii().to_string());
    /// assert_eq!("\\x9d", b'\x9d'.escape_ascii().to_string());
    /// ```
    #[unstable(feature = "inherent_ascii_escape", issue = "77174")]
    #[inline]
    pub fn escape_ascii(&self) -> ascii::EscapeDefault {
        ascii::escape_default(*self)
    }
}

#[lang = "u16"]
impl u16 {
    uint_impl! { u16, u16, 16, 65535, 4, "0xa003", "0x3a", "0x1234", "0x3412", "0x2c48",
    "[0x34, 0x12]", "[0x12, 0x34]", "", "" }
}

#[lang = "u32"]
impl u32 {
    uint_impl! { u32, u32, 32, 4294967295, 8, "0x10000b3", "0xb301", "0x12345678",
    "0x78563412", "0x1e6a2c48", "[0x78, 0x56, 0x34, 0x12]", "[0x12, 0x34, 0x56, 0x78]", "", "" }
}

#[lang = "u64"]
impl u64 {
    uint_impl! { u64, u64, 64, 18446744073709551615, 12, "0xaa00000000006e1", "0x6e10aa",
    "0x1234567890123456", "0x5634129078563412", "0x6a2c48091e6a2c48",
    "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
    "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]",
    "", ""}
}

#[lang = "u128"]
impl u128 {
    uint_impl! { u128, u128, 128, 340282366920938463463374607431768211455, 16,
    "0x13f40000000000000000000000004f76", "0x4f7613f4", "0x12345678901234567890123456789012",
    "0x12907856341290785634129078563412", "0x48091e6a2c48091e6a2c48091e6a2c48",
    "[0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, \
      0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
    "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, \
      0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12]",
     "", ""}
}

#[cfg(target_pointer_width = "16")]
#[lang = "usize"]
impl usize {
    uint_impl! { usize, u16, 16, 65535, 4, "0xa003", "0x3a", "0x1234", "0x3412", "0x2c48",
    "[0x34, 0x12]", "[0x12, 0x34]",
    usize_isize_to_xe_bytes_doc!(), usize_isize_from_xe_bytes_doc!() }
}
#[cfg(target_pointer_width = "32")]
#[lang = "usize"]
impl usize {
    uint_impl! { usize, u32, 32, 4294967295, 8, "0x10000b3", "0xb301", "0x12345678",
    "0x78563412", "0x1e6a2c48", "[0x78, 0x56, 0x34, 0x12]", "[0x12, 0x34, 0x56, 0x78]",
    usize_isize_to_xe_bytes_doc!(), usize_isize_from_xe_bytes_doc!() }
}

#[cfg(target_pointer_width = "64")]
#[lang = "usize"]
impl usize {
    uint_impl! { usize, u64, 64, 18446744073709551615, 12, "0xaa00000000006e1", "0x6e10aa",
    "0x1234567890123456", "0x5634129078563412", "0x6a2c48091e6a2c48",
    "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
     "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]",
    usize_isize_to_xe_bytes_doc!(), usize_isize_from_xe_bytes_doc!() }
}

/// A classification of floating point numbers.
///
/// This `enum` is used as the return type for [`f32::classify`] and [`f64::classify`]. See
/// their documentation for more.
///
/// # Examples
///
/// ```
/// use std::num::FpCategory;
///
/// let num = 12.4_f32;
/// let inf = f32::INFINITY;
/// let zero = 0f32;
/// let sub: f32 = 1.1754942e-38;
/// let nan = f32::NAN;
///
/// assert_eq!(num.classify(), FpCategory::Normal);
/// assert_eq!(inf.classify(), FpCategory::Infinite);
/// assert_eq!(zero.classify(), FpCategory::Zero);
/// assert_eq!(nan.classify(), FpCategory::Nan);
/// assert_eq!(sub.classify(), FpCategory::Subnormal);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum FpCategory {
    /// "Not a Number", often obtained by dividing by zero.
    #[stable(feature = "rust1", since = "1.0.0")]
    Nan,

    /// Positive or negative infinity.
    #[stable(feature = "rust1", since = "1.0.0")]
    Infinite,

    /// Positive or negative zero.
    #[stable(feature = "rust1", since = "1.0.0")]
    Zero,

    /// De-normalized floating point representation (less precise than `Normal`).
    #[stable(feature = "rust1", since = "1.0.0")]
    Subnormal,

    /// A regular floating point number.
    #[stable(feature = "rust1", since = "1.0.0")]
    Normal,
}

#[doc(hidden)]
trait FromStrRadixHelper: PartialOrd + Copy {
    fn min_value() -> Self;
    fn max_value() -> Self;
    fn from_u32(u: u32) -> Self;
    fn checked_mul(&self, other: u32) -> Option<Self>;
    fn checked_sub(&self, other: u32) -> Option<Self>;
    fn checked_add(&self, other: u32) -> Option<Self>;
}

macro_rules! from_str_radix_int_impl {
    ($($t:ty)*) => {$(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl FromStr for $t {
            type Err = ParseIntError;
            fn from_str(src: &str) -> Result<Self, ParseIntError> {
                from_str_radix(src, 10)
            }
        }
    )*}
}
from_str_radix_int_impl! { isize i8 i16 i32 i64 i128 usize u8 u16 u32 u64 u128 }

macro_rules! doit {
    ($($t:ty)*) => ($(impl FromStrRadixHelper for $t {
        #[inline]
        fn min_value() -> Self { Self::MIN }
        #[inline]
        fn max_value() -> Self { Self::MAX }
        #[inline]
        fn from_u32(u: u32) -> Self { u as Self }
        #[inline]
        fn checked_mul(&self, other: u32) -> Option<Self> {
            Self::checked_mul(*self, other as Self)
        }
        #[inline]
        fn checked_sub(&self, other: u32) -> Option<Self> {
            Self::checked_sub(*self, other as Self)
        }
        #[inline]
        fn checked_add(&self, other: u32) -> Option<Self> {
            Self::checked_add(*self, other as Self)
        }
    })*)
}
doit! { i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize }

fn from_str_radix<T: FromStrRadixHelper>(src: &str, radix: u32) -> Result<T, ParseIntError> {
    use self::IntErrorKind::*;
    use self::ParseIntError as PIE;

    assert!(
        (2..=36).contains(&radix),
        "from_str_radix_int: must lie in the range `[2, 36]` - found {}",
        radix
    );

    if src.is_empty() {
        return Err(PIE { kind: Empty });
    }

    let is_signed_ty = T::from_u32(0) > T::min_value();

    // all valid digits are ascii, so we will just iterate over the utf8 bytes
    // and cast them to chars. .to_digit() will safely return None for anything
    // other than a valid ascii digit for the given radix, including the first-byte
    // of multi-byte sequences
    let src = src.as_bytes();

    let (is_positive, digits) = match src[0] {
        b'+' | b'-' if src[1..].is_empty() => {
            return Err(PIE { kind: InvalidDigit });
        }
        b'+' => (true, &src[1..]),
        b'-' if is_signed_ty => (false, &src[1..]),
        _ => (true, src),
    };

    let mut result = T::from_u32(0);
    if is_positive {
        // The number is positive
        for &c in digits {
            let x = match (c as char).to_digit(radix) {
                Some(x) => x,
                None => return Err(PIE { kind: InvalidDigit }),
            };
            result = match result.checked_mul(radix) {
                Some(result) => result,
                None => return Err(PIE { kind: PosOverflow }),
            };
            result = match result.checked_add(x) {
                Some(result) => result,
                None => return Err(PIE { kind: PosOverflow }),
            };
        }
    } else {
        // The number is negative
        for &c in digits {
            let x = match (c as char).to_digit(radix) {
                Some(x) => x,
                None => return Err(PIE { kind: InvalidDigit }),
            };
            result = match result.checked_mul(radix) {
                Some(result) => result,
                None => return Err(PIE { kind: NegOverflow }),
            };
            result = match result.checked_sub(x) {
                Some(result) => result,
                None => return Err(PIE { kind: NegOverflow }),
            };
        }
    }
    Ok(result)
}
