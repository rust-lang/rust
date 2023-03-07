//! Numeric traits and functions for the built-in numeric types.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::ascii;
use crate::convert::TryInto;
use crate::intrinsics;
use crate::mem;
use crate::ops::{Add, Mul, Sub};
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

macro_rules! widening_impl {
    ($SelfT:ty, $WideT:ty, $BITS:literal, unsigned) => {
        /// Calculates the complete product `self * rhs` without the possibility to overflow.
        ///
        /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
        /// of the result as two separate values, in that order.
        ///
        /// If you also need to add a carry to the wide result, then you want
        /// [`Self::carrying_mul`] instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `u32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5u32.widening_mul(2), (10, 0));
        /// assert_eq!(1_000_000_000u32.widening_mul(10), (1410065408, 2));
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "const_bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn widening_mul(self, rhs: Self) -> (Self, Self) {
            // note: longer-term this should be done via an intrinsic,
            //   but for now we can deal without an impl for u128/i128
            // SAFETY: overflow will be contained within the wider types
            let wide = unsafe { (self as $WideT).unchecked_mul(rhs as $WideT) };
            (wide as $SelfT, (wide >> $BITS) as $SelfT)
        }

        /// Calculates the "full multiplication" `self * rhs + carry`
        /// without the possibility to overflow.
        ///
        /// This returns the low-order (wrapping) bits and the high-order (overflow) bits
        /// of the result as two separate values, in that order.
        ///
        /// Performs "long multiplication" which takes in an extra amount to add, and may return an
        /// additional amount of overflow. This allows for chaining together multiple
        /// multiplications to create "big integers" which represent larger values.
        ///
        /// If you don't need the `carry`, then you can use [`Self::widening_mul`] instead.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `u32` is used here.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(5u32.carrying_mul(2, 0), (10, 0));
        /// assert_eq!(5u32.carrying_mul(2, 10), (20, 0));
        /// assert_eq!(1_000_000_000u32.carrying_mul(10, 0), (1410065408, 2));
        /// assert_eq!(1_000_000_000u32.carrying_mul(10, 10), (1410065418, 2));
        #[doc = concat!("assert_eq!(",
            stringify!($SelfT), "::MAX.carrying_mul(", stringify!($SelfT), "::MAX, ", stringify!($SelfT), "::MAX), ",
            "(0, ", stringify!($SelfT), "::MAX));"
        )]
        /// ```
        ///
        /// This is the core operation needed for scalar multiplication when
        /// implementing it for wider-than-native types.
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// fn scalar_mul_eq(little_endian_digits: &mut Vec<u16>, multiplicand: u16) {
        ///     let mut carry = 0;
        ///     for d in little_endian_digits.iter_mut() {
        ///         (*d, carry) = d.carrying_mul(multiplicand, carry);
        ///     }
        ///     if carry != 0 {
        ///         little_endian_digits.push(carry);
        ///     }
        /// }
        ///
        /// let mut v = vec![10, 20];
        /// scalar_mul_eq(&mut v, 3);
        /// assert_eq!(v, [30, 60]);
        ///
        /// assert_eq!(0x87654321_u64 * 0xFEED, 0x86D3D159E38D);
        /// let mut v = vec![0x4321, 0x8765];
        /// scalar_mul_eq(&mut v, 0xFEED);
        /// assert_eq!(v, [0xE38D, 0xD159, 0x86D3]);
        /// ```
        ///
        /// If `carry` is zero, this is similar to [`overflowing_mul`](Self::overflowing_mul),
        /// except that it gives the value of the overflow instead of just whether one happened:
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// let r = u8::carrying_mul(7, 13, 0);
        /// assert_eq!((r.0, r.1 != 0), u8::overflowing_mul(7, 13));
        /// let r = u8::carrying_mul(13, 42, 0);
        /// assert_eq!((r.0, r.1 != 0), u8::overflowing_mul(13, 42));
        /// ```
        ///
        /// The value of the first field in the returned tuple matches what you'd get
        /// by combining the [`wrapping_mul`](Self::wrapping_mul) and
        /// [`wrapping_add`](Self::wrapping_add) methods:
        ///
        /// ```
        /// #![feature(bigint_helper_methods)]
        /// assert_eq!(
        ///     789_u16.carrying_mul(456, 123).0,
        ///     789_u16.wrapping_mul(456).wrapping_add(123),
        /// );
        /// ```
        #[unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[rustc_const_unstable(feature = "bigint_helper_methods", issue = "85532")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn carrying_mul(self, rhs: Self, carry: Self) -> (Self, Self) {
            // note: longer-term this should be done via an intrinsic,
            //   but for now we can deal without an impl for u128/i128
            // SAFETY: overflow will be contained within the wider types
            let wide = unsafe {
                (self as $WideT).unchecked_mul(rhs as $WideT).unchecked_add(carry as $WideT)
            };
            (wide as $SelfT, (wide >> $BITS) as $SelfT)
        }
    };
}

impl i8 {
    int_impl! {
        Self = i8,
        ActualT = i8,
        UnsignedT = u8,
        BITS = 8,
        BITS_MINUS_ONE = 7,
        Min = -128,
        Max = 127,
        rot = 2,
        rot_op = "-0x7e",
        rot_result = "0xa",
        swap_op = "0x12",
        swapped = "0x12",
        reversed = "0x48",
        le_bytes = "[0x12]",
        be_bytes = "[0x12]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
}

impl i16 {
    int_impl! {
        Self = i16,
        ActualT = i16,
        UnsignedT = u16,
        BITS = 16,
        BITS_MINUS_ONE = 15,
        Min = -32768,
        Max = 32767,
        rot = 4,
        rot_op = "-0x5ffd",
        rot_result = "0x3a",
        swap_op = "0x1234",
        swapped = "0x3412",
        reversed = "0x2c48",
        le_bytes = "[0x34, 0x12]",
        be_bytes = "[0x12, 0x34]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
}

impl i32 {
    int_impl! {
        Self = i32,
        ActualT = i32,
        UnsignedT = u32,
        BITS = 32,
        BITS_MINUS_ONE = 31,
        Min = -2147483648,
        Max = 2147483647,
        rot = 8,
        rot_op = "0x10000b3",
        rot_result = "0xb301",
        swap_op = "0x12345678",
        swapped = "0x78563412",
        reversed = "0x1e6a2c48",
        le_bytes = "[0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
}

impl i64 {
    int_impl! {
        Self = i64,
        ActualT = i64,
        UnsignedT = u64,
        BITS = 64,
        BITS_MINUS_ONE = 63,
        Min = -9223372036854775808,
        Max = 9223372036854775807,
        rot = 12,
        rot_op = "0xaa00000000006e1",
        rot_result = "0x6e10aa",
        swap_op = "0x1234567890123456",
        swapped = "0x5634129078563412",
        reversed = "0x6a2c48091e6a2c48",
        le_bytes = "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
}

impl i128 {
    int_impl! {
        Self = i128,
        ActualT = i128,
        UnsignedT = u128,
        BITS = 128,
        BITS_MINUS_ONE = 127,
        Min = -170141183460469231731687303715884105728,
        Max = 170141183460469231731687303715884105727,
        rot = 16,
        rot_op = "0x13f40000000000000000000000004f76",
        rot_result = "0x4f7613f4",
        swap_op = "0x12345678901234567890123456789012",
        swapped = "0x12907856341290785634129078563412",
        reversed = "0x48091e6a2c48091e6a2c48091e6a2c48",
        le_bytes = "[0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, \
            0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, \
            0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
}

#[cfg(target_pointer_width = "16")]
impl isize {
    int_impl! {
        Self = isize,
        ActualT = i16,
        UnsignedT = usize,
        BITS = 16,
        BITS_MINUS_ONE = 15,
        Min = -32768,
        Max = 32767,
        rot = 4,
        rot_op = "-0x5ffd",
        rot_result = "0x3a",
        swap_op = "0x1234",
        swapped = "0x3412",
        reversed = "0x2c48",
        le_bytes = "[0x34, 0x12]",
        be_bytes = "[0x12, 0x34]",
        to_xe_bytes_doc = usize_isize_to_xe_bytes_doc!(),
        from_xe_bytes_doc = usize_isize_from_xe_bytes_doc!(),
        bound_condition = " on 16-bit targets",
    }
}

#[cfg(target_pointer_width = "32")]
impl isize {
    int_impl! {
        Self = isize,
        ActualT = i32,
        UnsignedT = usize,
        BITS = 32,
        BITS_MINUS_ONE = 31,
        Min = -2147483648,
        Max = 2147483647,
        rot = 8,
        rot_op = "0x10000b3",
        rot_result = "0xb301",
        swap_op = "0x12345678",
        swapped = "0x78563412",
        reversed = "0x1e6a2c48",
        le_bytes = "[0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78]",
        to_xe_bytes_doc = usize_isize_to_xe_bytes_doc!(),
        from_xe_bytes_doc = usize_isize_from_xe_bytes_doc!(),
        bound_condition = " on 32-bit targets",
    }
}

#[cfg(target_pointer_width = "64")]
impl isize {
    int_impl! {
        Self = isize,
        ActualT = i64,
        UnsignedT = usize,
        BITS = 64,
        BITS_MINUS_ONE = 63,
        Min = -9223372036854775808,
        Max = 9223372036854775807,
        rot = 12,
        rot_op = "0xaa00000000006e1",
        rot_result = "0x6e10aa",
        swap_op = "0x1234567890123456",
        swapped = "0x5634129078563412",
        reversed = "0x6a2c48091e6a2c48",
        le_bytes = "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]",
        to_xe_bytes_doc = usize_isize_to_xe_bytes_doc!(),
        from_xe_bytes_doc = usize_isize_from_xe_bytes_doc!(),
        bound_condition = " on 64-bit targets",
    }
}

/// If 6th bit set ascii is upper case.
const ASCII_CASE_MASK: u8 = 0b0010_0000;

impl u8 {
    uint_impl! {
        Self = u8,
        ActualT = u8,
        SignedT = i8,
        NonZeroT = NonZeroU8,
        BITS = 8,
        MAX = 255,
        rot = 2,
        rot_op = "0x82",
        rot_result = "0xa",
        swap_op = "0x12",
        swapped = "0x12",
        reversed = "0x48",
        le_bytes = "[0x12]",
        be_bytes = "[0x12]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
    widening_impl! { u8, u16, 8, unsigned }

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
    #[must_use]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_u8_is_ascii", since = "1.43.0")]
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
    #[must_use = "to uppercase the value in-place, use `make_ascii_uppercase()`"]
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_ascii_methods_on_intrinsics", since = "1.52.0")]
    #[inline]
    pub const fn to_ascii_uppercase(&self) -> u8 {
        // Toggle the fifth bit if this is a lowercase letter
        *self ^ ((self.is_ascii_lowercase() as u8) * ASCII_CASE_MASK)
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
    #[must_use = "to lowercase the value in-place, use `make_ascii_lowercase()`"]
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
    /// let esc = b'\x1b';
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
    /// let esc = b'\x1b';
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
    /// let esc = b'\x1b';
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
    /// let esc = b'\x1b';
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
    /// let esc = b'\x1b';
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
        matches!(*self, b'0'..=b'9')
    }

    /// Checks if the value is an ASCII octal digit:
    /// U+0030 '0' ..= U+0037 '7'.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(is_ascii_octdigit)]
    ///
    /// let uppercase_a = b'A';
    /// let a = b'a';
    /// let zero = b'0';
    /// let seven = b'7';
    /// let nine = b'9';
    /// let percent = b'%';
    /// let lf = b'\n';
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
    #[rustc_const_unstable(feature = "is_ascii_octdigit", issue = "101288")]
    #[inline]
    pub const fn is_ascii_octdigit(&self) -> bool {
        matches!(*self, b'0'..=b'7')
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
    /// let esc = b'\x1b';
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
        matches!(*self, b'0'..=b'9' | b'A'..=b'F' | b'a'..=b'f')
    }

    /// Checks if the value is an ASCII punctuation character:
    ///
    /// - U+0021 ..= U+002F `! " # $ % & ' ( ) * + , - . /`, or
    /// - U+003A ..= U+0040 `: ; < = > ? @`, or
    /// - U+005B ..= U+0060 `` [ \ ] ^ _ ` ``, or
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
    /// let esc = b'\x1b';
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
    /// let esc = b'\x1b';
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
    /// let esc = b'\x1b';
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
    /// let esc = b'\x1b';
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
    #[must_use = "this returns the escaped byte as an iterator, \
                  without modifying the original"]
    #[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
    #[inline]
    pub fn escape_ascii(self) -> ascii::EscapeDefault {
        ascii::escape_default(self)
    }

    #[inline]
    pub(crate) const fn is_utf8_char_boundary(self) -> bool {
        // This is bit magic equivalent to: b < 128 || b >= 192
        (self as i8) >= -0x40
    }
}

impl u16 {
    uint_impl! {
        Self = u16,
        ActualT = u16,
        SignedT = i16,
        NonZeroT = NonZeroU16,
        BITS = 16,
        MAX = 65535,
        rot = 4,
        rot_op = "0xa003",
        rot_result = "0x3a",
        swap_op = "0x1234",
        swapped = "0x3412",
        reversed = "0x2c48",
        le_bytes = "[0x34, 0x12]",
        be_bytes = "[0x12, 0x34]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
    widening_impl! { u16, u32, 16, unsigned }

    /// Checks if the value is a Unicode surrogate code point, which are disallowed values for [`char`].
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(utf16_extra)]
    ///
    /// let low_non_surrogate = 0xA000u16;
    /// let low_surrogate = 0xD800u16;
    /// let high_surrogate = 0xDC00u16;
    /// let high_non_surrogate = 0xE000u16;
    ///
    /// assert!(!low_non_surrogate.is_utf16_surrogate());
    /// assert!(low_surrogate.is_utf16_surrogate());
    /// assert!(high_surrogate.is_utf16_surrogate());
    /// assert!(!high_non_surrogate.is_utf16_surrogate());
    /// ```
    #[must_use]
    #[unstable(feature = "utf16_extra", issue = "94919")]
    #[rustc_const_unstable(feature = "utf16_extra_const", issue = "94919")]
    #[inline]
    pub const fn is_utf16_surrogate(self) -> bool {
        matches!(self, 0xD800..=0xDFFF)
    }
}

impl u32 {
    uint_impl! {
        Self = u32,
        ActualT = u32,
        SignedT = i32,
        NonZeroT = NonZeroU32,
        BITS = 32,
        MAX = 4294967295,
        rot = 8,
        rot_op = "0x10000b3",
        rot_result = "0xb301",
        swap_op = "0x12345678",
        swapped = "0x78563412",
        reversed = "0x1e6a2c48",
        le_bytes = "[0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
    widening_impl! { u32, u64, 32, unsigned }
}

impl u64 {
    uint_impl! {
        Self = u64,
        ActualT = u64,
        SignedT = i64,
        NonZeroT = NonZeroU64,
        BITS = 64,
        MAX = 18446744073709551615,
        rot = 12,
        rot_op = "0xaa00000000006e1",
        rot_result = "0x6e10aa",
        swap_op = "0x1234567890123456",
        swapped = "0x5634129078563412",
        reversed = "0x6a2c48091e6a2c48",
        le_bytes = "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
    widening_impl! { u64, u128, 64, unsigned }
}

impl u128 {
    uint_impl! {
        Self = u128,
        ActualT = u128,
        SignedT = i128,
        NonZeroT = NonZeroU128,
        BITS = 128,
        MAX = 340282366920938463463374607431768211455,
        rot = 16,
        rot_op = "0x13f40000000000000000000000004f76",
        rot_result = "0x4f7613f4",
        swap_op = "0x12345678901234567890123456789012",
        swapped = "0x12907856341290785634129078563412",
        reversed = "0x48091e6a2c48091e6a2c48091e6a2c48",
        le_bytes = "[0x12, 0x90, 0x78, 0x56, 0x34, 0x12, 0x90, 0x78, \
            0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56, \
            0x78, 0x90, 0x12, 0x34, 0x56, 0x78, 0x90, 0x12]",
        to_xe_bytes_doc = "",
        from_xe_bytes_doc = "",
        bound_condition = "",
    }
}

#[cfg(target_pointer_width = "16")]
impl usize {
    uint_impl! {
        Self = usize,
        ActualT = u16,
        SignedT = isize,
        NonZeroT = NonZeroUsize,
        BITS = 16,
        MAX = 65535,
        rot = 4,
        rot_op = "0xa003",
        rot_result = "0x3a",
        swap_op = "0x1234",
        swapped = "0x3412",
        reversed = "0x2c48",
        le_bytes = "[0x34, 0x12]",
        be_bytes = "[0x12, 0x34]",
        to_xe_bytes_doc = usize_isize_to_xe_bytes_doc!(),
        from_xe_bytes_doc = usize_isize_from_xe_bytes_doc!(),
        bound_condition = " on 16-bit targets",
    }
    widening_impl! { usize, u32, 16, unsigned }
}

#[cfg(target_pointer_width = "32")]
impl usize {
    uint_impl! {
        Self = usize,
        ActualT = u32,
        SignedT = isize,
        NonZeroT = NonZeroUsize,
        BITS = 32,
        MAX = 4294967295,
        rot = 8,
        rot_op = "0x10000b3",
        rot_result = "0xb301",
        swap_op = "0x12345678",
        swapped = "0x78563412",
        reversed = "0x1e6a2c48",
        le_bytes = "[0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78]",
        to_xe_bytes_doc = usize_isize_to_xe_bytes_doc!(),
        from_xe_bytes_doc = usize_isize_from_xe_bytes_doc!(),
        bound_condition = " on 32-bit targets",
    }
    widening_impl! { usize, u64, 32, unsigned }
}

#[cfg(target_pointer_width = "64")]
impl usize {
    uint_impl! {
        Self = usize,
        ActualT = u64,
        SignedT = isize,
        NonZeroT = NonZeroUsize,
        BITS = 64,
        MAX = 18446744073709551615,
        rot = 12,
        rot_op = "0xaa00000000006e1",
        rot_result = "0x6e10aa",
        swap_op = "0x1234567890123456",
        swapped = "0x5634129078563412",
        reversed = "0x6a2c48091e6a2c48",
        le_bytes = "[0x56, 0x34, 0x12, 0x90, 0x78, 0x56, 0x34, 0x12]",
        be_bytes = "[0x12, 0x34, 0x56, 0x78, 0x90, 0x12, 0x34, 0x56]",
        to_xe_bytes_doc = usize_isize_to_xe_bytes_doc!(),
        from_xe_bytes_doc = usize_isize_from_xe_bytes_doc!(),
        bound_condition = " on 64-bit targets",
    }
    widening_impl! { usize, u128, 64, unsigned }
}

impl usize {
    /// Returns an `usize` where every byte is equal to `x`.
    #[inline]
    pub(crate) const fn repeat_u8(x: u8) -> usize {
        usize::from_ne_bytes([x; mem::size_of::<usize>()])
    }

    /// Returns an `usize` where every byte pair is equal to `x`.
    #[inline]
    pub(crate) const fn repeat_u16(x: u16) -> usize {
        let mut r = 0usize;
        let mut i = 0;
        while i < mem::size_of::<usize>() {
            // Use `wrapping_shl` to make it work on targets with 16-bit `usize`
            r = r.wrapping_shl(16) | (x as usize);
            i += 2;
        }
        r
    }
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
/// assert_eq!(sub.classify(), FpCategory::Subnormal);
/// assert_eq!(nan.classify(), FpCategory::Nan);
/// ```
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum FpCategory {
    /// NaN (not a number): this value results from calculations like `(-1.0).sqrt()`.
    ///
    /// See [the documentation for `f32`](f32) for more information on the unusual properties
    /// of NaN.
    #[stable(feature = "rust1", since = "1.0.0")]
    Nan,

    /// Positive or negative infinity, which often results from dividing a nonzero number
    /// by zero.
    #[stable(feature = "rust1", since = "1.0.0")]
    Infinite,

    /// Positive or negative zero.
    ///
    /// See [the documentation for `f32`](f32) for more information on the signedness of zeroes.
    #[stable(feature = "rust1", since = "1.0.0")]
    Zero,

    /// “Subnormal” or “denormal” floating point representation (less precise, relative to
    /// their magnitude, than [`Normal`]).
    ///
    /// Subnormal numbers are larger in magnitude than [`Zero`] but smaller in magnitude than all
    /// [`Normal`] numbers.
    ///
    /// [`Normal`]: Self::Normal
    /// [`Zero`]: Self::Zero
    #[stable(feature = "rust1", since = "1.0.0")]
    Subnormal,

    /// A regular floating point number, not any of the exceptional categories.
    ///
    /// The smallest positive normal numbers are [`f32::MIN_POSITIVE`] and [`f64::MIN_POSITIVE`],
    /// and the largest positive normal numbers are [`f32::MAX`] and [`f64::MAX`]. (Unlike signed
    /// integers, floating point numbers are symmetric in their range, so negating any of these
    /// constants will produce their negative counterpart.)
    #[stable(feature = "rust1", since = "1.0.0")]
    Normal,
}

#[doc(hidden)]
trait FromStrRadixHelper:
    PartialOrd + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    const MIN: Self;
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

macro_rules! impl_helper_for {
    ($($t:ty)*) => ($(impl FromStrRadixHelper for $t {
        const MIN: Self = Self::MIN;
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
impl_helper_for! { i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize }

/// Determines if a string of text of that length of that radix could be guaranteed to be
/// stored in the given type T.
/// Note that if the radix is known to the compiler, it is just the check of digits.len that
/// is done at runtime.
#[doc(hidden)]
#[inline(always)]
#[unstable(issue = "none", feature = "std_internals")]
pub fn can_not_overflow<T>(radix: u32, is_signed_ty: bool, digits: &[u8]) -> bool {
    radix <= 16 && digits.len() <= mem::size_of::<T>() * 2 - is_signed_ty as usize
}

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

    let is_signed_ty = T::from_u32(0) > T::MIN;

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

    if can_not_overflow::<T>(radix, is_signed_ty, digits) {
        // If the len of the str is short compared to the range of the type
        // we are parsing into, then we can be certain that an overflow will not occur.
        // This bound is when `radix.pow(digits.len()) - 1 <= T::MAX` but the condition
        // above is a faster (conservative) approximation of this.
        //
        // Consider radix 16 as it has the highest information density per digit and will thus overflow the earliest:
        // `u8::MAX` is `ff` - any str of len 2 is guaranteed to not overflow.
        // `i8::MAX` is `7f` - only a str of len 1 is guaranteed to not overflow.
        macro_rules! run_unchecked_loop {
            ($unchecked_additive_op:expr) => {
                for &c in digits {
                    result = result * T::from_u32(radix);
                    let x = (c as char).to_digit(radix).ok_or(PIE { kind: InvalidDigit })?;
                    result = $unchecked_additive_op(result, T::from_u32(x));
                }
            };
        }
        if is_positive {
            run_unchecked_loop!(<T as core::ops::Add>::add)
        } else {
            run_unchecked_loop!(<T as core::ops::Sub>::sub)
        };
    } else {
        macro_rules! run_checked_loop {
            ($checked_additive_op:ident, $overflow_err:expr) => {
                for &c in digits {
                    // When `radix` is passed in as a literal, rather than doing a slow `imul`
                    // the compiler can use shifts if `radix` can be expressed as a
                    // sum of powers of 2 (x*10 can be written as x*8 + x*2).
                    // When the compiler can't use these optimisations,
                    // the latency of the multiplication can be hidden by issuing it
                    // before the result is needed to improve performance on
                    // modern out-of-order CPU as multiplication here is slower
                    // than the other instructions, we can get the end result faster
                    // doing multiplication first and let the CPU spends other cycles
                    // doing other computation and get multiplication result later.
                    let mul = result.checked_mul(radix);
                    let x = (c as char).to_digit(radix).ok_or(PIE { kind: InvalidDigit })?;
                    result = mul.ok_or_else($overflow_err)?;
                    result = T::$checked_additive_op(&result, x).ok_or_else($overflow_err)?;
                }
            };
        }
        if is_positive {
            run_checked_loop!(checked_add, || PIE { kind: PosOverflow })
        } else {
            run_checked_loop!(checked_sub, || PIE { kind: NegOverflow })
        };
    }
    Ok(result)
}
