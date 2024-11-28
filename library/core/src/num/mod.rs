//! Numeric traits and functions for the built-in numeric types.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::panic::const_panic;
use crate::str::FromStr;
use crate::ub_checks::assert_unsafe_precondition;
use crate::{ascii, intrinsics, mem};

// FIXME(const-hack): Used because the `?` operator is not allowed in a const context.
macro_rules! try_opt {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => return None,
        }
    };
}

// Use this when the generated code should differ between signed and unsigned types.
macro_rules! sign_dependent_expr {
    (signed ? if signed { $signed_case:expr } if unsigned { $unsigned_case:expr } ) => {
        $signed_case
    };
    (unsigned ? if signed { $signed_case:expr } if unsigned { $unsigned_case:expr } ) => {
        $unsigned_case
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
mod int_sqrt;
mod nonzero;
mod overflow_panic;
mod saturating;
mod wrapping;

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(not(no_fp_fmt_parse))]
pub use dec2flt::ParseFloatError;
#[stable(feature = "int_error_matching", since = "1.55.0")]
pub use error::IntErrorKind;
#[stable(feature = "rust1", since = "1.0.0")]
pub use error::ParseIntError;
#[stable(feature = "try_from", since = "1.34.0")]
pub use error::TryFromIntError;
#[stable(feature = "generic_nonzero", since = "1.79.0")]
pub use nonzero::NonZero;
#[unstable(
    feature = "nonzero_internals",
    reason = "implementation detail which may disappear or be replaced at any time",
    issue = "none"
)]
pub use nonzero::ZeroablePrimitive;
#[stable(feature = "signed_nonzero", since = "1.34.0")]
pub use nonzero::{NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize};
#[stable(feature = "nonzero", since = "1.28.0")]
pub use nonzero::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize};
#[stable(feature = "saturating_int_impl", since = "1.74.0")]
pub use saturating::Saturating;
#[stable(feature = "rust1", since = "1.0.0")]
pub use wrapping::Wrapping;

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

macro_rules! midpoint_impl {
    ($SelfT:ty, unsigned) => {
        /// Calculates the middle point of `self` and `rhs`.
        ///
        /// `midpoint(a, b)` is `(a + b) >> 1` as if it were performed in a
        /// sufficiently-large signed integral type. This implies that the result is
        /// always rounded towards negative infinity and that no overflow will ever occur.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(num_midpoint)]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(4), 2);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".midpoint(4), 2);")]
        /// ```
        #[unstable(feature = "num_midpoint", issue = "110840")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn midpoint(self, rhs: $SelfT) -> $SelfT {
            // Use the well known branchless algorithm from Hacker's Delight to compute
            // `(a + b) / 2` without overflowing: `((a ^ b) >> 1) + (a & b)`.
            ((self ^ rhs) >> 1) + (self & rhs)
        }
    };
    ($SelfT:ty, signed) => {
        /// Calculates the middle point of `self` and `rhs`.
        ///
        /// `midpoint(a, b)` is `(a + b) / 2` as if it were performed in a
        /// sufficiently-large signed integral type. This implies that the result is
        /// always rounded towards zero and that no overflow will ever occur.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(num_midpoint)]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(4), 2);")]
        #[doc = concat!("assert_eq!((-1", stringify!($SelfT), ").midpoint(2), 0);")]
        #[doc = concat!("assert_eq!((-7", stringify!($SelfT), ").midpoint(0), -3);")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(-7), -3);")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(7), 3);")]
        /// ```
        #[unstable(feature = "num_midpoint", issue = "110840")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn midpoint(self, rhs: Self) -> Self {
            // Use the well known branchless algorithm from Hacker's Delight to compute
            // `(a + b) / 2` without overflowing: `((a ^ b) >> 1) + (a & b)`.
            let t = ((self ^ rhs) >> 1) + (self & rhs);
            // Except that it fails for integers whose sum is an odd negative number as
            // their floor is one less than their average. So we adjust the result.
            t + (if t < 0 { 1 } else { 0 } & (self ^ rhs))
        }
    };
    ($SelfT:ty, $WideT:ty, unsigned) => {
        /// Calculates the middle point of `self` and `rhs`.
        ///
        /// `midpoint(a, b)` is `(a + b) >> 1` as if it were performed in a
        /// sufficiently-large signed integral type. This implies that the result is
        /// always rounded towards negative infinity and that no overflow will ever occur.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(num_midpoint)]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(4), 2);")]
        #[doc = concat!("assert_eq!(1", stringify!($SelfT), ".midpoint(4), 2);")]
        /// ```
        #[unstable(feature = "num_midpoint", issue = "110840")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn midpoint(self, rhs: $SelfT) -> $SelfT {
            ((self as $WideT + rhs as $WideT) / 2) as $SelfT
        }
    };
    ($SelfT:ty, $WideT:ty, signed) => {
        /// Calculates the middle point of `self` and `rhs`.
        ///
        /// `midpoint(a, b)` is `(a + b) / 2` as if it were performed in a
        /// sufficiently-large signed integral type. This implies that the result is
        /// always rounded towards zero and that no overflow will ever occur.
        ///
        /// # Examples
        ///
        /// ```
        /// #![feature(num_midpoint)]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(4), 2);")]
        #[doc = concat!("assert_eq!((-1", stringify!($SelfT), ").midpoint(2), 0);")]
        #[doc = concat!("assert_eq!((-7", stringify!($SelfT), ").midpoint(0), -3);")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(-7), -3);")]
        #[doc = concat!("assert_eq!(0", stringify!($SelfT), ".midpoint(7), 3);")]
        /// ```
        #[unstable(feature = "num_midpoint", issue = "110840")]
        #[must_use = "this returns the result of the operation, \
                      without modifying the original"]
        #[inline]
        pub const fn midpoint(self, rhs: $SelfT) -> $SelfT {
            ((self as $WideT + rhs as $WideT) / 2) as $SelfT
        }
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
    midpoint_impl! { i8, i16, signed }
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
    midpoint_impl! { i16, i32, signed }
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
    midpoint_impl! { i32, i64, signed }
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
    midpoint_impl! { i64, signed }
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
    midpoint_impl! { i128, signed }
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
    midpoint_impl! { isize, i32, signed }
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
    midpoint_impl! { isize, i64, signed }
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
    midpoint_impl! { isize, signed }
}

/// If the bit selected by this mask is set, ascii is lower case.
const ASCII_CASE_MASK: u8 = 0b0010_0000;

impl u8 {
    uint_impl! {
        Self = u8,
        ActualT = u8,
        SignedT = i8,
        BITS = 8,
        BITS_MINUS_ONE = 7,
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
    midpoint_impl! { u8, u16, unsigned }

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
        *self <= 127
    }

    /// If the value of this byte is within the ASCII range, returns it as an
    /// [ASCII character](ascii::Char).  Otherwise, returns `None`.
    #[must_use]
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[inline]
    pub const fn as_ascii(&self) -> Option<ascii::Char> {
        ascii::Char::from_u8(*self)
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
        // Toggle the 6th bit if this is a lowercase letter
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
        // Set the 6th bit if this is an uppercase letter
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
    #[rustc_const_stable(feature = "const_make_ascii", since = "CURRENT_RUSTC_VERSION")]
    #[inline]
    pub const fn make_ascii_uppercase(&mut self) {
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
    #[rustc_const_stable(feature = "const_make_ascii", since = "CURRENT_RUSTC_VERSION")]
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
        matches!(*self, b'0'..=b'9') | matches!(*self, b'A'..=b'Z') | matches!(*self, b'a'..=b'z')
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
        matches!(*self, b'0'..=b'9') | matches!(*self, b'A'..=b'F') | matches!(*self, b'a'..=b'f')
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
        matches!(*self, b'!'..=b'/')
            | matches!(*self, b':'..=b'@')
            | matches!(*self, b'['..=b'`')
            | matches!(*self, b'{'..=b'~')
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
        BITS = 16,
        BITS_MINUS_ONE = 15,
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
    midpoint_impl! { u16, u32, unsigned }

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
        BITS = 32,
        BITS_MINUS_ONE = 31,
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
    midpoint_impl! { u32, u64, unsigned }
}

impl u64 {
    uint_impl! {
        Self = u64,
        ActualT = u64,
        SignedT = i64,
        BITS = 64,
        BITS_MINUS_ONE = 63,
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
    midpoint_impl! { u64, u128, unsigned }
}

impl u128 {
    uint_impl! {
        Self = u128,
        ActualT = u128,
        SignedT = i128,
        BITS = 128,
        BITS_MINUS_ONE = 127,
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
    midpoint_impl! { u128, unsigned }
}

#[cfg(target_pointer_width = "16")]
impl usize {
    uint_impl! {
        Self = usize,
        ActualT = u16,
        SignedT = isize,
        BITS = 16,
        BITS_MINUS_ONE = 15,
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
    midpoint_impl! { usize, u32, unsigned }
}

#[cfg(target_pointer_width = "32")]
impl usize {
    uint_impl! {
        Self = usize,
        ActualT = u32,
        SignedT = isize,
        BITS = 32,
        BITS_MINUS_ONE = 31,
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
    midpoint_impl! { usize, u64, unsigned }
}

#[cfg(target_pointer_width = "64")]
impl usize {
    uint_impl! {
        Self = usize,
        ActualT = u64,
        SignedT = isize,
        BITS = 64,
        BITS_MINUS_ONE = 63,
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
    midpoint_impl! { usize, u128, unsigned }
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

macro_rules! from_str_radix_int_impl {
    ($($t:ty)*) => {$(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl FromStr for $t {
            type Err = ParseIntError;
            #[inline]
            fn from_str(src: &str) -> Result<Self, ParseIntError> {
                <$t>::from_str_radix(src, 10)
            }
        }
    )*}
}
from_str_radix_int_impl! { isize i8 i16 i32 i64 i128 usize u8 u16 u32 u64 u128 }

/// Determines if a string of text of that length of that radix could be guaranteed to be
/// stored in the given type T.
/// Note that if the radix is known to the compiler, it is just the check of digits.len that
/// is done at runtime.
#[doc(hidden)]
#[inline(always)]
#[unstable(issue = "none", feature = "std_internals")]
#[cfg_attr(bootstrap, rustc_const_stable(feature = "const_int_from_str", since = "1.82.0"))]
pub const fn can_not_overflow<T>(radix: u32, is_signed_ty: bool, digits: &[u8]) -> bool {
    radix <= 16 && digits.len() <= mem::size_of::<T>() * 2 - is_signed_ty as usize
}

#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[cold]
#[track_caller]
const fn from_str_radix_panic(radix: u32) -> ! {
    const_panic!(
        "from_str_radix_int: must lie in the range `[2, 36]`",
        "from_str_radix_int: must lie in the range `[2, 36]` - found {radix}",
        radix: u32 = radix,
    )
}

macro_rules! from_str_radix {
    ($signedness:ident $($int_ty:ty)+) => {$(
        impl $int_ty {
            /// Converts a string slice in a given base to an integer.
            ///
            /// The string is expected to be an optional
            #[doc = sign_dependent_expr!{
                $signedness ?
                if signed {
                    " `+` or `-` "
                }
                if unsigned {
                    " `+` "
                }
            }]
            /// sign followed by only digits. Leading and trailing non-digit characters (including
            /// whitespace) represent an error. Underscores (which are accepted in rust literals)
            /// also represent an error.
            ///
            /// Digits are a subset of these characters, depending on `radix`:
            /// * `0-9`
            /// * `a-z`
            /// * `A-Z`
            ///
            /// # Panics
            ///
            /// This function panics if `radix` is not in the range from 2 to 36.
            ///
            /// # Examples
            ///
            /// Basic usage:
            /// ```
            #[doc = concat!("assert_eq!(", stringify!($int_ty), "::from_str_radix(\"A\", 16), Ok(10));")]
            /// ```
            /// Trailing space returns error:
            /// ```
            #[doc = concat!("assert!(", stringify!($int_ty), "::from_str_radix(\"1 \", 10).is_err());")]
            /// ```
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_int_from_str", since = "1.82.0")]
            #[inline]
            pub const fn from_str_radix(src: &str, radix: u32) -> Result<$int_ty, ParseIntError> {
                use self::IntErrorKind::*;
                use self::ParseIntError as PIE;

                if 2 > radix || radix > 36 {
                    from_str_radix_panic(radix);
                }

                if src.is_empty() {
                    return Err(PIE { kind: Empty });
                }

                #[allow(unused_comparisons)]
                let is_signed_ty = 0 > <$int_ty>::MIN;

                // all valid digits are ascii, so we will just iterate over the utf8 bytes
                // and cast them to chars. .to_digit() will safely return None for anything
                // other than a valid ascii digit for the given radix, including the first-byte
                // of multi-byte sequences
                let src = src.as_bytes();

                let (is_positive, mut digits) = match src {
                    [b'+' | b'-'] => {
                        return Err(PIE { kind: InvalidDigit });
                    }
                    [b'+', rest @ ..] => (true, rest),
                    [b'-', rest @ ..] if is_signed_ty => (false, rest),
                    _ => (true, src),
                };

                let mut result = 0;

                macro_rules! unwrap_or_PIE {
                    ($option:expr, $kind:ident) => {
                        match $option {
                            Some(value) => value,
                            None => return Err(PIE { kind: $kind }),
                        }
                    };
                }

                if can_not_overflow::<$int_ty>(radix, is_signed_ty, digits) {
                    // If the len of the str is short compared to the range of the type
                    // we are parsing into, then we can be certain that an overflow will not occur.
                    // This bound is when `radix.pow(digits.len()) - 1 <= T::MAX` but the condition
                    // above is a faster (conservative) approximation of this.
                    //
                    // Consider radix 16 as it has the highest information density per digit and will thus overflow the earliest:
                    // `u8::MAX` is `ff` - any str of len 2 is guaranteed to not overflow.
                    // `i8::MAX` is `7f` - only a str of len 1 is guaranteed to not overflow.
                    macro_rules! run_unchecked_loop {
                        ($unchecked_additive_op:tt) => {{
                            while let [c, rest @ ..] = digits {
                                result = result * (radix as $int_ty);
                                let x = unwrap_or_PIE!((*c as char).to_digit(radix), InvalidDigit);
                                result = result $unchecked_additive_op (x as $int_ty);
                                digits = rest;
                            }
                        }};
                    }
                    if is_positive {
                        run_unchecked_loop!(+)
                    } else {
                        run_unchecked_loop!(-)
                    };
                } else {
                    macro_rules! run_checked_loop {
                        ($checked_additive_op:ident, $overflow_err:ident) => {{
                            while let [c, rest @ ..] = digits {
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
                                let mul = result.checked_mul(radix as $int_ty);
                                let x = unwrap_or_PIE!((*c as char).to_digit(radix), InvalidDigit) as $int_ty;
                                result = unwrap_or_PIE!(mul, $overflow_err);
                                result = unwrap_or_PIE!(<$int_ty>::$checked_additive_op(result, x), $overflow_err);
                                digits = rest;
                            }
                        }};
                    }
                    if is_positive {
                        run_checked_loop!(checked_add, PosOverflow)
                    } else {
                        run_checked_loop!(checked_sub, NegOverflow)
                    };
                }
                Ok(result)
            }
        }
    )+}
}

from_str_radix! { unsigned u8 u16 u32 u64 u128 }
from_str_radix! { signed i8 i16 i32 i64 i128 }

// Re-use the relevant implementation of from_str_radix for isize and usize to avoid outputting two
// identical functions.
macro_rules! from_str_radix_size_impl {
    ($($signedness:ident $t:ident $size:ty),*) => {$(
    impl $size {
        /// Converts a string slice in a given base to an integer.
        ///
        /// The string is expected to be an optional
        #[doc = sign_dependent_expr!{
            $signedness ?
            if signed {
                " `+` or `-` "
            }
            if unsigned {
                " `+` "
            }
        }]
        /// sign followed by only digits. Leading and trailing non-digit characters (including
        /// whitespace) represent an error. Underscores (which are accepted in rust literals)
        /// also represent an error.
        ///
        /// Digits are a subset of these characters, depending on `radix`:
        /// * `0-9`
        /// * `a-z`
        /// * `A-Z`
        ///
        /// # Panics
        ///
        /// This function panics if `radix` is not in the range from 2 to 36.
        ///
        /// # Examples
        ///
        /// Basic usage:
        /// ```
        #[doc = concat!("assert_eq!(", stringify!($size), "::from_str_radix(\"A\", 16), Ok(10));")]
        /// ```
        /// Trailing space returns error:
        /// ```
        #[doc = concat!("assert!(", stringify!($size), "::from_str_radix(\"1 \", 10).is_err());")]
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_from_str", since = "1.82.0")]
        #[inline]
        pub const fn from_str_radix(src: &str, radix: u32) -> Result<$size, ParseIntError> {
            match <$t>::from_str_radix(src, radix) {
                Ok(x) => Ok(x as $size),
                Err(e) => Err(e),
            }
        }
    })*}
}

#[cfg(target_pointer_width = "16")]
from_str_radix_size_impl! { signed i16 isize, unsigned u16 usize }
#[cfg(target_pointer_width = "32")]
from_str_radix_size_impl! { signed i32 isize, unsigned u32 usize }
#[cfg(target_pointer_width = "64")]
from_str_radix_size_impl! { signed i64 isize, unsigned u64 usize }
