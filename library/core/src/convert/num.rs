use crate::num::TryFromIntError;

mod private {
    /// This trait being unreachable from outside the crate
    /// prevents other implementations of the `FloatToInt` trait,
    /// which allows potentially adding more trait methods after the trait is `#[stable]`.
    #[unstable(feature = "convert_float_to_int", issue = "67057")]
    pub trait Sealed {}
}

/// Supporting trait for inherent methods of `f32` and `f64` such as `to_int_unchecked`.
/// Typically doesn’t need to be used directly.
#[unstable(feature = "convert_float_to_int", issue = "67057")]
pub trait FloatToInt<Int>: private::Sealed + Sized {
    #[unstable(feature = "convert_float_to_int", issue = "67057")]
    #[doc(hidden)]
    unsafe fn to_int_unchecked(self) -> Int;
}

macro_rules! impl_float_to_int {
    ($Float:ty => $($Int:ty),+) => {
        #[unstable(feature = "convert_float_to_int", issue = "67057")]
        impl private::Sealed for $Float {}
        $(
            #[unstable(feature = "convert_float_to_int", issue = "67057")]
            impl FloatToInt<$Int> for $Float {
                #[inline]
                unsafe fn to_int_unchecked(self) -> $Int {
                    // SAFETY: the safety contract must be upheld by the caller.
                    unsafe { crate::intrinsics::float_to_int_unchecked(self) }
                }
            }
        )+
    }
}

impl_float_to_int!(f16 => u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
impl_float_to_int!(f32 => u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
impl_float_to_int!(f64 => u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
impl_float_to_int!(f128 => u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

// Conversion traits for primitive integer and float types
// Conversions T -> T are covered by a blanket impl and therefore excluded
// Some conversions from and to usize/isize are not implemented due to portability concerns
macro_rules! impl_from {
    (bool => $Int:ty $(,)?) => {
        impl_from!(
            bool => $Int,
            #[stable(feature = "from_bool", since = "1.28.0")],
            concat!(
                "Converts a [`bool`] to [`", stringify!($Int), "`] losslessly.\n",
                "The resulting value is `0` for `false` and `1` for `true` values.\n",
                "\n",
                "# Examples\n",
                "\n",
                "```\n",
                "assert_eq!(", stringify!($Int), "::from(true), 1);\n",
                "assert_eq!(", stringify!($Int), "::from(false), 0);\n",
                "```\n",
            ),
        );
    };
    ($Small:ty => $Large:ty, #[$attr:meta] $(,)?) => {
        impl_from!(
            $Small => $Large,
            #[$attr],
            concat!("Converts [`", stringify!($Small), "`] to [`", stringify!($Large), "`] losslessly."),
        );
    };
    ($Small:ty => $Large:ty, #[$attr:meta], $doc:expr $(,)?) => {
        #[$attr]
        impl From<$Small> for $Large {
            // Rustdocs on the impl block show a "[+] show undocumented items" toggle.
            // Rustdocs on functions do not.
            #[doc = $doc]
            #[inline(always)]
            fn from(small: $Small) -> Self {
                small as Self
            }
        }
    };
}

// boolean -> integer
impl_from!(bool => u8);
impl_from!(bool => u16);
impl_from!(bool => u32);
impl_from!(bool => u64);
impl_from!(bool => u128);
impl_from!(bool => usize);
impl_from!(bool => i8);
impl_from!(bool => i16);
impl_from!(bool => i32);
impl_from!(bool => i64);
impl_from!(bool => i128);
impl_from!(bool => isize);

// unsigned integer -> unsigned integer
impl_from!(u8 => u16, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u8 => u32, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u8 => u64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u8 => u128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(u8 => usize, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u16 => u32, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u16 => u64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u16 => u128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(u32 => u64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u32 => u128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(u64 => u128, #[stable(feature = "i128", since = "1.26.0")]);

// signed integer -> signed integer
impl_from!(i8 => i16, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(i8 => i32, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(i8 => i64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(i8 => i128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(i8 => isize, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(i16 => i32, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(i16 => i64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(i16 => i128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(i32 => i64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(i32 => i128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(i64 => i128, #[stable(feature = "i128", since = "1.26.0")]);

// unsigned integer -> signed integer
impl_from!(u8 => i16, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u8 => i32, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u8 => i64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u8 => i128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(u16 => i32, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u16 => i64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u16 => i128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(u32 => i64, #[stable(feature = "lossless_int_conv", since = "1.5.0")]);
impl_from!(u32 => i128, #[stable(feature = "i128", since = "1.26.0")]);
impl_from!(u64 => i128, #[stable(feature = "i128", since = "1.26.0")]);

// The C99 standard defines bounds on INTPTR_MIN, INTPTR_MAX, and UINTPTR_MAX
// which imply that pointer-sized integers must be at least 16 bits:
// https://port70.net/~nsz/c/c99/n1256.html#7.18.2.4
impl_from!(u16 => usize, #[stable(feature = "lossless_iusize_conv", since = "1.26.0")]);
impl_from!(u8 => isize, #[stable(feature = "lossless_iusize_conv", since = "1.26.0")]);
impl_from!(i16 => isize, #[stable(feature = "lossless_iusize_conv", since = "1.26.0")]);

// RISC-V defines the possibility of a 128-bit address space (RV128).

// CHERI proposes 128-bit “capabilities”. Unclear if this would be relevant to usize/isize.
// https://www.cl.cam.ac.uk/research/security/ctsrd/pdfs/20171017a-cheri-poster.pdf
// https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-951.pdf

// Note: integers can only be represented with full precision in a float if
// they fit in the significand, which is 24 bits in f32 and 53 bits in f64.
// Lossy float conversions are not implemented at this time.

// signed integer -> float
impl_from!(i8 => f32, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(i8 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(i16 => f32, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(i16 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(i32 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);

// unsigned integer -> float
impl_from!(u8 => f32, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(u8 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(u16 => f32, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(u16 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(u32 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);

// float -> float
// FIXME(f16_f128): adding additional `From<{float}>` impls to `f32` breaks inference. See
// <https://github.com/rust-lang/rust/issues/123831>
impl_from!(f16 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(f16 => f128, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(f32 => f64, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(f32 => f128, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);
impl_from!(f64 => f128, #[stable(feature = "lossless_float_conv", since = "1.6.0")]);

macro_rules! impl_float_from_bool {
    ($float:ty) => {
        #[stable(feature = "float_from_bool", since = "1.68.0")]
        impl From<bool> for $float {
            #[doc = concat!("Converts a [`bool`] to [`", stringify!($float),"`] losslessly.")]
            /// The resulting value is positive `0.0` for `false` and `1.0` for `true` values.
            ///
            /// # Examples
            /// ```
            #[doc = concat!("let x: ", stringify!($float)," = false.into();")]
            /// assert_eq!(x, 0.0);
            /// assert!(x.is_sign_positive());
            ///
            #[doc = concat!("let y: ", stringify!($float)," = true.into();")]
            /// assert_eq!(y, 1.0);
            /// ```
            #[inline]
            fn from(small: bool) -> Self {
                small as u8 as Self
            }
        }
    };
}

// boolean -> float
impl_float_from_bool!(f32);
impl_float_from_bool!(f64);

// no possible bounds violation
macro_rules! impl_try_from_unbounded {
    ($source:ty => $($target:ty),+) => {$(
        #[stable(feature = "try_from", since = "1.34.0")]
        impl TryFrom<$source> for $target {
            type Error = TryFromIntError;

            /// Tries to create the target number type from a source
            /// number type. This returns an error if the source value
            /// is outside of the range of the target type.
            #[inline]
            fn try_from(value: $source) -> Result<Self, Self::Error> {
                Ok(value as Self)
            }
        }
    )*}
}

// only negative bounds
macro_rules! impl_try_from_lower_bounded {
    ($source:ty => $($target:ty),+) => {$(
        #[stable(feature = "try_from", since = "1.34.0")]
        impl TryFrom<$source> for $target {
            type Error = TryFromIntError;

            /// Tries to create the target number type from a source
            /// number type. This returns an error if the source value
            /// is outside of the range of the target type.
            #[inline]
            fn try_from(u: $source) -> Result<Self, Self::Error> {
                if u >= 0 {
                    Ok(u as Self)
                } else {
                    Err(TryFromIntError(()))
                }
            }
        }
    )*}
}

// unsigned to signed (only positive bound)
macro_rules! impl_try_from_upper_bounded {
    ($source:ty => $($target:ty),+) => {$(
        #[stable(feature = "try_from", since = "1.34.0")]
        impl TryFrom<$source> for $target {
            type Error = TryFromIntError;

            /// Tries to create the target number type from a source
            /// number type. This returns an error if the source value
            /// is outside of the range of the target type.
            #[inline]
            fn try_from(u: $source) -> Result<Self, Self::Error> {
                if u > (Self::MAX as $source) {
                    Err(TryFromIntError(()))
                } else {
                    Ok(u as Self)
                }
            }
        }
    )*}
}

// all other cases
macro_rules! impl_try_from_both_bounded {
    ($source:ty => $($target:ty),+) => {$(
        #[stable(feature = "try_from", since = "1.34.0")]
        impl TryFrom<$source> for $target {
            type Error = TryFromIntError;

            /// Tries to create the target number type from a source
            /// number type. This returns an error if the source value
            /// is outside of the range of the target type.
            #[inline]
            fn try_from(u: $source) -> Result<Self, Self::Error> {
                let min = Self::MIN as $source;
                let max = Self::MAX as $source;
                if u < min || u > max {
                    Err(TryFromIntError(()))
                } else {
                    Ok(u as Self)
                }
            }
        }
    )*}
}

macro_rules! rev {
    ($mac:ident, $source:ty => $($target:ty),+) => {$(
        $mac!($target => $source);
    )*}
}

// unsigned integer -> unsigned integer
impl_try_from_upper_bounded!(u16 => u8);
impl_try_from_upper_bounded!(u32 => u8, u16);
impl_try_from_upper_bounded!(u64 => u8, u16, u32);
impl_try_from_upper_bounded!(u128 => u8, u16, u32, u64);

// signed integer -> signed integer
impl_try_from_both_bounded!(i16 => i8);
impl_try_from_both_bounded!(i32 => i8, i16);
impl_try_from_both_bounded!(i64 => i8, i16, i32);
impl_try_from_both_bounded!(i128 => i8, i16, i32, i64);

// unsigned integer -> signed integer
impl_try_from_upper_bounded!(u8 => i8);
impl_try_from_upper_bounded!(u16 => i8, i16);
impl_try_from_upper_bounded!(u32 => i8, i16, i32);
impl_try_from_upper_bounded!(u64 => i8, i16, i32, i64);
impl_try_from_upper_bounded!(u128 => i8, i16, i32, i64, i128);

// signed integer -> unsigned integer
impl_try_from_lower_bounded!(i8 => u8, u16, u32, u64, u128);
impl_try_from_both_bounded!(i16 => u8);
impl_try_from_lower_bounded!(i16 => u16, u32, u64, u128);
impl_try_from_both_bounded!(i32 => u8, u16);
impl_try_from_lower_bounded!(i32 => u32, u64, u128);
impl_try_from_both_bounded!(i64 => u8, u16, u32);
impl_try_from_lower_bounded!(i64 => u64, u128);
impl_try_from_both_bounded!(i128 => u8, u16, u32, u64);
impl_try_from_lower_bounded!(i128 => u128);

// usize/isize
impl_try_from_upper_bounded!(usize => isize);
impl_try_from_lower_bounded!(isize => usize);

#[cfg(target_pointer_width = "16")]
mod ptr_try_from_impls {
    use super::TryFromIntError;

    impl_try_from_upper_bounded!(usize => u8);
    impl_try_from_unbounded!(usize => u16, u32, u64, u128);
    impl_try_from_upper_bounded!(usize => i8, i16);
    impl_try_from_unbounded!(usize => i32, i64, i128);

    impl_try_from_both_bounded!(isize => u8);
    impl_try_from_lower_bounded!(isize => u16, u32, u64, u128);
    impl_try_from_both_bounded!(isize => i8);
    impl_try_from_unbounded!(isize => i16, i32, i64, i128);

    rev!(impl_try_from_upper_bounded, usize => u32, u64, u128);
    rev!(impl_try_from_lower_bounded, usize => i8, i16);
    rev!(impl_try_from_both_bounded, usize => i32, i64, i128);

    rev!(impl_try_from_upper_bounded, isize => u16, u32, u64, u128);
    rev!(impl_try_from_both_bounded, isize => i32, i64, i128);
}

#[cfg(target_pointer_width = "32")]
mod ptr_try_from_impls {
    use super::TryFromIntError;

    impl_try_from_upper_bounded!(usize => u8, u16);
    impl_try_from_unbounded!(usize => u32, u64, u128);
    impl_try_from_upper_bounded!(usize => i8, i16, i32);
    impl_try_from_unbounded!(usize => i64, i128);

    impl_try_from_both_bounded!(isize => u8, u16);
    impl_try_from_lower_bounded!(isize => u32, u64, u128);
    impl_try_from_both_bounded!(isize => i8, i16);
    impl_try_from_unbounded!(isize => i32, i64, i128);

    rev!(impl_try_from_unbounded, usize => u32);
    rev!(impl_try_from_upper_bounded, usize => u64, u128);
    rev!(impl_try_from_lower_bounded, usize => i8, i16, i32);
    rev!(impl_try_from_both_bounded, usize => i64, i128);

    rev!(impl_try_from_unbounded, isize => u16);
    rev!(impl_try_from_upper_bounded, isize => u32, u64, u128);
    rev!(impl_try_from_unbounded, isize => i32);
    rev!(impl_try_from_both_bounded, isize => i64, i128);
}

#[cfg(target_pointer_width = "64")]
mod ptr_try_from_impls {
    use super::TryFromIntError;

    impl_try_from_upper_bounded!(usize => u8, u16, u32);
    impl_try_from_unbounded!(usize => u64, u128);
    impl_try_from_upper_bounded!(usize => i8, i16, i32, i64);
    impl_try_from_unbounded!(usize => i128);

    impl_try_from_both_bounded!(isize => u8, u16, u32);
    impl_try_from_lower_bounded!(isize => u64, u128);
    impl_try_from_both_bounded!(isize => i8, i16, i32);
    impl_try_from_unbounded!(isize => i64, i128);

    rev!(impl_try_from_unbounded, usize => u32, u64);
    rev!(impl_try_from_upper_bounded, usize => u128);
    rev!(impl_try_from_lower_bounded, usize => i8, i16, i32, i64);
    rev!(impl_try_from_both_bounded, usize => i128);

    rev!(impl_try_from_unbounded, isize => u16, u32);
    rev!(impl_try_from_upper_bounded, isize => u64, u128);
    rev!(impl_try_from_unbounded, isize => i32, i64);
    rev!(impl_try_from_both_bounded, isize => i128);
}

// Conversion traits for non-zero integer types
use crate::num::NonZero;

macro_rules! impl_nonzero_int_from_nonzero_int {
    ($Small:ty => $Large:ty) => {
        #[stable(feature = "nz_int_conv", since = "1.41.0")]
        impl From<NonZero<$Small>> for NonZero<$Large> {
            // Rustdocs on the impl block show a "[+] show undocumented items" toggle.
            // Rustdocs on functions do not.
            #[doc = concat!("Converts <code>[NonZero]\\<[", stringify!($Small), "]></code> ")]
            #[doc = concat!("to <code>[NonZero]\\<[", stringify!($Large), "]></code> losslessly.")]
            #[inline]
            fn from(small: NonZero<$Small>) -> Self {
                // SAFETY: input type guarantees the value is non-zero
                unsafe { Self::new_unchecked(From::from(small.get())) }
            }
        }
    };
}

// non-zero unsigned integer -> non-zero unsigned integer
impl_nonzero_int_from_nonzero_int!(u8 => u16);
impl_nonzero_int_from_nonzero_int!(u8 => u32);
impl_nonzero_int_from_nonzero_int!(u8 => u64);
impl_nonzero_int_from_nonzero_int!(u8 => u128);
impl_nonzero_int_from_nonzero_int!(u8 => usize);
impl_nonzero_int_from_nonzero_int!(u16 => u32);
impl_nonzero_int_from_nonzero_int!(u16 => u64);
impl_nonzero_int_from_nonzero_int!(u16 => u128);
impl_nonzero_int_from_nonzero_int!(u16 => usize);
impl_nonzero_int_from_nonzero_int!(u32 => u64);
impl_nonzero_int_from_nonzero_int!(u32 => u128);
impl_nonzero_int_from_nonzero_int!(u64 => u128);

// non-zero signed integer -> non-zero signed integer
impl_nonzero_int_from_nonzero_int!(i8 => i16);
impl_nonzero_int_from_nonzero_int!(i8 => i32);
impl_nonzero_int_from_nonzero_int!(i8 => i64);
impl_nonzero_int_from_nonzero_int!(i8 => i128);
impl_nonzero_int_from_nonzero_int!(i8 => isize);
impl_nonzero_int_from_nonzero_int!(i16 => i32);
impl_nonzero_int_from_nonzero_int!(i16 => i64);
impl_nonzero_int_from_nonzero_int!(i16 => i128);
impl_nonzero_int_from_nonzero_int!(i16 => isize);
impl_nonzero_int_from_nonzero_int!(i32 => i64);
impl_nonzero_int_from_nonzero_int!(i32 => i128);
impl_nonzero_int_from_nonzero_int!(i64 => i128);

// non-zero unsigned -> non-zero signed integer
impl_nonzero_int_from_nonzero_int!(u8 => i16);
impl_nonzero_int_from_nonzero_int!(u8 => i32);
impl_nonzero_int_from_nonzero_int!(u8 => i64);
impl_nonzero_int_from_nonzero_int!(u8 => i128);
impl_nonzero_int_from_nonzero_int!(u8 => isize);
impl_nonzero_int_from_nonzero_int!(u16 => i32);
impl_nonzero_int_from_nonzero_int!(u16 => i64);
impl_nonzero_int_from_nonzero_int!(u16 => i128);
impl_nonzero_int_from_nonzero_int!(u32 => i64);
impl_nonzero_int_from_nonzero_int!(u32 => i128);
impl_nonzero_int_from_nonzero_int!(u64 => i128);

macro_rules! impl_nonzero_int_try_from_int {
    ($Int:ty) => {
        #[stable(feature = "nzint_try_from_int_conv", since = "1.46.0")]
        impl TryFrom<$Int> for NonZero<$Int> {
            type Error = TryFromIntError;

            // Rustdocs on the impl block show a "[+] show undocumented items" toggle.
            // Rustdocs on functions do not.
            #[doc = concat!("Attempts to convert [`", stringify!($Int), "`] ")]
            #[doc = concat!("to <code>[NonZero]\\<[", stringify!($Int), "]></code>.")]
            #[inline]
            fn try_from(value: $Int) -> Result<Self, Self::Error> {
                Self::new(value).ok_or(TryFromIntError(()))
            }
        }
    };
}

// integer -> non-zero integer
impl_nonzero_int_try_from_int!(u8);
impl_nonzero_int_try_from_int!(u16);
impl_nonzero_int_try_from_int!(u32);
impl_nonzero_int_try_from_int!(u64);
impl_nonzero_int_try_from_int!(u128);
impl_nonzero_int_try_from_int!(usize);
impl_nonzero_int_try_from_int!(i8);
impl_nonzero_int_try_from_int!(i16);
impl_nonzero_int_try_from_int!(i32);
impl_nonzero_int_try_from_int!(i64);
impl_nonzero_int_try_from_int!(i128);
impl_nonzero_int_try_from_int!(isize);

macro_rules! impl_nonzero_int_try_from_nonzero_int {
    ($source:ty => $($target:ty),+) => {$(
        #[stable(feature = "nzint_try_from_nzint_conv", since = "1.49.0")]
        impl TryFrom<NonZero<$source>> for NonZero<$target> {
            type Error = TryFromIntError;

            // Rustdocs on the impl block show a "[+] show undocumented items" toggle.
            // Rustdocs on functions do not.
            #[doc = concat!("Attempts to convert <code>[NonZero]\\<[", stringify!($source), "]></code> ")]
            #[doc = concat!("to <code>[NonZero]\\<[", stringify!($target), "]></code>.")]
            #[inline]
            fn try_from(value: NonZero<$source>) -> Result<Self, Self::Error> {
                // SAFETY: Input is guaranteed to be non-zero.
                Ok(unsafe { Self::new_unchecked(<$target>::try_from(value.get())?) })
            }
        }
    )*};
}

// unsigned non-zero integer -> unsigned non-zero integer
impl_nonzero_int_try_from_nonzero_int!(u16 => u8);
impl_nonzero_int_try_from_nonzero_int!(u32 => u8, u16, usize);
impl_nonzero_int_try_from_nonzero_int!(u64 => u8, u16, u32, usize);
impl_nonzero_int_try_from_nonzero_int!(u128 => u8, u16, u32, u64, usize);
impl_nonzero_int_try_from_nonzero_int!(usize => u8, u16, u32, u64, u128);

// signed non-zero integer -> signed non-zero integer
impl_nonzero_int_try_from_nonzero_int!(i16 => i8);
impl_nonzero_int_try_from_nonzero_int!(i32 => i8, i16, isize);
impl_nonzero_int_try_from_nonzero_int!(i64 => i8, i16, i32, isize);
impl_nonzero_int_try_from_nonzero_int!(i128 => i8, i16, i32, i64, isize);
impl_nonzero_int_try_from_nonzero_int!(isize => i8, i16, i32, i64, i128);

// unsigned non-zero integer -> signed non-zero integer
impl_nonzero_int_try_from_nonzero_int!(u8 => i8);
impl_nonzero_int_try_from_nonzero_int!(u16 => i8, i16, isize);
impl_nonzero_int_try_from_nonzero_int!(u32 => i8, i16, i32, isize);
impl_nonzero_int_try_from_nonzero_int!(u64 => i8, i16, i32, i64, isize);
impl_nonzero_int_try_from_nonzero_int!(u128 => i8, i16, i32, i64, i128, isize);
impl_nonzero_int_try_from_nonzero_int!(usize => i8, i16, i32, i64, i128, isize);

// signed non-zero integer -> unsigned non-zero integer
impl_nonzero_int_try_from_nonzero_int!(i8 => u8, u16, u32, u64, u128, usize);
impl_nonzero_int_try_from_nonzero_int!(i16 => u8, u16, u32, u64, u128, usize);
impl_nonzero_int_try_from_nonzero_int!(i32 => u8, u16, u32, u64, u128, usize);
impl_nonzero_int_try_from_nonzero_int!(i64 => u8, u16, u32, u64, u128, usize);
impl_nonzero_int_try_from_nonzero_int!(i128 => u8, u16, u32, u64, u128, usize);
impl_nonzero_int_try_from_nonzero_int!(isize => u8, u16, u32, u64, u128, usize);
