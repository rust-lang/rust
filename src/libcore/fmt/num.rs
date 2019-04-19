//! Integer and floating-point number formatting


use crate::fmt;
use crate::ops::{Div, Rem, Sub};
use crate::str;
use crate::slice;
use crate::ptr;
use crate::mem::MaybeUninit;

#[doc(hidden)]
trait Int: PartialEq + PartialOrd + Div<Output=Self> + Rem<Output=Self> +
           Sub<Output=Self> + Copy {
    fn zero() -> Self;
    fn from_u8(u: u8) -> Self;
    fn to_u8(&self) -> u8;
    fn to_u16(&self) -> u16;
    fn to_u32(&self) -> u32;
    fn to_u64(&self) -> u64;
    fn to_u128(&self) -> u128;
}

macro_rules! doit {
    ($($t:ident)*) => ($(impl Int for $t {
        fn zero() -> $t { 0 }
        fn from_u8(u: u8) -> $t { u as $t }
        fn to_u8(&self) -> u8 { *self as u8 }
        fn to_u16(&self) -> u16 { *self as u16 }
        fn to_u32(&self) -> u32 { *self as u32 }
        fn to_u64(&self) -> u64 { *self as u64 }
        fn to_u128(&self) -> u128 { *self as u128 }
    })*)
}
doit! { i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize }

/// A type that represents a specific radix
#[doc(hidden)]
trait GenericRadix {
    /// The number of digits.
    const BASE: u8;

    /// A radix-specific prefix string.
    const PREFIX: &'static str;

    /// Converts an integer to corresponding radix digit.
    fn digit(x: u8) -> u8;

    /// Format an integer using the radix using a formatter.
    fn fmt_int<T: Int>(&self, mut x: T, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The radix can be as low as 2, so we need a buffer of at least 128
        // characters for a base 2 number.
        let zero = T::zero();
        let is_nonnegative = x >= zero;
        let mut buf = uninitialized_array![u8; 128];
        let mut curr = buf.len();
        let base = T::from_u8(Self::BASE);
        if is_nonnegative {
            // Accumulate each digit of the number from the least significant
            // to the most significant figure.
            for byte in buf.iter_mut().rev() {
                let n = x % base;               // Get the current place value.
                x = x / base;                   // Deaccumulate the number.
                byte.write(Self::digit(n.to_u8())); // Store the digit in the buffer.
                curr -= 1;
                if x == zero {
                    // No more digits left to accumulate.
                    break
                };
            }
        } else {
            // Do the same as above, but accounting for two's complement.
            for byte in buf.iter_mut().rev() {
                let n = zero - (x % base);      // Get the current place value.
                x = x / base;                   // Deaccumulate the number.
                byte.write(Self::digit(n.to_u8())); // Store the digit in the buffer.
                curr -= 1;
                if x == zero {
                    // No more digits left to accumulate.
                    break
                };
            }
        }
        let buf = &buf[curr..];
        let buf = unsafe { str::from_utf8_unchecked(slice::from_raw_parts(
            MaybeUninit::first_ptr(buf),
            buf.len()
        )) };
        f.pad_integral(is_nonnegative, Self::PREFIX, buf)
    }
}

/// A binary (base 2) radix
#[derive(Clone, PartialEq)]
struct Binary;

/// An octal (base 8) radix
#[derive(Clone, PartialEq)]
struct Octal;

/// A hexadecimal (base 16) radix, formatted with lower-case characters
#[derive(Clone, PartialEq)]
struct LowerHex;

/// A hexadecimal (base 16) radix, formatted with upper-case characters
#[derive(Clone, PartialEq)]
struct UpperHex;

macro_rules! radix {
    ($T:ident, $base:expr, $prefix:expr, $($x:pat => $conv:expr),+) => {
        impl GenericRadix for $T {
            const BASE: u8 = $base;
            const PREFIX: &'static str = $prefix;
            fn digit(x: u8) -> u8 {
                match x {
                    $($x => $conv,)+
                    x => panic!("number not in the range 0..={}: {}", Self::BASE - 1, x),
                }
            }
        }
    }
}

radix! { Binary,    2, "0b", x @  0 ..=  1 => b'0' + x }
radix! { Octal,     8, "0o", x @  0 ..=  7 => b'0' + x }
radix! { LowerHex, 16, "0x", x @  0 ..=  9 => b'0' + x,
                             x @ 10 ..= 15 => b'a' + (x - 10) }
radix! { UpperHex, 16, "0x", x @  0 ..=  9 => b'0' + x,
                             x @ 10 ..= 15 => b'A' + (x - 10) }

macro_rules! int_base {
    ($Trait:ident for $T:ident as $U:ident -> $Radix:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::$Trait for $T {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $Radix.fmt_int(*self as $U, f)
            }
        }
    }
}

macro_rules! debug {
    ($T:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::Debug for $T {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if f.debug_lower_hex() {
                    fmt::LowerHex::fmt(self, f)
                } else if f.debug_upper_hex() {
                    fmt::UpperHex::fmt(self, f)
                } else {
                    fmt::Display::fmt(self, f)
                }
            }
        }
    }
}

macro_rules! integer {
    ($Int:ident, $Uint:ident) => {
        int_base! { Binary   for $Int as $Uint  -> Binary }
        int_base! { Octal    for $Int as $Uint  -> Octal }
        int_base! { LowerHex for $Int as $Uint  -> LowerHex }
        int_base! { UpperHex for $Int as $Uint  -> UpperHex }
        debug! { $Int }

        int_base! { Binary   for $Uint as $Uint -> Binary }
        int_base! { Octal    for $Uint as $Uint -> Octal }
        int_base! { LowerHex for $Uint as $Uint -> LowerHex }
        int_base! { UpperHex for $Uint as $Uint -> UpperHex }
        debug! { $Uint }
    }
}
integer! { isize, usize }
integer! { i8, u8 }
integer! { i16, u16 }
integer! { i32, u32 }
integer! { i64, u64 }
integer! { i128, u128 }


static DEC_DIGITS_LUT: &[u8; 200] =
    b"0001020304050607080910111213141516171819\
      2021222324252627282930313233343536373839\
      4041424344454647484950515253545556575859\
      6061626364656667686970717273747576777879\
      8081828384858687888990919293949596979899";

macro_rules! impl_Display {
    ($($t:ident),* as $u:ident via $conv_fn:ident named $name:ident) => {
        fn $name(mut n: $u, is_nonnegative: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut buf = uninitialized_array![u8; 39];
            let mut curr = buf.len() as isize;
            let buf_ptr = MaybeUninit::first_ptr_mut(&mut buf);
            let lut_ptr = DEC_DIGITS_LUT.as_ptr();

            unsafe {
                // need at least 16 bits for the 4-characters-at-a-time to work.
                assert!(crate::mem::size_of::<$u>() >= 2);

                // eagerly decode 4 characters at a time
                while n >= 10000 {
                    let rem = (n % 10000) as isize;
                    n /= 10000;

                    let d1 = (rem / 100) << 1;
                    let d2 = (rem % 100) << 1;
                    curr -= 4;
                    ptr::copy_nonoverlapping(lut_ptr.offset(d1), buf_ptr.offset(curr), 2);
                    ptr::copy_nonoverlapping(lut_ptr.offset(d2), buf_ptr.offset(curr + 2), 2);
                }

                // if we reach here numbers are <= 9999, so at most 4 chars long
                let mut n = n as isize; // possibly reduce 64bit math

                // decode 2 more chars, if > 2 chars
                if n >= 100 {
                    let d1 = (n % 100) << 1;
                    n /= 100;
                    curr -= 2;
                    ptr::copy_nonoverlapping(lut_ptr.offset(d1), buf_ptr.offset(curr), 2);
                }

                // decode last 1 or 2 chars
                if n < 10 {
                    curr -= 1;
                    *buf_ptr.offset(curr) = (n as u8) + b'0';
                } else {
                    let d1 = n << 1;
                    curr -= 2;
                    ptr::copy_nonoverlapping(lut_ptr.offset(d1), buf_ptr.offset(curr), 2);
                }
            }

            let buf_slice = unsafe {
                str::from_utf8_unchecked(
                    slice::from_raw_parts(buf_ptr.offset(curr), buf.len() - curr as usize))
            };
            f.pad_integral(is_nonnegative, "", buf_slice)
        }

        $(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl fmt::Display for $t {
                #[allow(unused_comparisons)]
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let is_nonnegative = *self >= 0;
                    let n = if is_nonnegative {
                        self.$conv_fn()
                    } else {
                        // convert the negative num to positive by summing 1 to it's 2 complement
                        (!self.$conv_fn()).wrapping_add(1)
                    };
                    $name(n, is_nonnegative, f)
                }
            })*
    };
}

// Include wasm32 in here since it doesn't reflect the native pointer size, and
// often cares strongly about getting a smaller code size.
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
mod imp {
    use super::*;
    impl_Display!(
        i8, u8, i16, u16, i32, u32, i64, u64, usize, isize
            as u64 via to_u64 named fmt_u64
    );
}

#[cfg(not(any(target_pointer_width = "64", target_arch = "wasm32")))]
mod imp {
    use super::*;
    impl_Display!(i8, u8, i16, u16, i32, u32, isize, usize as u32 via to_u32 named fmt_u32);
    impl_Display!(i64, u64 as u64 via to_u64 named fmt_u64);
}

impl_Display!(i128, u128 as u128 via to_u128 named fmt_u128);
