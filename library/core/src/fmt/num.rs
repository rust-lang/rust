//! Integer and floating-point number formatting

use crate::fmt;
use crate::mem::MaybeUninit;
use crate::num::fmt as numfmt;
use crate::ops::{Div, Rem, Sub};
use crate::ptr;
use crate::slice;
use crate::str;

#[doc(hidden)]
trait DisplayInt:
    PartialEq + PartialOrd + Div<Output = Self> + Rem<Output = Self> + Sub<Output = Self> + Copy
{
    fn zero() -> Self;
    fn from_u8(u: u8) -> Self;
    fn to_u8(&self) -> u8;
    fn to_u16(&self) -> u16;
    fn to_u32(&self) -> u32;
    fn to_u64(&self) -> u64;
    fn to_u128(&self) -> u128;
}

macro_rules! impl_int {
    ($($t:ident)*) => (
      $(impl DisplayInt for $t {
          fn zero() -> Self { 0 }
          fn from_u8(u: u8) -> Self { u as Self }
          fn to_u8(&self) -> u8 { *self as u8 }
          fn to_u16(&self) -> u16 { *self as u16 }
          fn to_u32(&self) -> u32 { *self as u32 }
          fn to_u64(&self) -> u64 { *self as u64 }
          fn to_u128(&self) -> u128 { *self as u128 }
      })*
    )
}
macro_rules! impl_uint {
    ($($t:ident)*) => (
      $(impl DisplayInt for $t {
          fn zero() -> Self { 0 }
          fn from_u8(u: u8) -> Self { u as Self }
          fn to_u8(&self) -> u8 { *self as u8 }
          fn to_u16(&self) -> u16 { *self as u16 }
          fn to_u32(&self) -> u32 { *self as u32 }
          fn to_u64(&self) -> u64 { *self as u64 }
          fn to_u128(&self) -> u128 { *self as u128 }
      })*
    )
}

impl_int! { i8 i16 i32 i64 i128 isize }
impl_uint! { u8 u16 u32 u64 u128 usize }

/// A type that represents a specific radix
#[doc(hidden)]
trait GenericRadix: Sized {
    /// The number of digits.
    const BASE: u8;

    /// A radix-specific prefix string.
    const PREFIX: &'static str;

    /// Converts an integer to corresponding radix digit.
    fn digit(x: u8) -> u8;

    /// Format an integer using the radix using a formatter.
    fn fmt_int<T: DisplayInt>(&self, mut x: T, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The radix can be as low as 2, so we need a buffer of at least 128
        // characters for a base 2 number.
        let zero = T::zero();
        let is_nonnegative = x >= zero;
        let mut buf = [MaybeUninit::<u8>::uninit(); 128];
        let mut curr = buf.len();
        let base = T::from_u8(Self::BASE);
        if is_nonnegative {
            // Accumulate each digit of the number from the least significant
            // to the most significant figure.
            for byte in buf.iter_mut().rev() {
                let n = x % base; // Get the current place value.
                x = x / base; // Deaccumulate the number.
                byte.write(Self::digit(n.to_u8())); // Store the digit in the buffer.
                curr -= 1;
                if x == zero {
                    // No more digits left to accumulate.
                    break;
                };
            }
        } else {
            // Do the same as above, but accounting for two's complement.
            for byte in buf.iter_mut().rev() {
                let n = zero - (x % base); // Get the current place value.
                x = x / base; // Deaccumulate the number.
                byte.write(Self::digit(n.to_u8())); // Store the digit in the buffer.
                curr -= 1;
                if x == zero {
                    // No more digits left to accumulate.
                    break;
                };
            }
        }
        let buf = &buf[curr..];
        // SAFETY: The only chars in `buf` are created by `Self::digit` which are assumed to be
        // valid UTF-8
        let buf = unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(
                MaybeUninit::slice_as_ptr(buf),
                buf.len(),
            ))
        };
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
radix! { LowerHex, 16, "0x", x @  0 ..=  9 => b'0' + x, x @ 10 ..= 15 => b'a' + (x - 10) }
radix! { UpperHex, 16, "0x", x @  0 ..=  9 => b'0' + x, x @ 10 ..= 15 => b'A' + (x - 10) }

macro_rules! int_base {
    (fmt::$Trait:ident for $T:ident as $U:ident -> $Radix:ident) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::$Trait for $T {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $Radix.fmt_int(*self as $U, f)
            }
        }
    };
}

macro_rules! integer {
    ($Int:ident, $Uint:ident) => {
        int_base! { fmt::Binary   for $Int as $Uint  -> Binary }
        int_base! { fmt::Octal    for $Int as $Uint  -> Octal }
        int_base! { fmt::LowerHex for $Int as $Uint  -> LowerHex }
        int_base! { fmt::UpperHex for $Int as $Uint  -> UpperHex }

        int_base! { fmt::Binary   for $Uint as $Uint -> Binary }
        int_base! { fmt::Octal    for $Uint as $Uint -> Octal }
        int_base! { fmt::LowerHex for $Uint as $Uint -> LowerHex }
        int_base! { fmt::UpperHex for $Uint as $Uint -> UpperHex }
    };
}
integer! { isize, usize }
integer! { i8, u8 }
integer! { i16, u16 }
integer! { i32, u32 }
integer! { i64, u64 }
integer! { i128, u128 }
macro_rules! debug {
    ($($T:ident)*) => {$(
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
    )*};
}
debug! {
  i8 i16 i32 i64 i128 isize
  u8 u16 u32 u64 u128 usize
}

// 2 digit decimal look up table
static DEC_DIGITS_LUT: &[u8; 200] = b"0001020304050607080910111213141516171819\
      2021222324252627282930313233343536373839\
      4041424344454647484950515253545556575859\
      6061626364656667686970717273747576777879\
      8081828384858687888990919293949596979899";

macro_rules! impl_Display {
    ($($t:ident),* as $u:ident via $conv_fn:ident named $name:ident) => {
        fn $name(mut n: $u, is_nonnegative: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            // 2^128 is about 3*10^38, so 39 gives an extra byte of space
            let mut buf = [MaybeUninit::<u8>::uninit(); 39];
            let mut curr = buf.len() as isize;
            let buf_ptr = MaybeUninit::slice_as_mut_ptr(&mut buf);
            let lut_ptr = DEC_DIGITS_LUT.as_ptr();

            // SAFETY: Since `d1` and `d2` are always less than or equal to `198`, we
            // can copy from `lut_ptr[d1..d1 + 1]` and `lut_ptr[d2..d2 + 1]`. To show
            // that it's OK to copy into `buf_ptr`, notice that at the beginning
            // `curr == buf.len() == 39 > log(n)` since `n < 2^128 < 10^39`, and at
            // each step this is kept the same as `n` is divided. Since `n` is always
            // non-negative, this means that `curr > 0` so `buf_ptr[curr..curr + 1]`
            // is safe to access.
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

                    // We are allowed to copy to `buf_ptr[curr..curr + 3]` here since
                    // otherwise `curr < 0`. But then `n` was originally at least `10000^10`
                    // which is `10^40 > 2^128 > n`.
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

            // SAFETY: `curr` > 0 (since we made `buf` large enough), and all the chars are valid
            // UTF-8 since `DEC_DIGITS_LUT` is
            let buf_slice = unsafe {
                str::from_utf8_unchecked(
                    slice::from_raw_parts(buf_ptr.offset(curr), buf.len() - curr as usize))
            };
            f.pad_integral(is_nonnegative, "", buf_slice)
        }

        $(#[stable(feature = "rust1", since = "1.0.0")]
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

macro_rules! impl_Exp {
    ($($t:ident),* as $u:ident via $conv_fn:ident named $name:ident) => {
        fn $name(
            mut n: $u,
            is_nonnegative: bool,
            upper: bool,
            f: &mut fmt::Formatter<'_>
        ) -> fmt::Result {
            let (mut n, mut exponent, trailing_zeros, added_precision) = {
                let mut exponent = 0;
                // count and remove trailing decimal zeroes
                while n % 10 == 0 && n >= 10 {
                    n /= 10;
                    exponent += 1;
                }

                let (added_precision, subtracted_precision) = match f.precision() {
                    Some(fmt_prec) => {
                        // number of decimal digits minus 1
                        let mut tmp = n;
                        let mut prec = 0;
                        while tmp >= 10 {
                            tmp /= 10;
                            prec += 1;
                        }
                        (fmt_prec.saturating_sub(prec), prec.saturating_sub(fmt_prec))
                    }
                    None => (0,0)
                };
                for _ in 1..subtracted_precision {
                    n/=10;
                    exponent += 1;
                }
                if subtracted_precision != 0 {
                    let rem = n % 10;
                    n /= 10;
                    exponent += 1;
                    // round up last digit
                    if rem >= 5 {
                        n += 1;
                    }
                }
                (n, exponent, exponent, added_precision)
            };

            // 39 digits (worst case u128) + . = 40
            // Since `curr` always decreases by the number of digits copied, this means
            // that `curr >= 0`.
            let mut buf = [MaybeUninit::<u8>::uninit(); 40];
            let mut curr = buf.len() as isize; //index for buf
            let buf_ptr = MaybeUninit::slice_as_mut_ptr(&mut buf);
            let lut_ptr = DEC_DIGITS_LUT.as_ptr();

            // decode 2 chars at a time
            while n >= 100 {
                let d1 = ((n % 100) as isize) << 1;
                curr -= 2;
                // SAFETY: `d1 <= 198`, so we can copy from `lut_ptr[d1..d1 + 2]` since
                // `DEC_DIGITS_LUT` has a length of 200.
                unsafe {
                    ptr::copy_nonoverlapping(lut_ptr.offset(d1), buf_ptr.offset(curr), 2);
                }
                n /= 100;
                exponent += 2;
            }
            // n is <= 99, so at most 2 chars long
            let mut n = n as isize; // possibly reduce 64bit math
            // decode second-to-last character
            if n >= 10 {
                curr -= 1;
                // SAFETY: Safe since `40 > curr >= 0` (see comment)
                unsafe {
                    *buf_ptr.offset(curr) = (n as u8 % 10_u8) + b'0';
                }
                n /= 10;
                exponent += 1;
            }
            // add decimal point iff >1 mantissa digit will be printed
            if exponent != trailing_zeros || added_precision != 0 {
                curr -= 1;
                // SAFETY: Safe since `40 > curr >= 0`
                unsafe {
                    *buf_ptr.offset(curr) = b'.';
                }
            }

            // SAFETY: Safe since `40 > curr >= 0`
            let buf_slice = unsafe {
                // decode last character
                curr -= 1;
                *buf_ptr.offset(curr) = (n as u8) + b'0';

                let len = buf.len() - curr as usize;
                slice::from_raw_parts(buf_ptr.offset(curr), len)
            };

            // stores 'e' (or 'E') and the up to 2-digit exponent
            let mut exp_buf = [MaybeUninit::<u8>::uninit(); 3];
            let exp_ptr = MaybeUninit::slice_as_mut_ptr(&mut exp_buf);
            // SAFETY: In either case, `exp_buf` is written within bounds and `exp_ptr[..len]`
            // is contained within `exp_buf` since `len <= 3`.
            let exp_slice = unsafe {
                *exp_ptr.offset(0) = if upper {b'E'} else {b'e'};
                let len = if exponent < 10 {
                    *exp_ptr.offset(1) = (exponent as u8) + b'0';
                    2
                } else {
                    let off = exponent << 1;
                    ptr::copy_nonoverlapping(lut_ptr.offset(off), exp_ptr.offset(1), 2);
                    3
                };
                slice::from_raw_parts(exp_ptr, len)
            };

            let parts = &[
                numfmt::Part::Copy(buf_slice),
                numfmt::Part::Zero(added_precision),
                numfmt::Part::Copy(exp_slice)
            ];
            let sign = if !is_nonnegative {
                "-"
            } else if f.sign_plus() {
                "+"
            } else {
                ""
            };
            let formatted = numfmt::Formatted{sign, parts};
            f.pad_formatted_parts(&formatted)
        }

        $(
            #[stable(feature = "integer_exp_format", since = "1.42.0")]
            impl fmt::LowerExp for $t {
                #[allow(unused_comparisons)]
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let is_nonnegative = *self >= 0;
                    let n = if is_nonnegative {
                        self.$conv_fn()
                    } else {
                        // convert the negative num to positive by summing 1 to it's 2 complement
                        (!self.$conv_fn()).wrapping_add(1)
                    };
                    $name(n, is_nonnegative, false, f)
                }
            })*
        $(
            #[stable(feature = "integer_exp_format", since = "1.42.0")]
            impl fmt::UpperExp for $t {
                #[allow(unused_comparisons)]
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let is_nonnegative = *self >= 0;
                    let n = if is_nonnegative {
                        self.$conv_fn()
                    } else {
                        // convert the negative num to positive by summing 1 to it's 2 complement
                        (!self.$conv_fn()).wrapping_add(1)
                    };
                    $name(n, is_nonnegative, true, f)
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
    impl_Exp!(
        i8, u8, i16, u16, i32, u32, i64, u64, usize, isize
            as u64 via to_u64 named exp_u64
    );
}

#[cfg(not(any(target_pointer_width = "64", target_arch = "wasm32")))]
mod imp {
    use super::*;
    impl_Display!(i8, u8, i16, u16, i32, u32, isize, usize as u32 via to_u32 named fmt_u32);
    impl_Display!(i64, u64 as u64 via to_u64 named fmt_u64);
    impl_Exp!(i8, u8, i16, u16, i32, u32, isize, usize as u32 via to_u32 named exp_u32);
    impl_Exp!(i64, u64 as u64 via to_u64 named exp_u64);
}
impl_Exp!(i128, u128 as u128 via to_u128 named exp_u128);

/// Helper function for writing a u64 into `buf` going from last to first, with `curr`.
fn parse_u64_into<const N: usize>(mut n: u64, buf: &mut [MaybeUninit<u8>; N], curr: &mut isize) {
    let buf_ptr = MaybeUninit::slice_as_mut_ptr(buf);
    let lut_ptr = DEC_DIGITS_LUT.as_ptr();
    assert!(*curr > 19);

    // SAFETY:
    // Writes at most 19 characters into the buffer. Guaranteed that any ptr into LUT is at most
    // 198, so will never OOB. There is a check above that there are at least 19 characters
    // remaining.
    unsafe {
        if n >= 1e16 as u64 {
            let to_parse = n % 1e16 as u64;
            n /= 1e16 as u64;

            // Some of these are nops but it looks more elegant this way.
            let d1 = ((to_parse / 1e14 as u64) % 100) << 1;
            let d2 = ((to_parse / 1e12 as u64) % 100) << 1;
            let d3 = ((to_parse / 1e10 as u64) % 100) << 1;
            let d4 = ((to_parse / 1e8 as u64) % 100) << 1;
            let d5 = ((to_parse / 1e6 as u64) % 100) << 1;
            let d6 = ((to_parse / 1e4 as u64) % 100) << 1;
            let d7 = ((to_parse / 1e2 as u64) % 100) << 1;
            let d8 = ((to_parse / 1e0 as u64) % 100) << 1;

            *curr -= 16;

            ptr::copy_nonoverlapping(lut_ptr.offset(d1 as isize), buf_ptr.offset(*curr + 0), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d2 as isize), buf_ptr.offset(*curr + 2), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d3 as isize), buf_ptr.offset(*curr + 4), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d4 as isize), buf_ptr.offset(*curr + 6), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d5 as isize), buf_ptr.offset(*curr + 8), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d6 as isize), buf_ptr.offset(*curr + 10), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d7 as isize), buf_ptr.offset(*curr + 12), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d8 as isize), buf_ptr.offset(*curr + 14), 2);
        }
        if n >= 1e8 as u64 {
            let to_parse = n % 1e8 as u64;
            n /= 1e8 as u64;

            // Some of these are nops but it looks more elegant this way.
            let d1 = ((to_parse / 1e6 as u64) % 100) << 1;
            let d2 = ((to_parse / 1e4 as u64) % 100) << 1;
            let d3 = ((to_parse / 1e2 as u64) % 100) << 1;
            let d4 = ((to_parse / 1e0 as u64) % 100) << 1;
            *curr -= 8;

            ptr::copy_nonoverlapping(lut_ptr.offset(d1 as isize), buf_ptr.offset(*curr + 0), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d2 as isize), buf_ptr.offset(*curr + 2), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d3 as isize), buf_ptr.offset(*curr + 4), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d4 as isize), buf_ptr.offset(*curr + 6), 2);
        }
        // `n` < 1e8 < (1 << 32)
        let mut n = n as u32;
        if n >= 1e4 as u32 {
            let to_parse = n % 1e4 as u32;
            n /= 1e4 as u32;

            let d1 = (to_parse / 100) << 1;
            let d2 = (to_parse % 100) << 1;
            *curr -= 4;

            ptr::copy_nonoverlapping(lut_ptr.offset(d1 as isize), buf_ptr.offset(*curr + 0), 2);
            ptr::copy_nonoverlapping(lut_ptr.offset(d2 as isize), buf_ptr.offset(*curr + 2), 2);
        }

        // `n` < 1e4 < (1 << 16)
        let mut n = n as u16;
        if n >= 100 {
            let d1 = (n % 100) << 1;
            n /= 100;
            *curr -= 2;
            ptr::copy_nonoverlapping(lut_ptr.offset(d1 as isize), buf_ptr.offset(*curr), 2);
        }

        // decode last 1 or 2 chars
        if n < 10 {
            *curr -= 1;
            *buf_ptr.offset(*curr) = (n as u8) + b'0';
        } else {
            let d1 = n << 1;
            *curr -= 2;
            ptr::copy_nonoverlapping(lut_ptr.offset(d1 as isize), buf_ptr.offset(*curr), 2);
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for u128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_u128(*self, true, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for i128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let is_nonnegative = *self >= 0;
        let n = if is_nonnegative {
            self.to_u128()
        } else {
            // convert the negative num to positive by summing 1 to it's 2 complement
            (!self.to_u128()).wrapping_add(1)
        };
        fmt_u128(n, is_nonnegative, f)
    }
}

/// Specialized optimization for u128. Instead of taking two items at a time, it splits
/// into at most 2 u64s, and then chunks by 10e16, 10e8, 10e4, 10e2, and then 10e1.
/// It also has to handle 1 last item, as 10^40 > 2^128 > 10^39, whereas
/// 10^20 > 2^64 > 10^19.
fn fmt_u128(n: u128, is_nonnegative: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // 2^128 is about 3*10^38, so 39 gives an extra byte of space
    let mut buf = [MaybeUninit::<u8>::uninit(); 39];
    let mut curr = buf.len() as isize;

    let (n, rem) = udiv_1e19(n);
    parse_u64_into(rem, &mut buf, &mut curr);

    if n != 0 {
        // 0 pad up to point
        let target = (buf.len() - 19) as isize;
        // SAFETY: Guaranteed that we wrote at most 19 bytes, and there must be space
        // remaining since it has length 39
        unsafe {
            ptr::write_bytes(
                MaybeUninit::slice_as_mut_ptr(&mut buf).offset(target),
                b'0',
                (curr - target) as usize,
            );
        }
        curr = target;

        let (n, rem) = udiv_1e19(n);
        parse_u64_into(rem, &mut buf, &mut curr);
        // Should this following branch be annotated with unlikely?
        if n != 0 {
            let target = (buf.len() - 38) as isize;
            // The raw `buf_ptr` pointer is only valid until `buf` is used the next time,
            // buf `buf` is not used in this scope so we are good.
            let buf_ptr = MaybeUninit::slice_as_mut_ptr(&mut buf);
            // SAFETY: At this point we wrote at most 38 bytes, pad up to that point,
            // There can only be at most 1 digit remaining.
            unsafe {
                ptr::write_bytes(buf_ptr.offset(target), b'0', (curr - target) as usize);
                curr = target - 1;
                *buf_ptr.offset(curr) = (n as u8) + b'0';
            }
        }
    }

    // SAFETY: `curr` > 0 (since we made `buf` large enough), and all the chars are valid
    // UTF-8 since `DEC_DIGITS_LUT` is
    let buf_slice = unsafe {
        str::from_utf8_unchecked(slice::from_raw_parts(
            MaybeUninit::slice_as_mut_ptr(&mut buf).offset(curr),
            buf.len() - curr as usize,
        ))
    };
    f.pad_integral(is_nonnegative, "", buf_slice)
}

/// Partition of `n` into n > 1e19 and rem <= 1e19
///
/// Integer division algorithm is based on the following paper:
///
///   T. Granlund and P. Montgomery, “Division by Invariant Integers Using Multiplication”
///   in Proc. of the SIGPLAN94 Conference on Programming Language Design and
///   Implementation, 1994, pp. 61–72
///
fn udiv_1e19(n: u128) -> (u128, u64) {
    const DIV: u64 = 1e19 as u64;
    const FACTOR: u128 = 156927543384667019095894735580191660403;

    let quot = if n < 1 << 83 {
        ((n >> 19) as u64 / (DIV >> 19)) as u128
    } else {
        u128_mulhi(n, FACTOR) >> 62
    };

    let rem = (n - quot * DIV as u128) as u64;
    (quot, rem)
}

/// Multiply unsigned 128 bit integers, return upper 128 bits of the result
#[inline]
fn u128_mulhi(x: u128, y: u128) -> u128 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let y_lo = y as u64;
    let y_hi = (y >> 64) as u64;

    // handle possibility of overflow
    let carry = (x_lo as u128 * y_lo as u128) >> 64;
    let m = x_lo as u128 * y_hi as u128 + carry;
    let high1 = m >> 64;

    let m_lo = m as u64;
    let high2 = (x_hi as u128 * y_lo as u128 + m_lo as u128) >> 64;

    x_hi as u128 * y_hi as u128 + high1 + high2
}
