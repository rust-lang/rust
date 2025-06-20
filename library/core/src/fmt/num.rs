//! Integer and floating-point number formatting

use crate::mem::MaybeUninit;
use crate::num::fmt as numfmt;
use crate::ops::{Div, Rem, Sub};
use crate::{fmt, ptr, slice, str};

#[doc(hidden)]
trait DisplayInt:
    PartialEq + PartialOrd + Div<Output = Self> + Rem<Output = Self> + Sub<Output = Self> + Copy
{
    fn zero() -> Self;
    fn from_u8(u: u8) -> Self;
    fn to_u8(&self) -> u8;
    #[cfg(not(any(target_pointer_width = "64", target_arch = "wasm32")))]
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
            #[cfg(not(any(target_pointer_width = "64", target_arch = "wasm32")))]
            fn to_u32(&self) -> u32 { *self as u32 }
            fn to_u64(&self) -> u64 { *self as u64 }
            fn to_u128(&self) -> u128 { *self as u128 }
        })*
    )
}

impl_int! {
    i8 i16 i32 i64 i128 isize
    u8 u16 u32 u64 u128 usize
}

/// A type that represents a specific radix
///
/// # Safety
///
/// `digit` must return an ASCII character.
#[doc(hidden)]
unsafe trait GenericRadix: Sized {
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
            loop {
                let n = x % base; // Get the current place value.
                x = x / base; // Deaccumulate the number.
                curr -= 1;
                buf[curr].write(Self::digit(n.to_u8())); // Store the digit in the buffer.
                if x == zero {
                    // No more digits left to accumulate.
                    break;
                };
            }
        } else {
            // Do the same as above, but accounting for two's complement.
            loop {
                let n = zero - (x % base); // Get the current place value.
                x = x / base; // Deaccumulate the number.
                curr -= 1;
                buf[curr].write(Self::digit(n.to_u8())); // Store the digit in the buffer.
                if x == zero {
                    // No more digits left to accumulate.
                    break;
                };
            }
        }
        // SAFETY: `curr` is initialized to `buf.len()` and is only decremented, so it can't overflow. It is
        // decremented exactly once for each digit. Since u128 is the widest fixed width integer format supported,
        // the maximum number of digits (bits) is 128 for base-2, so `curr` won't underflow as well.
        let buf = unsafe { buf.get_unchecked(curr..) };
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
        unsafe impl GenericRadix for $T {
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

macro_rules! impl_Debug {
    ($($T:ident)*) => {
        $(
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
        )*
    };
}

// 2 digit decimal look up table
static DEC_DIGITS_LUT: &[u8; 200] = b"\
      0001020304050607080910111213141516171819\
      2021222324252627282930313233343536373839\
      4041424344454647484950515253545556575859\
      6061626364656667686970717273747576777879\
      8081828384858687888990919293949596979899";

macro_rules! impl_Display {
    ($($signed:ident, $unsigned:ident,)* ; as $u:ident via $conv_fn:ident named $gen_name:ident) => {

        $(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::Display for $unsigned {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                #[cfg(not(feature = "optimize_for_size"))]
                {
                    const MAX_DEC_N: usize = $unsigned::MAX.ilog10() as usize + 1;
                    // Buffer decimals for $unsigned with right alignment.
                    let mut buf = [MaybeUninit::<u8>::uninit(); MAX_DEC_N];

                    f.pad_integral(true, "", self._fmt(&mut buf))
                }
                #[cfg(feature = "optimize_for_size")]
                {
                    $gen_name(self.$conv_fn(), true, f)
                }
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::Display for $signed {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                #[cfg(not(feature = "optimize_for_size"))]
                {
                    const MAX_DEC_N: usize = $unsigned::MAX.ilog10() as usize + 1;
                    // Buffer decimals for $unsigned with right alignment.
                    let mut buf = [MaybeUninit::<u8>::uninit(); MAX_DEC_N];

                    f.pad_integral(*self >= 0, "", self.unsigned_abs()._fmt(&mut buf))
                }
                #[cfg(feature = "optimize_for_size")]
                {
                    return $gen_name(self.unsigned_abs().$conv_fn(), *self >= 0, f);
                }
            }
        }

        #[cfg(not(feature = "optimize_for_size"))]
        impl $unsigned {
            #[doc(hidden)]
            #[unstable(
                feature = "fmt_internals",
                reason = "specialized method meant to only be used by `SpecToString` implementation",
                issue = "none"
            )]
            pub fn _fmt<'a>(self, buf: &'a mut [MaybeUninit::<u8>]) -> &'a str {
                // Count the number of bytes in buf that are not initialized.
                let mut offset = buf.len();
                // Consume the least-significant decimals from a working copy.
                let mut remain = self;

                // Format per four digits from the lookup table.
                // Four digits need a 16-bit $unsigned or wider.
                while size_of::<Self>() > 1 && remain > 999.try_into().expect("branch is not hit for types that cannot fit 999 (u8)") {
                    // SAFETY: All of the decimals fit in buf due to MAX_DEC_N
                    // and the while condition ensures at least 4 more decimals.
                    unsafe { core::hint::assert_unchecked(offset >= 4) }
                    // SAFETY: The offset counts down from its initial buf.len()
                    // without underflow due to the previous precondition.
                    unsafe { core::hint::assert_unchecked(offset <= buf.len()) }
                    offset -= 4;

                    // pull two pairs
                    let scale: Self = 1_00_00.try_into().expect("branch is not hit for types that cannot fit 1E4 (u8)");
                    let quad = remain % scale;
                    remain /= scale;
                    let pair1 = (quad / 100) as usize;
                    let pair2 = (quad % 100) as usize;
                    buf[offset + 0].write(DEC_DIGITS_LUT[pair1 * 2 + 0]);
                    buf[offset + 1].write(DEC_DIGITS_LUT[pair1 * 2 + 1]);
                    buf[offset + 2].write(DEC_DIGITS_LUT[pair2 * 2 + 0]);
                    buf[offset + 3].write(DEC_DIGITS_LUT[pair2 * 2 + 1]);
                }

                // Format per two digits from the lookup table.
                if remain > 9 {
                    // SAFETY: All of the decimals fit in buf due to MAX_DEC_N
                    // and the if condition ensures at least 2 more decimals.
                    unsafe { core::hint::assert_unchecked(offset >= 2) }
                    // SAFETY: The offset counts down from its initial buf.len()
                    // without underflow due to the previous precondition.
                    unsafe { core::hint::assert_unchecked(offset <= buf.len()) }
                    offset -= 2;

                    let pair = (remain % 100) as usize;
                    remain /= 100;
                    buf[offset + 0].write(DEC_DIGITS_LUT[pair * 2 + 0]);
                    buf[offset + 1].write(DEC_DIGITS_LUT[pair * 2 + 1]);
                }

                // Format the last remaining digit, if any.
                if remain != 0 || self == 0 {
                    // SAFETY: All of the decimals fit in buf due to MAX_DEC_N
                    // and the if condition ensures (at least) 1 more decimals.
                    unsafe { core::hint::assert_unchecked(offset >= 1) }
                    // SAFETY: The offset counts down from its initial buf.len()
                    // without underflow due to the previous precondition.
                    unsafe { core::hint::assert_unchecked(offset <= buf.len()) }
                    offset -= 1;

                    // Either the compiler sees that remain < 10, or it prevents
                    // a boundary check up next.
                    let last = (remain & 15) as usize;
                    buf[offset].write(DEC_DIGITS_LUT[last * 2 + 1]);
                    // not used: remain = 0;
                }

                // SAFETY: All buf content since offset is set.
                let written = unsafe { buf.get_unchecked(offset..) };
                // SAFETY: Writes use ASCII from the lookup table exclusively.
                unsafe {
                    str::from_utf8_unchecked(slice::from_raw_parts(
                          MaybeUninit::slice_as_ptr(written),
                          written.len(),
                    ))
                }
            }
        })*

        #[cfg(feature = "optimize_for_size")]
        fn $gen_name(mut n: $u, is_nonnegative: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            const MAX_DEC_N: usize = $u::MAX.ilog10() as usize + 1;
            let mut buf = [MaybeUninit::<u8>::uninit(); MAX_DEC_N];
            let mut curr = MAX_DEC_N;
            let buf_ptr = MaybeUninit::slice_as_mut_ptr(&mut buf);

            // SAFETY: To show that it's OK to copy into `buf_ptr`, notice that at the beginning
            // `curr == buf.len() == 39 > log(n)` since `n < 2^128 < 10^39`, and at
            // each step this is kept the same as `n` is divided. Since `n` is always
            // non-negative, this means that `curr > 0` so `buf_ptr[curr..curr + 1]`
            // is safe to access.
            unsafe {
                loop {
                    curr -= 1;
                    buf_ptr.add(curr).write((n % 10) as u8 + b'0');
                    n /= 10;

                    if n == 0 {
                        break;
                    }
                }
            }

            // SAFETY: `curr` > 0 (since we made `buf` large enough), and all the chars are valid UTF-8
            let buf_slice = unsafe {
                str::from_utf8_unchecked(
                    slice::from_raw_parts(buf_ptr.add(curr), buf.len() - curr))
            };
            f.pad_integral(is_nonnegative, "", buf_slice)
        }
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
                    None => (0, 0)
                };
                for _ in 1..subtracted_precision {
                    n /= 10;
                    exponent += 1;
                }
                if subtracted_precision != 0 {
                    let rem = n % 10;
                    n /= 10;
                    exponent += 1;
                    // round up last digit, round to even on a tie
                    if rem > 5 || (rem == 5 && (n % 2 != 0 || subtracted_precision > 1 )) {
                        n += 1;
                        // if the digit is rounded to the next power
                        // instead adjust the exponent
                        if n.ilog10() > (n - 1).ilog10() {
                            n /= 10;
                            exponent += 1;
                        }
                    }
                }
                (n, exponent, exponent, added_precision)
            };

            // Since `curr` always decreases by the number of digits copied, this means
            // that `curr >= 0`.
            let mut buf = [MaybeUninit::<u8>::uninit(); 40];
            let mut curr = buf.len(); //index for buf
            let buf_ptr = MaybeUninit::slice_as_mut_ptr(&mut buf);
            let lut_ptr = DEC_DIGITS_LUT.as_ptr();

            // decode 2 chars at a time
            while n >= 100 {
                let d1 = ((n % 100) as usize) << 1;
                curr -= 2;
                // SAFETY: `d1 <= 198`, so we can copy from `lut_ptr[d1..d1 + 2]` since
                // `DEC_DIGITS_LUT` has a length of 200.
                unsafe {
                    ptr::copy_nonoverlapping(lut_ptr.add(d1), buf_ptr.add(curr), 2);
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
                    *buf_ptr.add(curr) = (n as u8 % 10_u8) + b'0';
                }
                n /= 10;
                exponent += 1;
            }
            // add decimal point iff >1 mantissa digit will be printed
            if exponent != trailing_zeros || added_precision != 0 {
                curr -= 1;
                // SAFETY: Safe since `40 > curr >= 0`
                unsafe {
                    *buf_ptr.add(curr) = b'.';
                }
            }

            // SAFETY: Safe since `40 > curr >= 0`
            let buf_slice = unsafe {
                // decode last character
                curr -= 1;
                *buf_ptr.add(curr) = (n as u8) + b'0';

                let len = buf.len() - curr as usize;
                slice::from_raw_parts(buf_ptr.add(curr), len)
            };

            // stores 'e' (or 'E') and the up to 2-digit exponent
            let mut exp_buf = [MaybeUninit::<u8>::uninit(); 3];
            let exp_ptr = MaybeUninit::slice_as_mut_ptr(&mut exp_buf);
            // SAFETY: In either case, `exp_buf` is written within bounds and `exp_ptr[..len]`
            // is contained within `exp_buf` since `len <= 3`.
            let exp_slice = unsafe {
                *exp_ptr.add(0) = if upper { b'E' } else { b'e' };
                let len = if exponent < 10 {
                    *exp_ptr.add(1) = (exponent as u8) + b'0';
                    2
                } else {
                    let off = exponent << 1;
                    ptr::copy_nonoverlapping(lut_ptr.add(off), exp_ptr.add(1), 2);
                    3
                };
                slice::from_raw_parts(exp_ptr, len)
            };

            let parts = &[
                numfmt::Part::Copy(buf_slice),
                numfmt::Part::Zero(added_precision),
                numfmt::Part::Copy(exp_slice),
            ];
            let sign = if !is_nonnegative {
                "-"
            } else if f.sign_plus() {
                "+"
            } else {
                ""
            };
            let formatted = numfmt::Formatted { sign, parts };
            // SAFETY: `buf_slice` and `exp_slice` contain only ASCII characters.
            unsafe { f.pad_formatted_parts(&formatted) }
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
                        // convert the negative num to positive by summing 1 to its 2s complement
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
                        // convert the negative num to positive by summing 1 to its 2s complement
                        (!self.$conv_fn()).wrapping_add(1)
                    };
                    $name(n, is_nonnegative, true, f)
                }
            })*
    };
}

impl_Debug! {
    i8 i16 i32 i64 i128 isize
    u8 u16 u32 u64 u128 usize
}

// Include wasm32 in here since it doesn't reflect the native pointer size, and
// often cares strongly about getting a smaller code size.
#[cfg(any(target_pointer_width = "64", target_arch = "wasm32"))]
mod imp {
    use super::*;
    impl_Display!(
        i8, u8,
        i16, u16,
        i32, u32,
        i64, u64,
        isize, usize,
        ; as u64 via to_u64 named fmt_u64
    );
    impl_Exp!(
        i8, u8, i16, u16, i32, u32, i64, u64, usize, isize
            as u64 via to_u64 named exp_u64
    );
}

#[cfg(not(any(target_pointer_width = "64", target_arch = "wasm32")))]
mod imp {
    use super::*;
    impl_Display!(
        i8, u8,
        i16, u16,
        i32, u32,
        isize, usize,
        ; as u32 via to_u32 named fmt_u32);
    impl_Display!(
        i64, u64,
        ; as u64 via to_u64 named fmt_u64);

    impl_Exp!(i8, u8, i16, u16, i32, u32, isize, usize as u32 via to_u32 named exp_u32);
    impl_Exp!(i64, u64 as u64 via to_u64 named exp_u64);
}
impl_Exp!(i128, u128 as u128 via to_u128 named exp_u128);

const U128_MAX_DEC_N: usize = u128::MAX.ilog10() as usize + 1;

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for u128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::<u8>::uninit(); U128_MAX_DEC_N];

        f.pad_integral(true, "", self._fmt(&mut buf))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for i128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This is not a typo, we use the maximum number of digits of `u128`, hence why we use
        // `U128_MAX_DEC_N`.
        let mut buf = [MaybeUninit::<u8>::uninit(); U128_MAX_DEC_N];

        let is_nonnegative = *self >= 0;
        f.pad_integral(is_nonnegative, "", self.unsigned_abs()._fmt(&mut buf))
    }
}

impl u128 {
    /// Format optimized for u128. Computation of 128 bits is limited by proccessing
    /// in batches of 16 decimals at a time.
    #[doc(hidden)]
    #[unstable(
        feature = "fmt_internals",
        reason = "specialized method meant to only be used by `SpecToString` implementation",
        issue = "none"
    )]
    pub fn _fmt<'a>(self, buf: &'a mut [MaybeUninit<u8>]) -> &'a str {
        // Optimize common-case zero, which would also need special treatment due to
        // its "leading" zero.
        if self == 0 {
            return "0";
        }

        // Take the 16 least-significant decimals.
        let (quot_1e16, mod_1e16) = div_rem_1e16(self);
        let (mut remain, mut offset) = if quot_1e16 == 0 {
            (mod_1e16, U128_MAX_DEC_N)
        } else {
            // Write digits at buf[23..39].
            enc_16lsd::<{ U128_MAX_DEC_N - 16 }>(buf, mod_1e16);

            // Take another 16 decimals.
            let (quot2, mod2) = div_rem_1e16(quot_1e16);
            if quot2 == 0 {
                (mod2, U128_MAX_DEC_N - 16)
            } else {
                // Write digits at buf[7..23].
                enc_16lsd::<{ U128_MAX_DEC_N - 32 }>(buf, mod2);
                // Quot2 has at most 7 decimals remaining after two 1e16 divisions.
                (quot2 as u64, U128_MAX_DEC_N - 32)
            }
        };

        // Format per four digits from the lookup table.
        while remain > 999 {
            // SAFETY: All of the decimals fit in buf due to U128_MAX_DEC_N
            // and the while condition ensures at least 4 more decimals.
            unsafe { core::hint::assert_unchecked(offset >= 4) }
            // SAFETY: The offset counts down from its initial buf.len()
            // without underflow due to the previous precondition.
            unsafe { core::hint::assert_unchecked(offset <= buf.len()) }
            offset -= 4;

            // pull two pairs
            let quad = remain % 1_00_00;
            remain /= 1_00_00;
            let pair1 = (quad / 100) as usize;
            let pair2 = (quad % 100) as usize;
            buf[offset + 0].write(DEC_DIGITS_LUT[pair1 * 2 + 0]);
            buf[offset + 1].write(DEC_DIGITS_LUT[pair1 * 2 + 1]);
            buf[offset + 2].write(DEC_DIGITS_LUT[pair2 * 2 + 0]);
            buf[offset + 3].write(DEC_DIGITS_LUT[pair2 * 2 + 1]);
        }

        // Format per two digits from the lookup table.
        if remain > 9 {
            // SAFETY: All of the decimals fit in buf due to U128_MAX_DEC_N
            // and the if condition ensures at least 2 more decimals.
            unsafe { core::hint::assert_unchecked(offset >= 2) }
            // SAFETY: The offset counts down from its initial buf.len()
            // without underflow due to the previous precondition.
            unsafe { core::hint::assert_unchecked(offset <= buf.len()) }
            offset -= 2;

            let pair = (remain % 100) as usize;
            remain /= 100;
            buf[offset + 0].write(DEC_DIGITS_LUT[pair * 2 + 0]);
            buf[offset + 1].write(DEC_DIGITS_LUT[pair * 2 + 1]);
        }

        // Format the last remaining digit, if any.
        if remain != 0 {
            // SAFETY: All of the decimals fit in buf due to U128_MAX_DEC_N
            // and the if condition ensures (at least) 1 more decimals.
            unsafe { core::hint::assert_unchecked(offset >= 1) }
            // SAFETY: The offset counts down from its initial buf.len()
            // without underflow due to the previous precondition.
            unsafe { core::hint::assert_unchecked(offset <= buf.len()) }
            offset -= 1;

            // Either the compiler sees that remain < 10, or it prevents
            // a boundary check up next.
            let last = (remain & 15) as usize;
            buf[offset].write(DEC_DIGITS_LUT[last * 2 + 1]);
            // not used: remain = 0;
        }

        // SAFETY: All buf content since offset is set.
        let written = unsafe { buf.get_unchecked(offset..) };
        // SAFETY: Writes use ASCII from the lookup table exclusively.
        unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(
                MaybeUninit::slice_as_ptr(written),
                written.len(),
            ))
        }
    }
}

/// Encodes the 16 least-significant decimals of n into `buf[OFFSET .. OFFSET +
/// 16 ]`.
fn enc_16lsd<const OFFSET: usize>(buf: &mut [MaybeUninit<u8>], n: u64) {
    // Consume the least-significant decimals from a working copy.
    let mut remain = n;

    // Format per four digits from the lookup table.
    for quad_index in (0..4).rev() {
        // pull two pairs
        let quad = remain % 1_00_00;
        remain /= 1_00_00;
        let pair1 = (quad / 100) as usize;
        let pair2 = (quad % 100) as usize;
        buf[quad_index * 4 + OFFSET + 0].write(DEC_DIGITS_LUT[pair1 * 2 + 0]);
        buf[quad_index * 4 + OFFSET + 1].write(DEC_DIGITS_LUT[pair1 * 2 + 1]);
        buf[quad_index * 4 + OFFSET + 2].write(DEC_DIGITS_LUT[pair2 * 2 + 0]);
        buf[quad_index * 4 + OFFSET + 3].write(DEC_DIGITS_LUT[pair2 * 2 + 1]);
    }
}

/// Euclidean division plus remainder with constant 1E16 basically consumes 16
/// decimals from n.
///
/// The integer division algorithm is based on the following paper:
///
///   T. Granlund and P. Montgomery, “Division by Invariant Integers Using Multiplication”
///   in Proc. of the SIGPLAN94 Conference on Programming Language Design and
///   Implementation, 1994, pp. 61–72
///
#[inline]
fn div_rem_1e16(n: u128) -> (u128, u64) {
    const D: u128 = 1_0000_0000_0000_0000;
    // The check inlines well with the caller flow.
    if n < D {
        return (0, n as u64);
    }

    // These constant values are computed with the CHOOSE_MULTIPLIER procedure
    // from the Granlund & Montgomery paper, using N=128, prec=128 and d=1E16.
    const M_HIGH: u128 = 76624777043294442917917351357515459181;
    const SH_POST: u8 = 51;

    let quot = n.widening_mul(M_HIGH).1 >> SH_POST;
    let rem = n - quot * D;
    (quot, rem as u64)
}
