//! Integer and floating-point number formatting

use crate::fmt::NumBuffer;
use crate::mem::MaybeUninit;
use crate::num::fmt as numfmt;
use crate::{fmt, str};

/// Formatting of integers with a non-decimal radix.
macro_rules! radix_integer {
    (fmt::$Trait:ident for $Signed:ident and $Unsigned:ident, $prefix:literal, $dig_tab:literal) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::$Trait for $Unsigned {
            /// Format unsigned integers in the radix.
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // Check macro arguments at compile time.
                const {
                    assert!($Unsigned::MIN == 0, "need unsigned");
                    assert!($dig_tab.is_ascii(), "need single-byte entries");
                }

                // ASCII digits in ascending order are used as a lookup table.
                const DIG_TAB: &[u8] = $dig_tab;
                const BASE: $Unsigned = DIG_TAB.len() as $Unsigned;
                const MAX_DIG_N: usize = $Unsigned::MAX.ilog(BASE) as usize + 1;

                // Buffer digits of self with right alignment.
                let mut buf = [MaybeUninit::<u8>::uninit(); MAX_DIG_N];
                // Count the number of bytes in buf that are not initialized.
                let mut offset = buf.len();

                // Accumulate each digit of the number from the least
                // significant to the most significant figure.
                let mut remain = *self;
                loop {
                    let digit = remain % BASE;
                    remain /= BASE;

                    offset -= 1;
                    // SAFETY: `remain` will reach 0 and we will break before `offset` wraps
                    unsafe { core::hint::assert_unchecked(offset < buf.len()) }
                    buf[offset].write(DIG_TAB[digit as usize]);
                    if remain == 0 {
                        break;
                    }
                }

                // SAFETY: Starting from `offset`, all elements of the slice have been set.
                let digits = unsafe { slice_buffer_to_str(&buf, offset) };
                f.pad_integral(true, $prefix, digits)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::$Trait for $Signed {
            /// Format signed integers in the two’s-complement form.
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::$Trait::fmt(&self.cast_unsigned(), f)
            }
        }
    };
}

/// Formatting of integers with a non-decimal radix.
macro_rules! radix_integers {
    ($Signed:ident, $Unsigned:ident) => {
        radix_integer! { fmt::Binary   for $Signed and $Unsigned, "0b", b"01" }
        radix_integer! { fmt::Octal    for $Signed and $Unsigned, "0o", b"01234567" }
        radix_integer! { fmt::LowerHex for $Signed and $Unsigned, "0x", b"0123456789abcdef" }
        radix_integer! { fmt::UpperHex for $Signed and $Unsigned, "0x", b"0123456789ABCDEF" }
    };
}
radix_integers! { isize, usize }
radix_integers! { i8, u8 }
radix_integers! { i16, u16 }
radix_integers! { i32, u32 }
radix_integers! { i64, u64 }
radix_integers! { i128, u128 }

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

// The string of all two-digit numbers in range 00..99 is used as a lookup table.
static DECIMAL_PAIRS: &[u8; 200] = b"\
      0001020304050607080910111213141516171819\
      2021222324252627282930313233343536373839\
      4041424344454647484950515253545556575859\
      6061626364656667686970717273747576777879\
      8081828384858687888990919293949596979899";

/// This function converts a slice of ascii characters into a `&str` starting from `offset`.
///
/// # Safety
///
/// `buf` content starting from `offset` index MUST BE initialized and MUST BE ascii
/// characters.
unsafe fn slice_buffer_to_str(buf: &[MaybeUninit<u8>], offset: usize) -> &str {
    // SAFETY: `offset` is always included between 0 and `buf`'s length.
    let written = unsafe { buf.get_unchecked(offset..) };
    // SAFETY: (`assume_init_ref`) All buf content since offset is set.
    // SAFETY: (`from_utf8_unchecked`) Writes use ASCII from the lookup table exclusively.
    unsafe { str::from_utf8_unchecked(written.assume_init_ref()) }
}

macro_rules! impl_Display {
    ($($Signed:ident, $Unsigned:ident),* ; as $T:ident into $fmt_fn:ident) => {

        $(
        const _: () = {
            assert!($Signed::MIN < 0, "need signed");
            assert!($Unsigned::MIN == 0, "need unsigned");
            assert!($Signed::BITS == $Unsigned::BITS, "need counterparts");
            assert!($Signed::BITS <= $T::BITS, "need lossless conversion");
            assert!($Unsigned::BITS <= $T::BITS, "need lossless conversion");
        };

        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::Display for $Unsigned {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                #[cfg(not(feature = "optimize_for_size"))]
                {
                    const MAX_DEC_N: usize = $Unsigned::MAX.ilog10() as usize + 1;
                    // Buffer decimals for self with right alignment.
                    let mut buf = [MaybeUninit::<u8>::uninit(); MAX_DEC_N];

                    // SAFETY: `buf` is always big enough to contain all the digits.
                    unsafe { f.pad_integral(true, "", self._fmt(&mut buf)) }
                }
                #[cfg(feature = "optimize_for_size")]
                {
                    // Lossless conversion (with as) is asserted at the top of
                    // this macro.
                    ${concat($fmt_fn, _small)}(*self as $T, true, f)
                }
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl fmt::Display for $Signed {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                #[cfg(not(feature = "optimize_for_size"))]
                {
                    const MAX_DEC_N: usize = $Unsigned::MAX.ilog10() as usize + 1;
                    // Buffer decimals for self with right alignment.
                    let mut buf = [MaybeUninit::<u8>::uninit(); MAX_DEC_N];

                    // SAFETY: `buf` is always big enough to contain all the digits.
                    unsafe { f.pad_integral(*self >= 0, "", self.unsigned_abs()._fmt(&mut buf)) }
                }
                #[cfg(feature = "optimize_for_size")]
                {
                    // Lossless conversion (with as) is asserted at the top of
                    // this macro.
                    return ${concat($fmt_fn, _small)}(self.unsigned_abs() as $T, *self >= 0, f);
                }
            }
        }

        #[cfg(not(feature = "optimize_for_size"))]
        impl $Unsigned {
            #[doc(hidden)]
            #[unstable(
                feature = "fmt_internals",
                reason = "specialized method meant to only be used by `SpecToString` implementation",
                issue = "none"
            )]
            pub unsafe fn _fmt<'a>(self, buf: &'a mut [MaybeUninit::<u8>]) -> &'a str {
                // SAFETY: `buf` will always be big enough to contain all digits.
                let offset = unsafe { self._fmt_inner(buf) };
                // SAFETY: Starting from `offset`, all elements of the slice have been set.
                unsafe { slice_buffer_to_str(buf, offset) }
            }

            unsafe fn _fmt_inner(self, buf: &mut [MaybeUninit::<u8>]) -> usize {
                // Count the number of bytes in buf that are not initialized.
                let mut offset = buf.len();
                // Consume the least-significant decimals from a working copy.
                let mut remain = self;

                // Format per four digits from the lookup table.
                // Four digits need a 16-bit $Unsigned or wider.
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
                    buf[offset + 0].write(DECIMAL_PAIRS[pair1 * 2 + 0]);
                    buf[offset + 1].write(DECIMAL_PAIRS[pair1 * 2 + 1]);
                    buf[offset + 2].write(DECIMAL_PAIRS[pair2 * 2 + 0]);
                    buf[offset + 3].write(DECIMAL_PAIRS[pair2 * 2 + 1]);
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
                    buf[offset + 0].write(DECIMAL_PAIRS[pair * 2 + 0]);
                    buf[offset + 1].write(DECIMAL_PAIRS[pair * 2 + 1]);
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
                    buf[offset].write(DECIMAL_PAIRS[last * 2 + 1]);
                    // not used: remain = 0;
                }

                offset
            }
        }

        impl $Signed {
            /// Allows users to write an integer (in signed decimal format) into a variable `buf` of
            /// type [`NumBuffer`] that is passed by the caller by mutable reference.
            ///
            /// # Examples
            ///
            /// ```
            /// #![feature(int_format_into)]
            /// use core::fmt::NumBuffer;
            ///
            #[doc = concat!("let n = 0", stringify!($Signed), ";")]
            /// let mut buf = NumBuffer::new();
            /// assert_eq!(n.format_into(&mut buf), "0");
            ///
            #[doc = concat!("let n1 = 32", stringify!($Signed), ";")]
            /// assert_eq!(n1.format_into(&mut buf), "32");
            ///
            #[doc = concat!("let n2 = ", stringify!($Signed::MAX), ";")]
            #[doc = concat!("assert_eq!(n2.format_into(&mut buf), ", stringify!($Signed::MAX), ".to_string());")]
            /// ```
            #[unstable(feature = "int_format_into", issue = "138215")]
            pub fn format_into(self, buf: &mut NumBuffer<Self>) -> &str {
                let mut offset;

                #[cfg(not(feature = "optimize_for_size"))]
                // SAFETY: `buf` will always be big enough to contain all digits.
                unsafe {
                    offset = self.unsigned_abs()._fmt_inner(&mut buf.buf);
                }
                #[cfg(feature = "optimize_for_size")]
                {
                    // Lossless conversion (with as) is asserted at the top of
                    // this macro.
                    offset = ${concat($fmt_fn, _in_buf_small)}(self.unsigned_abs() as $T, &mut buf.buf);
                }
                // Only difference between signed and unsigned are these 4 lines.
                if self < 0 {
                    offset -= 1;
                    buf.buf[offset].write(b'-');
                }
                // SAFETY: Starting from `offset`, all elements of the slice have been set.
                unsafe { slice_buffer_to_str(&buf.buf, offset) }
            }
        }

        impl $Unsigned {
            /// Allows users to write an integer (in signed decimal format) into a variable `buf` of
            /// type [`NumBuffer`] that is passed by the caller by mutable reference.
            ///
            /// # Examples
            ///
            /// ```
            /// #![feature(int_format_into)]
            /// use core::fmt::NumBuffer;
            ///
            #[doc = concat!("let n = 0", stringify!($Unsigned), ";")]
            /// let mut buf = NumBuffer::new();
            /// assert_eq!(n.format_into(&mut buf), "0");
            ///
            #[doc = concat!("let n1 = 32", stringify!($Unsigned), ";")]
            /// assert_eq!(n1.format_into(&mut buf), "32");
            ///
            #[doc = concat!("let n2 = ", stringify!($Unsigned::MAX), ";")]
            #[doc = concat!("assert_eq!(n2.format_into(&mut buf), ", stringify!($Unsigned::MAX), ".to_string());")]
            /// ```
            #[unstable(feature = "int_format_into", issue = "138215")]
            pub fn format_into(self, buf: &mut NumBuffer<Self>) -> &str {
                let offset;

                #[cfg(not(feature = "optimize_for_size"))]
                // SAFETY: `buf` will always be big enough to contain all digits.
                unsafe {
                    offset = self._fmt_inner(&mut buf.buf);
                }
                #[cfg(feature = "optimize_for_size")]
                {
                    // Lossless conversion (with as) is asserted at the top of
                    // this macro.
                    offset = ${concat($fmt_fn, _in_buf_small)}(self as $T, &mut buf.buf);
                }
                // SAFETY: Starting from `offset`, all elements of the slice have been set.
                unsafe { slice_buffer_to_str(&buf.buf, offset) }
            }
        }

        )*

        #[cfg(feature = "optimize_for_size")]
        fn ${concat($fmt_fn, _in_buf_small)}(mut n: $T, buf: &mut [MaybeUninit::<u8>]) -> usize {
            let mut curr = buf.len();

            // SAFETY: To show that it's OK to copy into `buf_ptr`, notice that at the beginning
            // `curr == buf.len() == 39 > log(n)` since `n < 2^128 < 10^39`, and at
            // each step this is kept the same as `n` is divided. Since `n` is always
            // non-negative, this means that `curr > 0` so `buf_ptr[curr..curr + 1]`
            // is safe to access.
            loop {
                curr -= 1;
                buf[curr].write((n % 10) as u8 + b'0');
                n /= 10;

                if n == 0 {
                    break;
                }
            }
            curr
        }

        #[cfg(feature = "optimize_for_size")]
        fn ${concat($fmt_fn, _small)}(n: $T, is_nonnegative: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            const MAX_DEC_N: usize = $T::MAX.ilog(10) as usize + 1;
            let mut buf = [MaybeUninit::<u8>::uninit(); MAX_DEC_N];

            let offset = ${concat($fmt_fn, _in_buf_small)}(n, &mut buf);
            // SAFETY: Starting from `offset`, all elements of the slice have been set.
            let buf_slice = unsafe { slice_buffer_to_str(&buf, offset) };
            f.pad_integral(is_nonnegative, "", buf_slice)
        }
    };
}

macro_rules! impl_Exp {
    ($($Signed:ident, $Unsigned:ident),* ; as $T:ident into $fmt_fn:ident) => {
        const _: () = assert!($T::MIN == 0, "need unsigned");

        fn $fmt_fn(
            f: &mut fmt::Formatter<'_>,
            n: $T,
            is_nonnegative: bool,
            letter_e: u8
        ) -> fmt::Result {
            debug_assert!(letter_e.is_ascii_alphabetic(), "single-byte character");

            // Print the integer as a coefficient in range (-10, 10).
            let mut exp = n.checked_ilog10().unwrap_or(0) as usize;
            debug_assert!(n / (10 as $T).pow(exp as u32) < 10);

            // Precisison is counted as the number of digits in the fraction.
            let mut coef_prec = exp;
            // Keep the digits as an integer (paired with its coef_prec count).
            let mut coef = n;

            // A Formatter may set the precision to a fixed number of decimals.
            let more_prec = match f.precision() {
                None => {
                    // Omit any and all trailing zeroes.
                    while coef_prec != 0 && coef % 10 == 0 {
                        coef /= 10;
                        coef_prec -= 1;
                    }
                    0
                },

                Some(fmt_prec) if fmt_prec >= coef_prec => {
                    // Count the number of additional zeroes needed.
                    fmt_prec - coef_prec
                },

                Some(fmt_prec) => {
                    // Count the number of digits to drop.
                    let less_prec = coef_prec - fmt_prec;
                    assert!(less_prec > 0);
                    // Scale down the coefficient/precision pair. For example,
                    // coef 123456 gets coef_prec 5 (to make 1.23456). To format
                    // the number with 2 decimals, i.e., fmt_prec 2, coef should
                    // be scaled by 10⁵⁻²=1000 to get coef 123 with coef_prec 2.

                    // SAFETY: Any precision less than coef_prec will cause a
                    // power of ten below the coef value.
                    let scale = unsafe {
                        (10 as $T).checked_pow(less_prec as u32).unwrap_unchecked()
                    };
                    let floor = coef / scale;
                    // Round half to even conform documentation.
                    let over = coef % scale;
                    let half = scale / 2;
                    let round_up = if over < half {
                        0
                    } else if over > half {
                        1
                    } else {
                        floor & 1 // round odd up to even
                    };
                    // Adding one to a scale down of at least 10 won't overflow.
                    coef = floor + round_up;
                    coef_prec = fmt_prec;

                    // The round_up may have caused the coefficient to reach 10
                    // (which is not permitted). For example, anything in range
                    // [9.95, 10) becomes 10.0 when adjusted to precision 1.
                    if round_up != 0 && coef.checked_ilog10().unwrap_or(0) as usize > coef_prec {
                        debug_assert_eq!(coef, (10 as $T).pow(coef_prec as u32 + 1));
                        coef /= 10; // drop one trailing zero
                        exp += 1;   // one power of ten higher
                    }
                    0
                },
            };

            // Allocate a text buffer with lazy initialization.
            const MAX_DEC_N: usize = $T::MAX.ilog10() as usize + 1;
            const MAX_COEF_LEN: usize = MAX_DEC_N + ".".len();
            const MAX_TEXT_LEN: usize = MAX_COEF_LEN + "e99".len();
            let mut buf = [MaybeUninit::<u8>::uninit(); MAX_TEXT_LEN];

            // Encode the coefficient in buf[..coef_len].
            let (lead_dec, coef_len) = if coef_prec == 0 && more_prec == 0 {
                (coef, 1_usize) // single digit; no fraction
            } else {
                buf[1].write(b'.');
                let fraction_range = 2..(2 + coef_prec);

                // Consume the least-significant decimals from a working copy.
                let mut remain = coef;
                #[cfg(feature = "optimize_for_size")] {
                    for i in fraction_range.clone().rev() {
                        let digit = (remain % 10) as usize;
                        remain /= 10;
                        buf[i].write(b'0' + digit as u8);
                    }
                }
                #[cfg(not(feature = "optimize_for_size"))] {
                    // Write digits per two at a time with a lookup table.
                    for i in fraction_range.clone().skip(1).rev().step_by(2) {
                        let pair = (remain % 100) as usize;
                        remain /= 100;
                        buf[i - 1].write(DECIMAL_PAIRS[pair * 2 + 0]);
                        buf[i - 0].write(DECIMAL_PAIRS[pair * 2 + 1]);
                    }
                    // An odd number of digits leave one digit remaining.
                    if coef_prec & 1 != 0 {
                        let digit = (remain % 10) as usize;
                        remain /= 10;
                        buf[fraction_range.start].write(b'0' + digit as u8);
                    }
                }

                (remain, fraction_range.end)
            };
            debug_assert!(lead_dec < 10);
            debug_assert!(lead_dec != 0 || coef == 0, "significant digits only");
            buf[0].write(b'0' + lead_dec as u8);

            // SAFETY: The number of decimals is limited, captured by MAX.
            unsafe { core::hint::assert_unchecked(coef_len <= MAX_COEF_LEN) }
            // Encode the scale factor in buf[coef_len..text_len].
            buf[coef_len].write(letter_e);
            let text_len: usize = match exp {
                ..10 => {
                    buf[coef_len + 1].write(b'0' + exp as u8);
                    coef_len + 2
                },
                10..100 => {
                    #[cfg(feature = "optimize_for_size")] {
                        buf[coef_len + 1].write(b'0' + (exp / 10) as u8);
                        buf[coef_len + 2].write(b'0' + (exp % 10) as u8);
                    }
                    #[cfg(not(feature = "optimize_for_size"))] {
                        buf[coef_len + 1].write(DECIMAL_PAIRS[exp * 2 + 0]);
                        buf[coef_len + 2].write(DECIMAL_PAIRS[exp * 2 + 1]);
                    }
                    coef_len + 3
                },
                _ => {
                    const { assert!($T::MAX.ilog10() < 100) };
                    // SAFETY: A `u256::MAX` would get exponent 77.
                    unsafe { core::hint::unreachable_unchecked() }
                }
            };
            // SAFETY: All bytes up until text_len have been set.
            let text = unsafe { buf[..text_len].assume_init_ref() };

            if more_prec == 0 {
                // SAFETY: Text is set with ASCII exclusively: either a decimal,
                // or a LETTER_E, or a dot. ASCII implies valid UTF-8.
                let as_str = unsafe { str::from_utf8_unchecked(text) };
                f.pad_integral(is_nonnegative, "", as_str)
            } else {
                let parts = &[
                    numfmt::Part::Copy(&text[..coef_len]),
                    numfmt::Part::Zero(more_prec),
                    numfmt::Part::Copy(&text[coef_len..]),
                ];
                let sign = if !is_nonnegative {
                    "-"
                } else if f.sign_plus() {
                    "+"
                } else {
                    ""
                };
                // SAFETY: Text is set with ASCII exclusively: either a decimal,
                // or a LETTER_E, or a dot. ASCII implies valid UTF-8.
                unsafe { f.pad_formatted_parts(&numfmt::Formatted { sign, parts }) }
            }
        }

        $(
        const _: () = {
            assert!($Signed::MIN < 0, "need signed");
            assert!($Unsigned::MIN == 0, "need unsigned");
            assert!($Signed::BITS == $Unsigned::BITS, "need counterparts");
            assert!($Signed::BITS <= $T::BITS, "need lossless conversion");
            assert!($Unsigned::BITS <= $T::BITS, "need lossless conversion");
        };
        #[stable(feature = "integer_exp_format", since = "1.42.0")]
        impl fmt::LowerExp for $Signed {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $fmt_fn(f, self.unsigned_abs() as $T, *self >= 0, b'e')
            }
        }
        #[stable(feature = "integer_exp_format", since = "1.42.0")]
        impl fmt::LowerExp for $Unsigned {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $fmt_fn(f, *self as $T, true, b'e')
            }
        }
        #[stable(feature = "integer_exp_format", since = "1.42.0")]
        impl fmt::UpperExp for $Signed {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $fmt_fn(f, self.unsigned_abs() as $T, *self >= 0, b'E')
            }
        }
        #[stable(feature = "integer_exp_format", since = "1.42.0")]
        impl fmt::UpperExp for $Unsigned {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                $fmt_fn(f, *self as $T, true, b'E')
            }
        }
        )*

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
    impl_Display!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize; as u64 into display_u64);
    impl_Exp!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize; as u64 into exp_u64);
}

#[cfg(not(any(target_pointer_width = "64", target_arch = "wasm32")))]
mod imp {
    use super::*;
    impl_Display!(i8, u8, i16, u16, i32, u32, isize, usize; as u32 into display_u32);
    impl_Display!(i64, u64; as u64 into display_u64);

    impl_Exp!(i8, u8, i16, u16, i32, u32, isize, usize; as u32 into exp_u32);
    impl_Exp!(i64, u64; as u64 into exp_u64);
}
impl_Exp!(i128, u128; as u128 into exp_u128);

const U128_MAX_DEC_N: usize = u128::MAX.ilog10() as usize + 1;

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for u128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [MaybeUninit::<u8>::uninit(); U128_MAX_DEC_N];

        // SAFETY: `buf` is always big enough to contain all the digits.
        unsafe { f.pad_integral(true, "", self._fmt(&mut buf)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for i128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // This is not a typo, we use the maximum number of digits of `u128`, hence why we use
        // `U128_MAX_DEC_N`.
        let mut buf = [MaybeUninit::<u8>::uninit(); U128_MAX_DEC_N];

        let is_nonnegative = *self >= 0;
        // SAFETY: `buf` is always big enough to contain all the digits.
        unsafe { f.pad_integral(is_nonnegative, "", self.unsigned_abs()._fmt(&mut buf)) }
    }
}

impl u128 {
    /// Format optimized for u128. Computation of 128 bits is limited by processing
    /// in batches of 16 decimals at a time.
    #[doc(hidden)]
    #[unstable(
        feature = "fmt_internals",
        reason = "specialized method meant to only be used by `SpecToString` implementation",
        issue = "none"
    )]
    pub unsafe fn _fmt<'a>(self, buf: &'a mut [MaybeUninit<u8>]) -> &'a str {
        // SAFETY: `buf` will always be big enough to contain all digits.
        let offset = unsafe { self._fmt_inner(buf) };
        // SAFETY: Starting from `offset`, all elements of the slice have been set.
        unsafe { slice_buffer_to_str(buf, offset) }
    }

    unsafe fn _fmt_inner(self, buf: &mut [MaybeUninit<u8>]) -> usize {
        // Optimize common-case zero, which would also need special treatment due to
        // its "leading" zero.
        if self == 0 {
            let offset = buf.len() - 1;
            buf[offset].write(b'0');
            return offset;
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
            buf[offset + 0].write(DECIMAL_PAIRS[pair1 * 2 + 0]);
            buf[offset + 1].write(DECIMAL_PAIRS[pair1 * 2 + 1]);
            buf[offset + 2].write(DECIMAL_PAIRS[pair2 * 2 + 0]);
            buf[offset + 3].write(DECIMAL_PAIRS[pair2 * 2 + 1]);
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
            buf[offset + 0].write(DECIMAL_PAIRS[pair * 2 + 0]);
            buf[offset + 1].write(DECIMAL_PAIRS[pair * 2 + 1]);
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
            buf[offset].write(DECIMAL_PAIRS[last * 2 + 1]);
            // not used: remain = 0;
        }
        offset
    }

    /// Allows users to write an integer (in signed decimal format) into a variable `buf` of
    /// type [`NumBuffer`] that is passed by the caller by mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(int_format_into)]
    /// use core::fmt::NumBuffer;
    ///
    /// let n = 0u128;
    /// let mut buf = NumBuffer::new();
    /// assert_eq!(n.format_into(&mut buf), "0");
    ///
    /// let n1 = 32u128;
    /// let mut buf1 = NumBuffer::new();
    /// assert_eq!(n1.format_into(&mut buf1), "32");
    ///
    /// let n2 = u128::MAX;
    /// let mut buf2 = NumBuffer::new();
    /// assert_eq!(n2.format_into(&mut buf2), u128::MAX.to_string());
    /// ```
    #[unstable(feature = "int_format_into", issue = "138215")]
    pub fn format_into(self, buf: &mut NumBuffer<Self>) -> &str {
        let diff = buf.capacity() - U128_MAX_DEC_N;
        // FIXME: Once const generics are better, use `NumberBufferTrait::BUF_SIZE` as generic const
        // for `fmt_u128_inner`.
        //
        // In the meantime, we have to use a slice starting at index 1 and add 1 to the returned
        // offset to ensure the number is correctly generated at the end of the buffer.
        // SAFETY: `diff` will always be between 0 and its initial value.
        unsafe { self._fmt(buf.buf.get_unchecked_mut(diff..)) }
    }
}

impl i128 {
    /// Allows users to write an integer (in signed decimal format) into a variable `buf` of
    /// type [`NumBuffer`] that is passed by the caller by mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(int_format_into)]
    /// use core::fmt::NumBuffer;
    ///
    /// let n = 0i128;
    /// let mut buf = NumBuffer::new();
    /// assert_eq!(n.format_into(&mut buf), "0");
    ///
    /// let n1 = i128::MIN;
    /// assert_eq!(n1.format_into(&mut buf), i128::MIN.to_string());
    ///
    /// let n2 = i128::MAX;
    /// assert_eq!(n2.format_into(&mut buf), i128::MAX.to_string());
    /// ```
    #[unstable(feature = "int_format_into", issue = "138215")]
    pub fn format_into(self, buf: &mut NumBuffer<Self>) -> &str {
        let diff = buf.capacity() - U128_MAX_DEC_N;
        // FIXME: Once const generics are better, use `NumberBufferTrait::BUF_SIZE` as generic const
        // for `fmt_u128_inner`.
        //
        // In the meantime, we have to use a slice starting at index 1 and add 1 to the returned
        // offset to ensure the number is correctly generated at the end of the buffer.
        let mut offset =
            // SAFETY: `buf` will always be big enough to contain all digits.
            unsafe { self.unsigned_abs()._fmt_inner(buf.buf.get_unchecked_mut(diff..)) };
        // We put back the offset at the right position.
        offset += diff;
        // Only difference between signed and unsigned are these 4 lines.
        if self < 0 {
            offset -= 1;
            // SAFETY: `buf` will always be big enough to contain all digits plus the minus sign.
            unsafe {
                buf.buf.get_unchecked_mut(offset).write(b'-');
            }
        }
        // SAFETY: Starting from `offset`, all elements of the slice have been set.
        unsafe { slice_buffer_to_str(&buf.buf, offset) }
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
        buf[quad_index * 4 + OFFSET + 0].write(DECIMAL_PAIRS[pair1 * 2 + 0]);
        buf[quad_index * 4 + OFFSET + 1].write(DECIMAL_PAIRS[pair1 * 2 + 1]);
        buf[quad_index * 4 + OFFSET + 2].write(DECIMAL_PAIRS[pair2 * 2 + 0]);
        buf[quad_index * 4 + OFFSET + 3].write(DECIMAL_PAIRS[pair2 * 2 + 1]);
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
