// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679

#![allow(missing_docs)]

use self::ExponentFormat::*;
use self::SignificantDigits::*;
use self::SignFormat::*;

use char::{self, CharExt};
use num::{self, Int, Float, ToPrimitive};
use num::FpCategory as Fp;
use ops::FnMut;
use slice::SliceExt;
use str::StrExt;
use string::String;
use vec::Vec;

/// A flag that specifies whether to use exponential (scientific) notation.
#[derive(Copy)]
pub enum ExponentFormat {
    /// Do not use exponential notation.
    ExpNone,
    /// Use exponential notation with the exponent having a base of 10 and the
    /// exponent sign being `e` or `E`. For example, 1000 would be printed
    /// 1e3.
    ExpDec,
    /// Use exponential notation with the exponent having a base of 2 and the
    /// exponent sign being `p` or `P`. For example, 8 would be printed 1p3.
    ExpBin,
}

/// The number of digits used for emitting the fractional part of a number, if
/// any.
#[derive(Copy)]
pub enum SignificantDigits {
    /// All calculable digits will be printed.
    ///
    /// Note that bignums or fractions may cause a surprisingly large number
    /// of digits to be printed.
    DigAll,

    /// At most the given number of digits will be printed, truncating any
    /// trailing zeroes.
    DigMax(uint),

    /// Precisely the given number of digits will be printed.
    DigExact(uint)
}

/// How to emit the sign of a number.
#[derive(Copy)]
pub enum SignFormat {
    /// No sign will be printed. The exponent sign will also be emitted.
    SignNone,
    /// `-` will be printed for negative values, but no sign will be emitted
    /// for positive numbers.
    SignNeg,
    /// `+` will be printed for positive values, and `-` will be printed for
    /// negative values.
    SignAll,
}

/// Converts an integral number to its string representation as a byte vector.
/// This is meant to be a common base implementation for all integral string
/// conversion functions like `to_string()` or `to_str_radix()`.
///
/// # Arguments
///
/// - `num`           - The number to convert. Accepts any number that
///                     implements the numeric traits.
/// - `radix`         - Base to use. Accepts only the values 2-36.
/// - `sign`          - How to emit the sign. Options are:
///     - `SignNone`: No sign at all. Basically emits `abs(num)`.
///     - `SignNeg`:  Only `-` on negative values.
///     - `SignAll`:  Both `+` on positive, and `-` on negative numbers.
/// - `f`             - a callback which will be invoked for each ascii character
///                     which composes the string representation of this integer
///
/// # Panics
///
/// - Panics if `radix` < 2 or `radix` > 36.
fn int_to_str_bytes_common<T, F>(num: T, radix: uint, sign: SignFormat, mut f: F) where
    T: Int,
    F: FnMut(u8),
{
    assert!(2 <= radix && radix <= 36);

    let _0: T = Int::zero();

    let neg = num < _0;
    let radix_gen: T = num::cast(radix).unwrap();

    let mut deccum = num;
    // This is just for integral types, the largest of which is a u64. The
    // smallest base that we can have is 2, so the most number of digits we're
    // ever going to have is 64
    let mut buf = [0u8; 64];
    let mut cur = 0;

    // Loop at least once to make sure at least a `0` gets emitted.
    loop {
        // Calculate the absolute value of each digit instead of only
        // doing it once for the whole number because a
        // representable negative number doesn't necessary have an
        // representable additive inverse of the same type
        // (See twos complement). But we assume that for the
        // numbers [-35 .. 0] we always have [0 .. 35].
        let current_digit_signed = deccum % radix_gen;
        let current_digit = if current_digit_signed < _0 {
            _0 - current_digit_signed
        } else {
            current_digit_signed
        };
        buf[cur] = match current_digit.to_u8().unwrap() {
            i @ 0...9 => b'0' + i,
            i         => b'a' + (i - 10),
        };
        cur += 1;

        deccum = deccum / radix_gen;
        // No more digits to calculate for the non-fractional part -> break
        if deccum == _0 { break; }
    }

    // Decide what sign to put in front
    match sign {
        SignNeg | SignAll if neg => { f(b'-'); }
        SignAll => { f(b'+'); }
        _ => ()
    }

    // We built the number in reverse order, so un-reverse it here
    while cur > 0 {
        cur -= 1;
        f(buf[cur]);
    }
}

/// Converts a number to its string representation as a byte vector.
/// This is meant to be a common base implementation for all numeric string
/// conversion functions like `to_string()` or `to_str_radix()`.
///
/// # Arguments
///
/// - `num`           - The number to convert. Accepts any number that
///                     implements the numeric traits.
/// - `radix`         - Base to use. Accepts only the values 2-36. If the exponential notation
///                     is used, then this base is only used for the significand. The exponent
///                     itself always printed using a base of 10.
/// - `negative_zero` - Whether to treat the special value `-0` as
///                     `-0` or as `+0`.
/// - `sign`          - How to emit the sign. See `SignFormat`.
/// - `digits`        - The amount of digits to use for emitting the fractional
///                     part, if any. See `SignificantDigits`.
/// - `exp_format`   - Whether or not to use the exponential (scientific) notation.
///                    See `ExponentFormat`.
/// - `exp_capital`   - Whether or not to use a capital letter for the exponent sign, if
///                     exponential notation is desired.
///
/// # Return value
///
/// A tuple containing the byte vector, and a boolean flag indicating
/// whether it represents a special value like `inf`, `-inf`, `NaN` or not.
/// It returns a tuple because there can be ambiguity between a special value
/// and a number representation at higher bases.
///
/// # Panics
///
/// - Panics if `radix` < 2 or `radix` > 36.
/// - Panics if `radix` > 14 and `exp_format` is `ExpDec` due to conflict
///   between digit and exponent sign `'e'`.
/// - Panics if `radix` > 25 and `exp_format` is `ExpBin` due to conflict
///   between digit and exponent sign `'p'`.
pub fn float_to_str_bytes_common<T: Float>(
        num: T, radix: u32, negative_zero: bool,
        sign: SignFormat, digits: SignificantDigits, exp_format: ExponentFormat, exp_upper: bool
        ) -> (Vec<u8>, bool) {
    assert!(2 <= radix && radix <= 36);
    match exp_format {
        ExpDec if radix >= DIGIT_E_RADIX       // decimal exponent 'e'
          => panic!("float_to_str_bytes_common: radix {} incompatible with \
                    use of 'e' as decimal exponent", radix),
        ExpBin if radix >= DIGIT_P_RADIX       // binary exponent 'p'
          => panic!("float_to_str_bytes_common: radix {} incompatible with \
                    use of 'p' as binary exponent", radix),
        _ => ()
    }

    let _0: T = Float::zero();
    let _1: T = Float::one();

    match num.classify() {
        Fp::Nan => { return (b"NaN".to_vec(), true); }
        Fp::Infinite if num > _0 => {
            return match sign {
                SignAll => (b"+inf".to_vec(), true),
                _       => (b"inf".to_vec(), true)
            };
        }
        Fp::Infinite if num < _0 => {
            return match sign {
                SignNone => (b"inf".to_vec(), true),
                _        => (b"-inf".to_vec(), true),
            };
        }
        _ => {}
    }

    let neg = num < _0 || (negative_zero && _1 / num == Float::neg_infinity());
    let mut buf = Vec::new();
    let radix_gen: T = num::cast(radix as int).unwrap();

    let (num, exp) = match exp_format {
        ExpNone => (num, 0i32),
        ExpDec | ExpBin => {
            if num == _0 {
                (num, 0i32)
            } else {
                let (exp, exp_base) = match exp_format {
                    ExpDec => (num.abs().log10().floor(), num::cast::<f64, T>(10.0f64).unwrap()),
                    ExpBin => (num.abs().log2().floor(), num::cast::<f64, T>(2.0f64).unwrap()),
                    ExpNone => unreachable!()
                };

                (num / exp_base.powf(exp), num::cast::<T, i32>(exp).unwrap())
            }
        }
    };

    // First emit the non-fractional part, looping at least once to make
    // sure at least a `0` gets emitted.
    let mut deccum = num.trunc();
    loop {
        // Calculate the absolute value of each digit instead of only
        // doing it once for the whole number because a
        // representable negative number doesn't necessary have an
        // representable additive inverse of the same type
        // (See twos complement). But we assume that for the
        // numbers [-35 .. 0] we always have [0 .. 35].
        let current_digit = (deccum % radix_gen).abs();

        // Decrease the deccumulator one digit at a time
        deccum = deccum / radix_gen;
        deccum = deccum.trunc();

        buf.push(char::from_digit(current_digit.to_int().unwrap() as u32, radix)
             .unwrap() as u8);

        // No more digits to calculate for the non-fractional part -> break
        if deccum == _0 { break; }
    }

    // If limited digits, calculate one digit more for rounding.
    let (limit_digits, digit_count, exact) = match digits {
        DigAll          => (false, 0,       false),
        DigMax(count)   => (true,  count+1, false),
        DigExact(count) => (true,  count+1, true)
    };

    // Decide what sign to put in front
    match sign {
        SignNeg | SignAll if neg => {
            buf.push(b'-');
        }
        SignAll => {
            buf.push(b'+');
        }
        _ => ()
    }

    buf.reverse();

    // Remember start of the fractional digits.
    // Points one beyond end of buf if none get generated,
    // or at the '.' otherwise.
    let start_fractional_digits = buf.len();

    // Now emit the fractional part, if any
    deccum = num.fract();
    if deccum != _0 || (limit_digits && exact && digit_count > 0) {
        buf.push(b'.');
        let mut dig = 0;

        // calculate new digits while
        // - there is no limit and there are digits left
        // - or there is a limit, it's not reached yet and
        //   - it's exact
        //   - or it's a maximum, and there are still digits left
        while (!limit_digits && deccum != _0)
           || (limit_digits && dig < digit_count && (
                   exact
                || (!exact && deccum != _0)
              )
        ) {
            // Shift first fractional digit into the integer part
            deccum = deccum * radix_gen;

            // Calculate the absolute value of each digit.
            // See note in first loop.
            let current_digit = deccum.trunc().abs();

            buf.push(char::from_digit(
                current_digit.to_int().unwrap() as u32, radix).unwrap() as u8);

            // Decrease the deccumulator one fractional digit at a time
            deccum = deccum.fract();
            dig += 1;
        }

        // If digits are limited, and that limit has been reached,
        // cut off the one extra digit, and depending on its value
        // round the remaining ones.
        if limit_digits && dig == digit_count {
            let ascii2value = |chr: u8| {
                (chr as char).to_digit(radix).unwrap()
            };
            let value2ascii = |val: u32| {
                char::from_digit(val, radix).unwrap() as u8
            };

            let extra_digit = ascii2value(buf.pop().unwrap());
            if extra_digit >= radix / 2 { // -> need to round
                let mut i: int = buf.len() as int - 1;
                loop {
                    // If reached left end of number, have to
                    // insert additional digit:
                    if i < 0
                    || buf[i as uint] == b'-'
                    || buf[i as uint] == b'+' {
                        buf.insert((i + 1) as uint, value2ascii(1));
                        break;
                    }

                    // Skip the '.'
                    if buf[i as uint] == b'.' { i -= 1; continue; }

                    // Either increment the digit,
                    // or set to 0 if max and carry the 1.
                    let current_digit = ascii2value(buf[i as uint]);
                    if current_digit < (radix - 1) {
                        buf[i as uint] = value2ascii(current_digit+1);
                        break;
                    } else {
                        buf[i as uint] = value2ascii(0);
                        i -= 1;
                    }
                }
            }
        }
    }

    // if number of digits is not exact, remove all trailing '0's up to
    // and including the '.'
    if !exact {
        let buf_max_i = buf.len() - 1;

        // index to truncate from
        let mut i = buf_max_i;

        // discover trailing zeros of fractional part
        while i > start_fractional_digits && buf[i] == b'0' {
            i -= 1;
        }

        // Only attempt to truncate digits if buf has fractional digits
        if i >= start_fractional_digits {
            // If buf ends with '.', cut that too.
            if buf[i] == b'.' { i -= 1 }

            // only resize buf if we actually remove digits
            if i < buf_max_i {
                buf = buf[.. (i + 1)].to_vec();
            }
        }
    } // If exact and trailing '.', just cut that
    else {
        let max_i = buf.len() - 1;
        if buf[max_i] == b'.' {
            buf = buf[.. max_i].to_vec();
        }
    }

    match exp_format {
        ExpNone => (),
        _ => {
            buf.push(match exp_format {
                ExpDec if exp_upper => 'E',
                ExpDec if !exp_upper => 'e',
                ExpBin if exp_upper => 'P',
                ExpBin if !exp_upper => 'p',
                _ => unreachable!()
            } as u8);

            int_to_str_bytes_common(exp, 10, sign, |c| buf.push(c));
        }
    }

    (buf, false)
}

/// Converts a number to its string representation. This is a wrapper for
/// `to_str_bytes_common()`, for details see there.
#[inline]
pub fn float_to_str_common<T: Float>(
        num: T, radix: u32, negative_zero: bool,
        sign: SignFormat, digits: SignificantDigits, exp_format: ExponentFormat, exp_capital: bool
        ) -> (String, bool) {
    let (bytes, special) = float_to_str_bytes_common(num, radix,
                               negative_zero, sign, digits, exp_format, exp_capital);
    (String::from_utf8(bytes).unwrap(), special)
}

// Some constants for from_str_bytes_common's input validation,
// they define minimum radix values for which the character is a valid digit.
static DIGIT_P_RADIX: u32 = ('p' as u32) - ('a' as u32) + 11;
static DIGIT_E_RADIX: u32 = ('e' as u32) - ('a' as u32) + 11;

#[cfg(test)]
mod tests {
    use string::ToString;

    #[test]
    fn test_int_to_str_overflow() {
        let mut i8_val: i8 = 127_i8;
        assert_eq!(i8_val.to_string(), "127");

        i8_val += 1 as i8;
        assert_eq!(i8_val.to_string(), "-128");

        let mut i16_val: i16 = 32_767_i16;
        assert_eq!(i16_val.to_string(), "32767");

        i16_val += 1 as i16;
        assert_eq!(i16_val.to_string(), "-32768");

        let mut i32_val: i32 = 2_147_483_647_i32;
        assert_eq!(i32_val.to_string(), "2147483647");

        i32_val += 1 as i32;
        assert_eq!(i32_val.to_string(), "-2147483648");

        let mut i64_val: i64 = 9_223_372_036_854_775_807_i64;
        assert_eq!(i64_val.to_string(), "9223372036854775807");

        i64_val += 1 as i64;
        assert_eq!(i64_val.to_string(), "-9223372036854775808");
    }
}

#[cfg(test)]
mod bench {
    #![allow(deprecated)] // rand
    extern crate test;

    mod uint {
        use super::test::Bencher;
        use rand::{weak_rng, Rng};
        use std::fmt;

        #[inline]
        fn to_string(x: uint, base: u8) {
            format!("{}", fmt::radix(x, base));
        }

        #[bench]
        fn to_str_bin(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<uint>(), 2); })
        }

        #[bench]
        fn to_str_oct(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<uint>(), 8); })
        }

        #[bench]
        fn to_str_dec(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<uint>(), 10); })
        }

        #[bench]
        fn to_str_hex(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<uint>(), 16); })
        }

        #[bench]
        fn to_str_base_36(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<uint>(), 36); })
        }
    }

    mod int {
        use super::test::Bencher;
        use rand::{weak_rng, Rng};
        use std::fmt;

        #[inline]
        fn to_string(x: int, base: u8) {
            format!("{}", fmt::radix(x, base));
        }

        #[bench]
        fn to_str_bin(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<int>(), 2); })
        }

        #[bench]
        fn to_str_oct(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<int>(), 8); })
        }

        #[bench]
        fn to_str_dec(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<int>(), 10); })
        }

        #[bench]
        fn to_str_hex(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<int>(), 16); })
        }

        #[bench]
        fn to_str_base_36(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { to_string(rng.gen::<int>(), 36); })
        }
    }

    mod f64 {
        use super::test::Bencher;
        use rand::{weak_rng, Rng};
        use f64;

        #[bench]
        fn float_to_string(b: &mut Bencher) {
            let mut rng = weak_rng();
            b.iter(|| { f64::to_string(rng.gen()); })
        }
    }
}
