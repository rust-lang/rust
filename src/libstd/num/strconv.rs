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

#![allow(missing_doc)]

use char;
use clone::Clone;
use collections::{Collection, MutableSeq};
use num::{NumCast, Zero, One, cast, Int};
use num::{Float, FPNaN, FPInfinite, ToPrimitive};
use num;
use ops::{Add, Sub, Mul, Div, Rem, Neg};
use option::{None, Option, Some};
use slice::{ImmutableSlice, MutableSlice};
use std::cmp::{PartialOrd, PartialEq};
use str::StrSlice;
use string::String;
use vec::Vec;

/// A flag that specifies whether to use exponential (scientific) notation.
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

/// Encompasses functions used by the string converter.
pub trait NumStrConv {
    /// Returns the NaN value.
    fn nan()      -> Option<Self>;

    /// Returns the infinite value.
    fn inf()      -> Option<Self>;

    /// Returns the negative infinite value.
    fn neg_inf()  -> Option<Self>;

    /// Returns -0.0.
    fn neg_zero() -> Option<Self>;

    /// Rounds the number toward zero.
    fn round_to_zero(&self)   -> Self;

    /// Returns the fractional part of the number.
    fn fractional_part(&self) -> Self;
}

macro_rules! impl_NumStrConv_Floating (($t:ty) => (
    impl NumStrConv for $t {
        #[inline]
        fn nan()      -> Option<$t> { Some( 0.0 / 0.0) }
        #[inline]
        fn inf()      -> Option<$t> { Some( 1.0 / 0.0) }
        #[inline]
        fn neg_inf()  -> Option<$t> { Some(-1.0 / 0.0) }
        #[inline]
        fn neg_zero() -> Option<$t> { Some(-0.0      ) }

        #[inline]
        fn round_to_zero(&self) -> $t { self.trunc() }
        #[inline]
        fn fractional_part(&self) -> $t { self.fract() }
    }
))

macro_rules! impl_NumStrConv_Integer (($t:ty) => (
    impl NumStrConv for $t {
        #[inline] fn nan()      -> Option<$t> { None }
        #[inline] fn inf()      -> Option<$t> { None }
        #[inline] fn neg_inf()  -> Option<$t> { None }
        #[inline] fn neg_zero() -> Option<$t> { None }

        #[inline] fn round_to_zero(&self)   -> $t { *self }
        #[inline] fn fractional_part(&self) -> $t {     0 }
    }
))

// FIXME: #4955
// Replace by two generic impls for traits 'Integral' and 'Floating'
impl_NumStrConv_Floating!(f32)
impl_NumStrConv_Floating!(f64)

impl_NumStrConv_Integer!(int)
impl_NumStrConv_Integer!(i8)
impl_NumStrConv_Integer!(i16)
impl_NumStrConv_Integer!(i32)
impl_NumStrConv_Integer!(i64)

impl_NumStrConv_Integer!(uint)
impl_NumStrConv_Integer!(u8)
impl_NumStrConv_Integer!(u16)
impl_NumStrConv_Integer!(u32)
impl_NumStrConv_Integer!(u64)


// Special value strings as [u8] consts.
static INF_BUF:     [u8, ..3] = [b'i', b'n', b'f'];
static POS_INF_BUF: [u8, ..4] = [b'+', b'i', b'n', b'f'];
static NEG_INF_BUF: [u8, ..4] = [b'-', b'i', b'n', b'f'];
static NAN_BUF:     [u8, ..3] = [b'N', b'a', b'N'];

/**
 * Converts an integral number to its string representation as a byte vector.
 * This is meant to be a common base implementation for all integral string
 * conversion functions like `to_string()` or `to_str_radix()`.
 *
 * # Arguments
 * - `num`           - The number to convert. Accepts any number that
 *                     implements the numeric traits.
 * - `radix`         - Base to use. Accepts only the values 2-36.
 * - `sign`          - How to emit the sign. Options are:
 *     - `SignNone`: No sign at all. Basically emits `abs(num)`.
 *     - `SignNeg`:  Only `-` on negative values.
 *     - `SignAll`:  Both `+` on positive, and `-` on negative numbers.
 * - `f`             - a callback which will be invoked for each ascii character
 *                     which composes the string representation of this integer
 *
 * # Return value
 * A tuple containing the byte vector, and a boolean flag indicating
 * whether it represents a special value like `inf`, `-inf`, `NaN` or not.
 * It returns a tuple because there can be ambiguity between a special value
 * and a number representation at higher bases.
 *
 * # Failure
 * - Fails if `radix` < 2 or `radix` > 36.
 */
#[deprecated = "format!() and friends should be favored instead"]
pub fn int_to_str_bytes_common<T: Int>(num: T, radix: uint, sign: SignFormat, f: |u8|) {
    assert!(2 <= radix && radix <= 36);

    let _0: T = Zero::zero();

    let neg = num < _0;
    let radix_gen: T = cast(radix).unwrap();

    let mut deccum = num;
    // This is just for integral types, the largest of which is a u64. The
    // smallest base that we can have is 2, so the most number of digits we're
    // ever going to have is 64
    let mut buf = [0u8, ..64];
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
            -current_digit_signed
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

/**
 * Converts a number to its string representation as a byte vector.
 * This is meant to be a common base implementation for all numeric string
 * conversion functions like `to_string()` or `to_str_radix()`.
 *
 * # Arguments
 * - `num`           - The number to convert. Accepts any number that
 *                     implements the numeric traits.
 * - `radix`         - Base to use. Accepts only the values 2-36. If the exponential notation
 *                     is used, then this base is only used for the significand. The exponent
 *                     itself always printed using a base of 10.
 * - `negative_zero` - Whether to treat the special value `-0` as
 *                     `-0` or as `+0`.
 * - `sign`          - How to emit the sign. See `SignFormat`.
 * - `digits`        - The amount of digits to use for emitting the fractional
 *                     part, if any. See `SignificantDigits`.
 * - `exp_format`   - Whether or not to use the exponential (scientific) notation.
 *                    See `ExponentFormat`.
 * - `exp_capital`   - Whether or not to use a capital letter for the exponent sign, if
 *                     exponential notation is desired.
 *
 * # Return value
 * A tuple containing the byte vector, and a boolean flag indicating
 * whether it represents a special value like `inf`, `-inf`, `NaN` or not.
 * It returns a tuple because there can be ambiguity between a special value
 * and a number representation at higher bases.
 *
 * # Failure
 * - Fails if `radix` < 2 or `radix` > 36.
 * - Fails if `radix` > 14 and `exp_format` is `ExpDec` due to conflict
 *   between digit and exponent sign `'e'`.
 * - Fails if `radix` > 25 and `exp_format` is `ExpBin` due to conflict
 *   between digit and exponent sign `'p'`.
 */
#[allow(deprecated)]
pub fn float_to_str_bytes_common<T:NumCast+Zero+One+PartialEq+PartialOrd+Float+
                                  Div<T,T>+Neg<T>+Rem<T,T>+Mul<T,T>>(
        num: T, radix: uint, negative_zero: bool,
        sign: SignFormat, digits: SignificantDigits, exp_format: ExponentFormat, exp_upper: bool
        ) -> (Vec<u8>, bool) {
    assert!(2 <= radix && radix <= 36);
    match exp_format {
        ExpDec if radix >= DIGIT_E_RADIX       // decimal exponent 'e'
          => fail!("float_to_str_bytes_common: radix {} incompatible with \
                    use of 'e' as decimal exponent", radix),
        ExpBin if radix >= DIGIT_P_RADIX       // binary exponent 'p'
          => fail!("float_to_str_bytes_common: radix {} incompatible with \
                    use of 'p' as binary exponent", radix),
        _ => ()
    }

    let _0: T = Zero::zero();
    let _1: T = One::one();

    match num.classify() {
        FPNaN => { return (Vec::from_slice("NaN".as_bytes()), true); }
        FPInfinite if num > _0 => {
            return match sign {
                SignAll => (Vec::from_slice("+inf".as_bytes()), true),
                _       => (Vec::from_slice("inf".as_bytes()), true)
            };
        }
        FPInfinite if num < _0 => {
            return match sign {
                SignNone => (Vec::from_slice("inf".as_bytes()), true),
                _        => (Vec::from_slice("-inf".as_bytes()), true),
            };
        }
        _ => {}
    }

    let neg = num < _0 || (negative_zero && _1 / num == Float::neg_infinity());
    let mut buf = Vec::new();
    let radix_gen: T   = cast(radix as int).unwrap();

    let (num, exp) = match exp_format {
        ExpNone => (num, 0i32),
        ExpDec | ExpBin => {
            if num == _0 {
                (num, 0i32)
            } else {
                let (exp, exp_base) = match exp_format {
                    ExpDec => (num.abs().log10().floor(), cast::<f64, T>(10.0f64).unwrap()),
                    ExpBin => (num.abs().log2().floor(), cast::<f64, T>(2.0f64).unwrap()),
                    ExpNone => unreachable!()
                };

                (num / exp_base.powf(exp), cast::<T, i32>(exp).unwrap())
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

        buf.push(char::from_digit(current_digit.to_int().unwrap() as uint, radix)
             .unwrap() as u8);

        // No more digits to calculate for the non-fractional part -> break
        if deccum == _0 { break; }
    }

    // If limited digits, calculate one digit more for rounding.
    let (limit_digits, digit_count, exact) = match digits {
        DigAll          => (false, 0u,      false),
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
        let mut dig = 0u;

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
                current_digit.to_int().unwrap() as uint, radix).unwrap() as u8);

            // Decrease the deccumulator one fractional digit at a time
            deccum = deccum.fract();
            dig += 1u;
        }

        // If digits are limited, and that limit has been reached,
        // cut off the one extra digit, and depending on its value
        // round the remaining ones.
        if limit_digits && dig == digit_count {
            let ascii2value = |chr: u8| {
                char::to_digit(chr as char, radix).unwrap()
            };
            let value2ascii = |val: uint| {
                char::from_digit(val, radix).unwrap() as u8
            };

            let extra_digit = ascii2value(buf.pop().unwrap());
            if extra_digit >= radix / 2 { // -> need to round
                let mut i: int = buf.len() as int - 1;
                loop {
                    // If reached left end of number, have to
                    // insert additional digit:
                    if i < 0
                    || *buf.get(i as uint) == b'-'
                    || *buf.get(i as uint) == b'+' {
                        buf.insert((i + 1) as uint, value2ascii(1));
                        break;
                    }

                    // Skip the '.'
                    if *buf.get(i as uint) == b'.' { i -= 1; continue; }

                    // Either increment the digit,
                    // or set to 0 if max and carry the 1.
                    let current_digit = ascii2value(*buf.get(i as uint));
                    if current_digit < (radix - 1) {
                        *buf.get_mut(i as uint) = value2ascii(current_digit+1);
                        break;
                    } else {
                        *buf.get_mut(i as uint) = value2ascii(0);
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
        while i > start_fractional_digits && *buf.get(i) == b'0' {
            i -= 1;
        }

        // Only attempt to truncate digits if buf has fractional digits
        if i >= start_fractional_digits {
            // If buf ends with '.', cut that too.
            if *buf.get(i) == b'.' { i -= 1 }

            // only resize buf if we actually remove digits
            if i < buf_max_i {
                buf = Vec::from_slice(buf.slice(0, i + 1));
            }
        }
    } // If exact and trailing '.', just cut that
    else {
        let max_i = buf.len() - 1;
        if *buf.get(max_i) == b'.' {
            buf = Vec::from_slice(buf.slice(0, max_i));
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

/**
 * Converts a number to its string representation. This is a wrapper for
 * `to_str_bytes_common()`, for details see there.
 */
#[inline]
pub fn float_to_str_common<T:NumCast+Zero+One+PartialEq+PartialOrd+NumStrConv+Float+
                             Div<T,T>+Neg<T>+Rem<T,T>+Mul<T,T>>(
        num: T, radix: uint, negative_zero: bool,
        sign: SignFormat, digits: SignificantDigits, exp_format: ExponentFormat, exp_capital: bool
        ) -> (String, bool) {
    let (bytes, special) = float_to_str_bytes_common(num, radix,
                               negative_zero, sign, digits, exp_format, exp_capital);
    (String::from_utf8(bytes).unwrap(), special)
}

// Some constants for from_str_bytes_common's input validation,
// they define minimum radix values for which the character is a valid digit.
static DIGIT_P_RADIX: uint = ('p' as uint) - ('a' as uint) + 11u;
static DIGIT_I_RADIX: uint = ('i' as uint) - ('a' as uint) + 11u;
static DIGIT_E_RADIX: uint = ('e' as uint) - ('a' as uint) + 11u;

/**
 * Parses a byte slice as a number. This is meant to
 * be a common base implementation for all numeric string conversion
 * functions like `from_str()` or `from_str_radix()`.
 *
 * # Arguments
 * - `buf`        - The byte slice to parse.
 * - `radix`      - Which base to parse the number as. Accepts 2-36.
 * - `negative`   - Whether to accept negative numbers.
 * - `fractional` - Whether to accept numbers with fractional parts.
 * - `special`    - Whether to accept special values like `inf`
 *                  and `NaN`. Can conflict with `radix`, see Failure.
 * - `exponent`   - Which exponent format to accept. Options are:
 *     - `ExpNone`: No Exponent, accepts just plain numbers like `42` or
 *                  `-8.2`.
 *     - `ExpDec`:  Accepts numbers with a decimal exponent like `42e5` or
 *                  `8.2E-2`. The exponent string itself is always base 10.
 *                  Can conflict with `radix`, see Failure.
 *     - `ExpBin`:  Accepts numbers with a binary exponent like `42P-8` or
 *                  `FFp128`. The exponent string itself is always base 10.
 *                  Can conflict with `radix`, see Failure.
 * - `empty_zero` - Whether to accept an empty `buf` as a 0 or not.
 * - `ignore_underscores` - Whether all underscores within the string should
 *                          be ignored.
 *
 * # Return value
 * Returns `Some(n)` if `buf` parses to a number n without overflowing, and
 * `None` otherwise, depending on the constraints set by the remaining
 * arguments.
 *
 * # Failure
 * - Fails if `radix` < 2 or `radix` > 36.
 * - Fails if `radix` > 14 and `exponent` is `ExpDec` due to conflict
 *   between digit and exponent sign `'e'`.
 * - Fails if `radix` > 25 and `exponent` is `ExpBin` due to conflict
 *   between digit and exponent sign `'p'`.
 * - Fails if `radix` > 18 and `special == true` due to conflict
 *   between digit and lowest first character in `inf` and `NaN`, the `'i'`.
 */
pub fn from_str_bytes_common<T:NumCast+Zero+One+PartialEq+PartialOrd+Div<T,T>+
                                    Mul<T,T>+Sub<T,T>+Neg<T>+Add<T,T>+
                                    NumStrConv+Clone>(
        buf: &[u8], radix: uint, negative: bool, fractional: bool,
        special: bool, exponent: ExponentFormat, empty_zero: bool,
        ignore_underscores: bool
        ) -> Option<T> {
    match exponent {
        ExpDec if radix >= DIGIT_E_RADIX       // decimal exponent 'e'
          => fail!("from_str_bytes_common: radix {} incompatible with \
                    use of 'e' as decimal exponent", radix),
        ExpBin if radix >= DIGIT_P_RADIX       // binary exponent 'p'
          => fail!("from_str_bytes_common: radix {} incompatible with \
                    use of 'p' as binary exponent", radix),
        _ if special && radix >= DIGIT_I_RADIX // first digit of 'inf'
          => fail!("from_str_bytes_common: radix {} incompatible with \
                    special values 'inf' and 'NaN'", radix),
        _ if (radix as int) < 2
          => fail!("from_str_bytes_common: radix {} to low, \
                    must lie in the range [2, 36]", radix),
        _ if (radix as int) > 36
          => fail!("from_str_bytes_common: radix {} to high, \
                    must lie in the range [2, 36]", radix),
        _ => ()
    }

    let _0: T = Zero::zero();
    let _1: T = One::one();
    let radix_gen: T = cast(radix as int).unwrap();

    let len = buf.len();

    if len == 0 {
        if empty_zero {
            return Some(_0);
        } else {
            return None;
        }
    }

    if special {
        if buf == INF_BUF || buf == POS_INF_BUF {
            return NumStrConv::inf();
        } else if buf == NEG_INF_BUF {
            if negative {
                return NumStrConv::neg_inf();
            } else {
                return None;
            }
        } else if buf == NAN_BUF {
            return NumStrConv::nan();
        }
    }

    let (start, accum_positive) = match buf[0] as char {
      '-' if !negative => return None,
      '-' => (1u, false),
      '+' => (1u, true),
       _  => (0u, true)
    };

    // Initialize accumulator with signed zero for floating point parsing to
    // work
    let mut accum      = if accum_positive { _0.clone() } else { -_1 * _0};
    let mut last_accum = accum.clone(); // Necessary to detect overflow
    let mut i          = start;
    let mut exp_found  = false;

    // Parse integer part of number
    while i < len {
        let c = buf[i] as char;

        match char::to_digit(c, radix) {
            Some(digit) => {
                // shift accum one digit left
                accum = accum * radix_gen.clone();

                // add/subtract current digit depending on sign
                if accum_positive {
                    accum = accum + cast(digit as int).unwrap();
                } else {
                    accum = accum - cast(digit as int).unwrap();
                }

                // Detect overflow by comparing to last value, except
                // if we've not seen any non-zero digits.
                if last_accum != _0 {
                    if accum_positive && accum <= last_accum { return NumStrConv::inf(); }
                    if !accum_positive && accum >= last_accum { return NumStrConv::neg_inf(); }

                    // Detect overflow by reversing the shift-and-add process
                    if accum_positive &&
                        (last_accum != ((accum - cast(digit as int).unwrap())/radix_gen.clone())) {
                        return NumStrConv::inf();
                    }
                    if !accum_positive &&
                        (last_accum != ((accum + cast(digit as int).unwrap())/radix_gen.clone())) {
                        return NumStrConv::neg_inf();
                    }
                }
                last_accum = accum.clone();
            }
            None => match c {
                '_' if ignore_underscores => {}
                'e' | 'E' | 'p' | 'P' => {
                    exp_found = true;
                    break;                       // start of exponent
                }
                '.' if fractional => {
                    i += 1u;                     // skip the '.'
                    break;                       // start of fractional part
                }
                _ => return None                 // invalid number
            }
        }

        i += 1u;
    }

    // Parse fractional part of number
    // Skip if already reached start of exponent
    if !exp_found {
        let mut power = _1.clone();

        while i < len {
            let c = buf[i] as char;

            match char::to_digit(c, radix) {
                Some(digit) => {
                    // Decrease power one order of magnitude
                    power = power / radix_gen;

                    let digit_t: T = cast(digit).unwrap();

                    // add/subtract current digit depending on sign
                    if accum_positive {
                        accum = accum + digit_t * power;
                    } else {
                        accum = accum - digit_t * power;
                    }

                    // Detect overflow by comparing to last value
                    if accum_positive && accum < last_accum { return NumStrConv::inf(); }
                    if !accum_positive && accum > last_accum { return NumStrConv::neg_inf(); }
                    last_accum = accum.clone();
                }
                None => match c {
                    '_' if ignore_underscores => {}
                    'e' | 'E' | 'p' | 'P' => {
                        exp_found = true;
                        break;                   // start of exponent
                    }
                    _ => return None             // invalid number
                }
            }

            i += 1u;
        }
    }

    // Special case: buf not empty, but does not contain any digit in front
    // of the exponent sign -> number is empty string
    if i == start {
        if empty_zero {
            return Some(_0);
        } else {
            return None;
        }
    }

    let mut multiplier = _1.clone();

    if exp_found {
        let c = buf[i] as char;
        let base: T = match (c, exponent) {
            // c is never _ so don't need to handle specially
            ('e', ExpDec) | ('E', ExpDec) => cast(10u).unwrap(),
            ('p', ExpBin) | ('P', ExpBin) => cast(2u).unwrap(),
            _ => return None // char doesn't fit given exponent format
        };

        // parse remaining bytes as decimal integer,
        // skipping the exponent char
        let exp: Option<int> = from_str_bytes_common(
            buf[i+1..len], 10, true, false, false, ExpNone, false,
            ignore_underscores);

        match exp {
            Some(exp_pow) => {
                multiplier = if exp_pow < 0 {
                    _1 / num::pow(base, (-exp_pow.to_int().unwrap()) as uint)
                } else {
                    num::pow(base, exp_pow.to_int().unwrap() as uint)
                }
            }
            None => return None // invalid exponent -> invalid number
        }
    }

    Some(accum * multiplier)
}

/**
 * Parses a string as a number. This is a wrapper for
 * `from_str_bytes_common()`, for details see there.
 */
#[inline]
pub fn from_str_common<T:NumCast+Zero+One+PartialEq+PartialOrd+Div<T,T>+Mul<T,T>+
                              Sub<T,T>+Neg<T>+Add<T,T>+NumStrConv+Clone>(
        buf: &str, radix: uint, negative: bool, fractional: bool,
        special: bool, exponent: ExponentFormat, empty_zero: bool,
        ignore_underscores: bool
        ) -> Option<T> {
    from_str_bytes_common(buf.as_bytes(), radix, negative,
                          fractional, special, exponent, empty_zero,
                          ignore_underscores)
}

#[cfg(test)]
mod test {
    use super::*;
    use option::*;

    #[test]
    fn from_str_ignore_underscores() {
        let s : Option<u8> = from_str_common("__1__", 2, false, false, false,
                                             ExpNone, false, true);
        assert_eq!(s, Some(1u8));

        let n : Option<u8> = from_str_common("__1__", 2, false, false, false,
                                             ExpNone, false, false);
        assert_eq!(n, None);

        let f : Option<f32> = from_str_common("_1_._5_e_1_", 10, false, true, false,
                                              ExpDec, false, true);
        assert_eq!(f, Some(1.5e1f32));
    }

    #[test]
    fn from_str_issue5770() {
        // try to parse 0b1_1111_1111 = 511 as a u8. Caused problems
        // since 255*2+1 == 255 (mod 256) so the overflow wasn't
        // detected.
        let n : Option<u8> = from_str_common("111111111", 2, false, false, false,
                                             ExpNone, false, false);
        assert_eq!(n, None);
    }

    #[test]
    fn from_str_issue7588() {
        let u : Option<u8> = from_str_common("1000", 10, false, false, false,
                                            ExpNone, false, false);
        assert_eq!(u, None);
        let s : Option<i16> = from_str_common("80000", 10, false, false, false,
                                             ExpNone, false, false);
        assert_eq!(s, None);
        let f : Option<f32> = from_str_common(
            "10000000000000000000000000000000000000000", 10, false, false, false,
            ExpNone, false, false);
        assert_eq!(f, NumStrConv::inf())
        let fe : Option<f32> = from_str_common("1e40", 10, false, false, false,
                                            ExpDec, false, false);
        assert_eq!(fe, NumStrConv::inf())
    }
}

#[cfg(test)]
mod bench {
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
