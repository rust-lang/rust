// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An interface for numeric types
use core::cmp::{Ord, Eq};
use option::{None, Option, Some};
use char;
use str;
use kinds::Copy;
use vec;

pub trait Num {
    // FIXME: Trait composition. (#2616)
    pure fn add(&self, other: &Self) -> Self;
    pure fn sub(&self, other: &Self) -> Self;
    pure fn mul(&self, other: &Self) -> Self;
    pure fn div(&self, other: &Self) -> Self;
    pure fn modulo(&self, other: &Self) -> Self;
    pure fn neg(&self) -> Self;
}

pub trait IntConvertible {
    pure fn to_int(&self) -> int;
    static pure fn from_int(n: int) -> Self;
}

pub trait Zero {
    static pure fn zero() -> Self;
}

pub trait One {
    static pure fn one() -> Self;
}

pub trait Round {
    pure fn round(&self, mode: RoundMode) -> Self;

    pure fn floor(&self) -> Self;
    pure fn ceil(&self)  -> Self;
    pure fn fract(&self) -> Self;
}

/**
 * Cast a number the the enclosing type
 *
 * # Example
 *
 * ~~~
 * let twenty: f32 = num::cast(0x14);
 * assert twenty == 20f32;
 * ~~~
 */
#[inline(always)]
pub pure fn cast<T:NumCast, U:NumCast>(n: T) -> U {
    NumCast::from(n)
}

/**
 * An interface for generic numeric type casts
 */
pub trait NumCast {
    static pure fn from<T:NumCast>(n: T) -> Self;

    pure fn to_u8(&self) -> u8;
    pure fn to_u16(&self) -> u16;
    pure fn to_u32(&self) -> u32;
    pure fn to_u64(&self) -> u64;
    pure fn to_uint(&self) -> uint;

    pure fn to_i8(&self) -> i8;
    pure fn to_i16(&self) -> i16;
    pure fn to_i32(&self) -> i32;
    pure fn to_i64(&self) -> i64;
    pure fn to_int(&self) -> int;

    pure fn to_f32(&self) -> f32;
    pure fn to_f64(&self) -> f64;
    pure fn to_float(&self) -> float;
}

pub enum RoundMode {
    RoundDown,
    RoundUp,
    RoundToZero,
    RoundFromZero
}

pub trait ToStrRadix {
    pub pure fn to_str_radix(&self, radix: uint) -> ~str;
}

pub trait FromStrRadix {
    static pub pure fn from_str_radix(str: &str, radix: uint) -> Option<Self>;
}

// Generic math functions:

/// Dynamically calculates the value `inf` (`1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn infinity<T: Num One Zero>() -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();
    _1 / _0
}

/// Dynamically calculates the value `-inf` (`-1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn neg_infinity<T: Num One Zero>() -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();
    - _1 / _0
}

/// Dynamically calculates the value `NaN` (`0/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn NaN<T: Num Zero>() -> T {
    let _0: T = Zero::zero();
    _0 / _0
}

/// Returns `true` if `num` has the value `inf` (`1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn is_infinity<T: Num One Zero Eq>(num: &T) -> bool {
    (*num) == (infinity::<T>())
}

/// Returns `true` if `num` has the value `-inf` (`-1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn is_neg_infinity<T: Num One Zero Eq>(num: &T) -> bool {
    (*num) == (neg_infinity::<T>())
}

/// Returns `true` if `num` has the value `NaN` (is not equal to itself).
#[inline(always)]
pub pure fn is_NaN<T: Num Eq>(num: &T) -> bool {
    (*num) != (*num)
}

/// Returns `true` if `num` has the value `-0` (`1/num == -1/0`).
/// Can fail on integer types.
#[inline(always)]
pub pure fn is_neg_zero<T: Num One Zero Eq>(num: &T) -> bool {
    let _1: T = One::one();
    let _0: T = Zero::zero();
    *num == _0 && is_neg_infinity(&(_1 / *num))
}

/**
 * Calculates a power to a given radix, optimized for uint `pow` and `radix`.
 *
 * Returns `radix^pow` as `T`.
 *
 * Note:
 * Also returns `1` for `0^0`, despite that technically being an
 * undefined number. The reason for this is twofold:
 * - If code written to use this function cares about that special case, it's
 *   probably going to catch it before making the call.
 * - If code written to use this function doesn't care about it, it's
 *   probably assuming that `x^0` always equals `1`.
 */
pub pure fn pow_with_uint<T: Num NumCast One Zero Copy>(radix: uint,
                                                        pow: uint) -> T {
    let _0: T = Zero::zero();
    let _1: T = One::one();

    if pow   == 0u { return _1; }
    if radix == 0u { return _0; }
    let mut my_pow     = pow;
    let mut total      = _1;
    let mut multiplier = cast(radix as int);
    while (my_pow > 0u) {
        if my_pow % 2u == 1u {
            total *= multiplier;
        }
        my_pow     /= 2u;
        multiplier *= multiplier;
    }
    total
}

pub enum ExponentFormat {
    ExpNone,
    ExpDec,
    ExpBin
}

pub enum SignificantDigits {
    DigAll,
    DigMax(uint),
    DigExact(uint)
}

pub enum SignFormat {
    SignNone,
    SignNeg,
    SignAll
}

/**
 * Converts a number to its string representation as a byte vector.
 * This is meant to be a common base implementation for all numeric string
 * conversion functions like `to_str()` or `to_str_radix()`.
 *
 * # Arguments
 * - `num`           - The number to convert. Accepts any number that
 *                     implements the numeric traits.
 * - `radix`         - Base to use. Accepts only the values 2-36.
 * - `special`       - Whether to attempt to compare to special values like
 *                     `inf` or `NaN`. Also needed to detect negative 0.
 *                     Can fail if it doesn't match `num`s type
 *                     (see safety note).
 * - `negative_zero` - Whether to treat the special value `-0` as
 *                     `-0` or as `+0`.
 * - `sign`          - How to emit the sign. Options are:
 *     - `SignNone`: No sign at all. Basically emits `abs(num)`.
 *     - `SignNeg`:  Only `-` on negative values.
 *     - `SignAll`:  Both `+` on positive, and `-` on negative numbers.
 * - `digits`        - The amount of digits to use for emitting the
 *                     fractional part, if any. Options are:
 *     - `DigAll`:         All calculatable digits. Beware of bignums or
 *                         fractions!
 *     - `DigMax(uint)`:   Maximum N digits, truncating any trailing zeros.
 *     - `DigExact(uint)`: Exactly N digits.
 *
 * # Return value
 * A tuple containing the byte vector, and a boolean flag indicating
 * whether it represents a special value like `inf`, `-inf`, `NaN` or not.
 * It returns a tuple because there can be ambiguity between a special value
 * and a number representation at higher bases.
 *
 * # Failure
 * - Fails if `radix` < 2 or `radix` > 36.
 * - Fails on wrong value for `special` (see safety note).
 *
 * # Safety note
 * The function detects the special values `inf`, `-inf` and `NaN` by
 * dynamically comparing `num` to `1 / 0`, `-1 / 0` and `0 / 0`
 * (each of type T) if `special` is `true`. This will fail on integer types
 * with a 'divide by zero'. Likewise, it will fail if `num` **is** one of
 * those special values, and `special` is `false`, because then the
 * algorithm just does normal calculations on them.
 */
pub pure fn to_str_bytes_common<T: Num NumCast Zero One Eq Ord Round Copy>(
        num: &T, radix: uint, special: bool, negative_zero: bool,
        sign: SignFormat, digits: SignificantDigits) -> (~[u8], bool) {
    if radix as int <  2 {
        die!(fmt!("to_str_bytes_common: radix %? to low, \
                   must lie in the range [2, 36]", radix));
    } else if radix as int > 36 {
        die!(fmt!("to_str_bytes_common: radix %? to high, \
                   must lie in the range [2, 36]", radix));
    }

    let _0: T = Zero::zero();
    let _1: T = One::one();

    if special {
        if is_NaN(num) {
            return (str::to_bytes("NaN"), true);
        } else if is_infinity(num){
            return match sign {
                SignAll => (str::to_bytes("+inf"), true),
                _       => (str::to_bytes("inf"), true)
            }
        } else if is_neg_infinity(num) {
            return match sign {
                SignNone => (str::to_bytes("inf"), true),
                _        => (str::to_bytes("-inf"), true),
            }
        }
    }

    let neg = *num < _0 || (negative_zero && *num == _0
                            && special && is_neg_zero(num));
    let mut buf: ~[u8] = ~[];
    let radix_gen: T   = cast(radix as int);

    let mut deccum;

    // First emit the non-fractional part, looping at least once to make
    // sure at least a `0` gets emitted.
    deccum = num.round(RoundToZero);
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

        // Decrease the deccumulator one digit at a time
        deccum /= radix_gen;
        deccum = deccum.round(RoundToZero);

        unsafe { // FIXME: Pureness workaround (#4568)
            buf.push(char::from_digit(current_digit.to_int() as uint, radix)
                 .unwrap() as u8);
        }

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
            unsafe { // FIXME: Pureness workaround (#4568)
                buf.push('-' as u8);
            }
        }
        SignAll => {
            unsafe { // FIXME: Pureness workaround (#4568)
                buf.push('+' as u8);
            }
        }
        _ => ()
    }

    unsafe { // FIXME: Pureness workaround (#4568)
        vec::reverse(buf);
    }

    // Remember start of the fractional digits.
    // Points one beyond end of buf if none get generated,
    // or at the '.' otherwise.
    let start_fractional_digits = buf.len();

    // Now emit the fractional part, if any
    deccum = num.fract();
    if deccum != _0 || (limit_digits && exact && digit_count > 0) {
        unsafe { // FIXME: Pureness workaround (#4568)
            buf.push('.' as u8);
        }
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
            deccum *= radix_gen;

            // Calculate the absolute value of each digit.
            // See note in first loop.
            let current_digit_signed = deccum.round(RoundToZero);
            let current_digit = if current_digit_signed < _0 {
                -current_digit_signed
            } else {
                current_digit_signed
            };

            unsafe { // FIXME: Pureness workaround (#4568)
                buf.push(char::from_digit(
                    current_digit.to_int() as uint, radix).unwrap() as u8);
            }

            // Decrease the deccumulator one fractional digit at a time
            deccum = deccum.fract();
            dig += 1u;
        }

        // If digits are limited, and that limit has been reached,
        // cut off the one extra digit, and depending on its value
        // round the remaining ones.
        if limit_digits && dig == digit_count {
            let ascii2value = |chr: u8| {
                char::to_digit(chr as char, radix).unwrap() as uint
            };
            let value2ascii = |val: uint| {
                char::from_digit(val, radix).unwrap() as u8
            };

            unsafe { // FIXME: Pureness workaround (#4568)
                let extra_digit = ascii2value(buf.pop());
                if extra_digit >= radix / 2 { // -> need to round
                    let mut i: int = buf.len() as int - 1;
                    loop {
                        // If reached left end of number, have to
                        // insert additional digit:
                        if i < 0
                        || buf[i] == '-' as u8
                        || buf[i] == '+' as u8 {
                            buf.insert((i + 1) as uint, value2ascii(1));
                            break;
                        }

                        // Skip the '.'
                        if buf[i] == '.' as u8 { i -= 1; loop; }

                        // Either increment the digit,
                        // or set to 0 if max and carry the 1.
                        let current_digit = ascii2value(buf[i]);
                        if current_digit < (radix - 1) {
                            buf[i] = value2ascii(current_digit+1);
                            break;
                        } else {
                            buf[i] = value2ascii(0);
                            i -= 1;
                        }
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
        while i > start_fractional_digits && buf[i] == '0' as u8 {
            i -= 1;
        }

        // Only attempt to truncate digits if buf has fractional digits
        if i >= start_fractional_digits {
            // If buf ends with '.', cut that too.
            if buf[i] == '.' as u8 { i -= 1 }

            // only resize buf if we actually remove digits
            if i < buf_max_i {
                buf = buf.slice(0, i + 1);
            }
        }
    } // If exact and trailing '.', just cut that
    else {
        let max_i = buf.len() - 1;
        if buf[max_i] == '.' as u8 {
            buf = buf.slice(0, max_i);
        }
    }

    (buf, false)
}

/**
 * Converts a number to its string representation. This is a wrapper for
 * `to_str_bytes_common()`, for details see there.
 */
#[inline(always)]
pub pure fn to_str_common<T: Num NumCast Zero One Eq Ord Round Copy>(
        num: &T, radix: uint, special: bool, negative_zero: bool,
        sign: SignFormat, digits: SignificantDigits) -> (~str, bool) {
    let (bytes, special) = to_str_bytes_common(num, radix, special,
                               negative_zero, sign, digits);
    (str::from_bytes(bytes), special)
}

// Some constants for from_str_bytes_common's input validation,
// they define minimum radix values for which the character is a valid digit.
priv const DIGIT_P_RADIX: uint = ('p' as uint) - ('a' as uint) + 11u;
priv const DIGIT_I_RADIX: uint = ('i' as uint) - ('a' as uint) + 11u;
priv const DIGIT_E_RADIX: uint = ('e' as uint) - ('a' as uint) + 11u;

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
 * - `empty_zero` - Whether to accept a empty `buf` as a 0 or not.
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
 *
 * # Possible improvements
 * - Could accept option to allow ignoring underscores, allowing for numbers
 *   formated like `FF_AE_FF_FF`.
 */
pub pure fn from_str_bytes_common<T: Num NumCast Zero One Ord Copy>(
        buf: &[u8], radix: uint, negative: bool, fractional: bool,
        special: bool, exponent: ExponentFormat, empty_zero: bool
        ) -> Option<T> {
    match exponent {
        ExpDec if radix >= DIGIT_E_RADIX       // decimal exponent 'e'
          => die!(fmt!("from_str_bytes_common: radix %? incompatible with \
                        use of 'e' as decimal exponent", radix)),
        ExpBin if radix >= DIGIT_P_RADIX       // binary exponent 'p'
          => die!(fmt!("from_str_bytes_common: radix %? incompatible with \
                        use of 'p' as binary exponent", radix)),
        _ if special && radix >= DIGIT_I_RADIX // first digit of 'inf'
          => die!(fmt!("from_str_bytes_common: radix %? incompatible with \
                        special values 'inf' and 'NaN'", radix)),
        _ if radix as int < 2
          => die!(fmt!("from_str_bytes_common: radix %? to low, \
                        must lie in the range [2, 36]", radix)),
        _ if radix as int > 36
          => die!(fmt!("from_str_bytes_common: radix %? to high, \
                        must lie in the range [2, 36]", radix)),
        _ => ()
    }

    let _0: T = Zero::zero();
    let _1: T = One::one();
    let radix_gen: T = cast(radix as int);

    let len = buf.len();

    if len == 0 {
        if empty_zero {
            return Some(_0);
        } else {
            return None;
        }
    }

    if special {
        if buf == str::to_bytes("inf") || buf == str::to_bytes("+inf") {
            return Some(infinity());
        } else if buf == str::to_bytes("-inf") {
            if negative {
                return Some(neg_infinity());
            } else {
                return None;
            }
        } else if buf == str::to_bytes("NaN") {
            return Some(NaN());
        }
    }

    let (start, accum_positive) = match buf[0] {
      '-' as u8 if !negative => return None,
      '-' as u8 => (1u, false),
      '+' as u8 => (1u, true),
       _        => (0u, true)
    };

    // Initialize accumulator with signed zero for floating point parsing to
    // work
    let mut accum      = if accum_positive { _0 } else { -_1 * _0};
    let mut last_accum = accum; // Necessary to detect overflow
    let mut i          = start;
    let mut exp_found  = false;

    // Parse integer part of number
    while i < len {
        let c = buf[i] as char;

        match char::to_digit(c, radix) {
            Some(digit) => {
                // shift accum one digit left
                accum *= radix_gen;

                // add/subtract current digit depending on sign
                if accum_positive {
                    accum += cast(digit as int);
                } else {
                    accum -= cast(digit as int);
                }

                // Detect overflow by comparing to last value
                if accum_positive && accum < last_accum { return None; }
                if !accum_positive && accum > last_accum { return None; }
                last_accum = accum;
            }
            None => match c {
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
        let mut power = _1;

        while i < len {
            let c = buf[i] as char;

            match char::to_digit(c, radix) {
                Some(digit) => {
                    // Decrease power one order of magnitude
                    power /= radix_gen;

                    let digit_t: T = cast(digit);

                    // add/subtract current digit depending on sign
                    if accum_positive {
                        accum += digit_t * power;
                    } else {
                        accum -= digit_t * power;
                    }

                    // Detect overflow by comparing to last value
                    if accum_positive && accum < last_accum { return None; }
                    if !accum_positive && accum > last_accum { return None; }
                    last_accum = accum;
                }
                None => match c {
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

    let mut multiplier = _1;

    if exp_found {
        let c = buf[i] as char;
        let base = match (c, exponent) {
            ('e', ExpDec) | ('E', ExpDec) => 10u,
            ('p', ExpBin) | ('P', ExpBin) => 2u,
            _ => return None // char doesn't fit given exponent format
        };

        // parse remaining bytes as decimal integer,
        // skipping the exponent char
        let exp: Option<int> = from_str_bytes_common(
            buf.view(i+1, len), 10, true, false, false, ExpNone, false);

        match exp {
            Some(exp_pow) => {
                multiplier = if exp_pow < 0 {
                    _1 / pow_with_uint::<T>(base, (-exp_pow.to_int()) as uint)
                } else {
                    pow_with_uint::<T>(base, exp_pow.to_int() as uint)
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
#[inline(always)]
pub pure fn from_str_common<T: Num NumCast Zero One Ord Copy>(
        buf: &str, radix: uint, negative: bool, fractional: bool,
        special: bool, exponent: ExponentFormat, empty_zero: bool
        ) -> Option<T> {
    from_str_bytes_common(str::to_bytes(buf), radix, negative,
                            fractional, special, exponent, empty_zero)
}
