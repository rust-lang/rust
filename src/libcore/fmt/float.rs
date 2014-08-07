// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_doc)]

use char;
use collections::Collection;
use fmt;
use iter::{range, DoubleEndedIterator};
use num::{Float, FPNaN, FPInfinite, ToPrimitive, Primitive};
use num::{Zero, One, cast};
use result::Ok;
use slice::{ImmutableSlice, MutableSlice};
use slice;
use str::StrSlice;

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

static DIGIT_P_RADIX: uint = ('p' as uint) - ('a' as uint) + 11u;
static DIGIT_E_RADIX: uint = ('e' as uint) - ('a' as uint) + 11u;

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
 * - `f`             - A closure to invoke with the bytes representing the
 *                     float.
 *
 * # Failure
 * - Fails if `radix` < 2 or `radix` > 36.
 * - Fails if `radix` > 14 and `exp_format` is `ExpDec` due to conflict
 *   between digit and exponent sign `'e'`.
 * - Fails if `radix` > 25 and `exp_format` is `ExpBin` due to conflict
 *   between digit and exponent sign `'p'`.
 */
pub fn float_to_str_bytes_common<T: Primitive + Float, U>(
    num: T,
    radix: uint,
    negative_zero: bool,
    sign: SignFormat,
    digits: SignificantDigits,
    exp_format: ExponentFormat,
    exp_upper: bool,
    f: |&[u8]| -> U
) -> U {
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
        FPNaN => return f("NaN".as_bytes()),
        FPInfinite if num > _0 => {
            return match sign {
                SignAll => return f("+inf".as_bytes()),
                _       => return f("inf".as_bytes()),
            };
        }
        FPInfinite if num < _0 => {
            return match sign {
                SignNone => return f("inf".as_bytes()),
                _        => return f("-inf".as_bytes()),
            };
        }
        _ => {}
    }

    let neg = num < _0 || (negative_zero && _1 / num == Float::neg_infinity());
    // For an f64 the exponent is in the range of [-1022, 1023] for base 2, so
    // we may have up to that many digits. Give ourselves some extra wiggle room
    // otherwise as well.
    let mut buf = [0u8, ..1536];
    let mut end = 0;
    let radix_gen: T = cast(radix as int).unwrap();

    let (num, exp) = match exp_format {
        ExpNone => (num, 0i32),
        ExpDec | ExpBin if num == _0 => (num, 0i32),
        ExpDec | ExpBin => {
            let (exp, exp_base) = match exp_format {
                ExpDec => (num.abs().log10().floor(), cast::<f64, T>(10.0f64).unwrap()),
                ExpBin => (num.abs().log2().floor(), cast::<f64, T>(2.0f64).unwrap()),
                ExpNone => fail!("unreachable"),
            };

            (num / exp_base.powf(exp), cast::<T, i32>(exp).unwrap())
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

        let c = char::from_digit(current_digit.to_int().unwrap() as uint, radix);
        buf[end] = c.unwrap() as u8;
        end += 1;

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
            buf[end] = b'-';
            end += 1;
        }
        SignAll => {
            buf[end] = b'+';
            end += 1;
        }
        _ => ()
    }

    buf.mut_slice_to(end).reverse();

    // Remember start of the fractional digits.
    // Points one beyond end of buf if none get generated,
    // or at the '.' otherwise.
    let start_fractional_digits = end;

    // Now emit the fractional part, if any
    deccum = num.fract();
    if deccum != _0 || (limit_digits && exact && digit_count > 0) {
        buf[end] = b'.';
        end += 1;
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

            let c = char::from_digit(current_digit.to_int().unwrap() as uint,
                                     radix);
            buf[end] = c.unwrap() as u8;
            end += 1;

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

            let extra_digit = ascii2value(buf[end - 1]);
            end -= 1;
            if extra_digit >= radix / 2 { // -> need to round
                let mut i: int = end as int - 1;
                loop {
                    // If reached left end of number, have to
                    // insert additional digit:
                    if i < 0
                    || buf[i as uint] == b'-'
                    || buf[i as uint] == b'+' {
                        for j in range(i as uint + 1, end).rev() {
                            buf[j + 1] = buf[j];
                        }
                        buf[(i + 1) as uint] = value2ascii(1);
                        end += 1;
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
        let buf_max_i = end - 1;

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
                end = i + 1;
            }
        }
    } // If exact and trailing '.', just cut that
    else {
        let max_i = end - 1;
        if buf[max_i] == b'.' {
            end = max_i;
        }
    }

    match exp_format {
        ExpNone => {},
        _ => {
            buf[end] = match exp_format {
                ExpDec if exp_upper => 'E',
                ExpDec if !exp_upper => 'e',
                ExpBin if exp_upper => 'P',
                ExpBin if !exp_upper => 'p',
                _ => fail!("unreachable"),
            } as u8;
            end += 1;

            struct Filler<'a> {
                buf: &'a mut [u8],
                end: &'a mut uint,
            }

            impl<'a> fmt::FormatWriter for Filler<'a> {
                fn write(&mut self, bytes: &[u8]) -> fmt::Result {
                    slice::bytes::copy_memory(self.buf.mut_slice_from(*self.end),
                                              bytes);
                    *self.end += bytes.len();
                    Ok(())
                }
            }

            let mut filler = Filler { buf: buf, end: &mut end };
            match sign {
                SignNeg => {
                    let _ = format_args!(|args| {
                        fmt::write(&mut filler, args)
                    }, "{:-}", exp);
                }
                SignNone | SignAll => {
                    let _ = format_args!(|args| {
                        fmt::write(&mut filler, args)
                    }, "{}", exp);
                }
            }
        }
    }

    f(buf.slice_to(end))
}
