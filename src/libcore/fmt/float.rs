// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::ExponentFormat::*;
pub use self::SignificantDigits::*;

use prelude::*;

use char;
use fmt;
use num::Float;
use num::FpCategory as Fp;
use ops::{Div, Rem, Mul};
use slice;
use str;

/// A flag that specifies whether to use exponential (scientific) notation.
pub enum ExponentFormat {
    /// Do not use exponential notation.
    ExpNone,
    /// Use exponential notation with the exponent having a base of 10 and the
    /// exponent sign being `e` or `E`. For example, 1000 would be printed
    /// 1e3.
    ExpDec
}

/// The number of digits used for emitting the fractional part of a number, if
/// any.
pub enum SignificantDigits {
    /// At most the given number of digits will be printed, truncating any
    /// trailing zeroes.
    DigMax(usize),

    /// Precisely the given number of digits will be printed.
    DigExact(usize)
}

#[doc(hidden)]
pub trait MyFloat: Float + PartialEq + PartialOrd + Div<Output=Self> +
                   Mul<Output=Self> + Rem<Output=Self> + Copy {
    fn from_u32(u: u32) -> Self;
    fn to_i32(&self) -> i32;
}

macro_rules! doit {
    ($($t:ident)*) => ($(impl MyFloat for $t {
        fn from_u32(u: u32) -> $t { u as $t }
        fn to_i32(&self) -> i32 { *self as i32 }
    })*)
}
doit! { f32 f64 }

/// Converts a float number to its string representation.
/// This is meant to be a common base implementation for various formatting styles.
/// The number is assumed to be non-negative, callers use `Formatter::pad_integral`
/// to add the right sign, if any.
///
/// # Arguments
///
/// - `num`           - The number to convert (non-negative). Accepts any number that
///                     implements the numeric traits.
/// - `digits`        - The amount of digits to use for emitting the fractional
///                     part, if any. See `SignificantDigits`.
/// - `exp_format`   - Whether or not to use the exponential (scientific) notation.
///                    See `ExponentFormat`.
/// - `exp_capital`   - Whether or not to use a capital letter for the exponent sign, if
///                     exponential notation is desired.
/// - `f`             - A closure to invoke with the string representing the
///                     float.
///
/// # Panics
///
/// - Panics if `num` is negative.
pub fn float_to_str_bytes_common<T: MyFloat, U, F>(
    num: T,
    digits: SignificantDigits,
    exp_format: ExponentFormat,
    exp_upper: bool,
    f: F
) -> U where
    F: FnOnce(&str) -> U,
{
    let _0: T = T::zero();
    let _1: T = T::one();
    let radix: u32 = 10;
    let radix_f = T::from_u32(radix);

    assert!(num.is_nan() || num >= _0, "float_to_str_bytes_common: number is negative");

    match num.classify() {
        Fp::Nan => return f("NaN"),
        Fp::Infinite if num > _0 => {
            return f("inf");
        }
        Fp::Infinite if num < _0 => {
            return f("-inf");
        }
        _ => {}
    }

    // For an f64 the (decimal) exponent is roughly in the range of [-307, 308], so
    // we may have up to that many digits. We err on the side of caution and
    // add 50% extra wiggle room.
    let mut buf = [0; 462];
    let mut end = 0;

    let (num, exp) = match exp_format {
        ExpDec if num != _0 => {
            let exp = num.log10().floor();
            (num / radix_f.powf(exp), exp.to_i32())
        }
        _ => (num, 0)
    };

    // First emit the non-fractional part, looping at least once to make
    // sure at least a `0` gets emitted.
    let mut deccum = num.trunc();
    loop {
        let current_digit = deccum % radix_f;

        // Decrease the deccumulator one digit at a time
        deccum = deccum / radix_f;
        deccum = deccum.trunc();

        let c = char::from_digit(current_digit.to_i32() as u32, radix);
        buf[end] = c.unwrap() as u8;
        end += 1;

        // No more digits to calculate for the non-fractional part -> break
        if deccum == _0 { break; }
    }

    // If limited digits, calculate one digit more for rounding.
    let (limit_digits, digit_count, exact) = match digits {
        DigMax(count)   => (true, count + 1, false),
        DigExact(count) => (true, count + 1, true)
    };

    buf[..end].reverse();

    // Remember start of the fractional digits.
    // Points one beyond end of buf if none get generated,
    // or at the '.' otherwise.
    let start_fractional_digits = end;

    // Now emit the fractional part, if any
    deccum = num.fract();
    if deccum != _0 || (limit_digits && exact && digit_count > 0) {
        buf[end] = b'.';
        end += 1;
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
            deccum = deccum * radix_f;

            let current_digit = deccum.trunc();

            let c = char::from_digit(current_digit.to_i32() as u32, radix);
            buf[end] = c.unwrap() as u8;
            end += 1;

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

            let extra_digit = ascii2value(buf[end - 1]);
            end -= 1;
            if extra_digit >= radix / 2 { // -> need to round
                let mut i: isize = end as isize - 1;
                loop {
                    // If reached left end of number, have to
                    // insert additional digit:
                    if i < 0
                    || buf[i as usize] == b'-'
                    || buf[i as usize] == b'+' {
                        for j in ((i + 1) as usize..end).rev() {
                            buf[j + 1] = buf[j];
                        }
                        buf[(i + 1) as usize] = value2ascii(1);
                        end += 1;
                        break;
                    }

                    // Skip the '.'
                    if buf[i as usize] == b'.' { i -= 1; continue; }

                    // Either increment the digit,
                    // or set to 0 if max and carry the 1.
                    let current_digit = ascii2value(buf[i as usize]);
                    if current_digit < (radix - 1) {
                        buf[i as usize] = value2ascii(current_digit+1);
                        break;
                    } else {
                        buf[i as usize] = value2ascii(0);
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
        ExpDec => {
            buf[end] = if exp_upper { b'E' } else { b'e' };
            end += 1;

            struct Filler<'a> {
                buf: &'a mut [u8],
                end: &'a mut usize,
            }

            impl<'a> fmt::Write for Filler<'a> {
                fn write_str(&mut self, s: &str) -> fmt::Result {
                    slice::bytes::copy_memory(s.as_bytes(),
                                              &mut self.buf[(*self.end)..]);
                    *self.end += s.len();
                    Ok(())
                }
            }

            let mut filler = Filler { buf: &mut buf, end: &mut end };
            let _ = fmt::write(&mut filler, format_args!("{:-}", exp));
        }
    }

    f(unsafe { str::from_utf8_unchecked(&buf[..end]) })
}
