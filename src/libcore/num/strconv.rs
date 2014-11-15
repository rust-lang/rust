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

use char::Char;
use iter::Iterator;
use num;
use num::{Int, Float};
use option::{None, Option, Some};
use str::{from_str, StrPrelude};

pub fn from_str_radix_float<T: Float>(src: &str, radix: uint) -> Option<T> {
   assert!(radix >= 2 && radix <= 36,
           "from_str_radix_float: must lie in the range `[2, 36]` - found {}",
           radix);

    let _0: T = Float::zero();
    let _1: T = Float::one();
    let radix_t: T = num::cast(radix as int).unwrap();

    // Special values
    match src {
        "inf"   => return Some(Float::infinity()),
        "-inf"  => return Some(Float::neg_infinity()),
        "NaN"   => return Some(Float::nan()),
        _       => {},
    }

    let (is_positive, src) =  match src.slice_shift_char() {
        (None, _)        => return None,
        (Some('-'), "")  => return None,
        (Some('-'), src) => (false, src),
        (Some(_), _)     => (true,  src),
    };

    // The significand to accumulate
    let mut sig = if is_positive { _0 } else { -_0 };
    // Necessary to detect overflow
    let mut prev_sig = sig;
    let mut cs = src.chars().enumerate();
    // Exponent prefix and exponent index offset
    let mut exp_info = None::<(char, uint)>;

    // Parse the integer part of the significand
    for (i, c) in cs {
        match c.to_digit(radix) {
            Some(digit) => {
                // shift significand one digit left
                sig = sig * radix_t;

                // add/subtract current digit depending on sign
                if is_positive {
                    sig = sig + num::cast(digit as int).unwrap();
                } else {
                    sig = sig - num::cast(digit as int).unwrap();
                }

                // Detect overflow by comparing to last value, except
                // if we've not seen any non-zero digits.
                if prev_sig != _0 {
                    if is_positive && sig <= prev_sig
                        { return Some(Float::infinity()); }
                    if !is_positive && sig >= prev_sig
                        { return Some(Float::neg_infinity()); }

                    // Detect overflow by reversing the shift-and-add process
                    let digit: T = num::cast(digit as int).unwrap();
                    if is_positive && (prev_sig != ((sig - digit) / radix_t))
                        { return Some(Float::infinity()); }
                    if !is_positive && (prev_sig != ((sig + digit) / radix_t))
                        { return Some(Float::neg_infinity()); }
                }
                prev_sig = sig;
            },
            None => match c {
                'e' | 'E' | 'p' | 'P' => {
                    exp_info = Some((c, i + 1));
                    break;  // start of exponent
                },
                '.' => {
                    break;  // start of fractional part
                },
                _ => {
                    return None;
                },
            },
        }
    }

    // If we are not yet at the exponent parse the fractional
    // part of the significand
    if exp_info.is_none() {
        let mut power = _1;
        for (i, c) in cs {
            match c.to_digit(radix) {
                Some(digit) => {
                    let digit: T = num::cast(digit).unwrap();
                    // Decrease power one order of magnitude
                    power = power / radix_t;
                    // add/subtract current digit depending on sign
                    sig = if is_positive {
                        sig + digit * power
                    } else {
                        sig - digit * power
                    };
                    // Detect overflow by comparing to last value
                    if is_positive && sig < prev_sig
                        { return Some(Float::infinity()); }
                    if !is_positive && sig > prev_sig
                        { return Some(Float::neg_infinity()); }
                    prev_sig = sig;
                },
                None => match c {
                    'e' | 'E' | 'p' | 'P' => {
                        exp_info = Some((c, i + 1));
                        break; // start of exponent
                    },
                    _ => {
                        return None; // invalid number
                    },
                },
            }
        }
    }

    // Parse and calculate the exponent
    let exp = match exp_info {
        Some((c, offset)) => {
            let base: T = match c {
                'E' | 'e' if radix == 10 => num::cast(10u).unwrap(),
                'P' | 'p' if radix == 16 => num::cast(2u).unwrap(),
                _ => return None,
            };

            // Parse the exponent as decimal integer
            let src = src[offset..];
            let (is_positive, exp) = match src.slice_shift_char() {
                (Some('-'), src) => (false, from_str::<uint>(src)),
                (Some('+'), src) => (true,  from_str::<uint>(src)),
                (Some(_), _)     => (true,  from_str::<uint>(src)),
                (None, _)        => return None,
            };

            match (is_positive, exp) {
                (true,  Some(exp)) => base.powi(exp as i32),
                (false, Some(exp)) => _1 / base.powi(exp as i32),
                (_, None)          => return None,
            }
        },
        None => _1, // no exponent
    };

    Some(sig * exp)
}

pub fn from_str_radix_int<T: Int>(src: &str, radix: uint) -> Option<T> {
   assert!(radix >= 2 && radix <= 36,
           "from_str_radix_int: must lie in the range `[2, 36]` - found {}",
           radix);

    fn cast<T: Int>(x: uint) -> T {
        num::cast(x).unwrap()
    }

    let _0: T = Int::zero();
    let _1: T = Int::one();
    let is_signed = _0 > Int::min_value();

    let (is_positive, src) =  match src.slice_shift_char() {
        (Some('-'), src) if is_signed => (false, src),
        (Some(_), _) => (true, src),
        (None, _) => return None,
    };

    let mut xs = src.chars().map(|c| {
        c.to_digit(radix).map(cast)
    });
    let radix = cast(radix);
    let mut result = _0;

    if is_positive {
        for x in xs {
            let x = match x {
                Some(x) => x,
                None => return None,
            };
            result = match result.checked_mul(radix) {
                Some(result) => result,
                None => return None,
            };
            result = match result.checked_add(x) {
                Some(result) => result,
                None => return None,
            };
        }
    } else {
        for x in xs {
            let x = match x {
                Some(x) => x,
                None => return None,
            };
            result = match result.checked_mul(radix) {
                Some(result) => result,
                None => return None,
            };
            result = match result.checked_sub(x) {
                Some(result) => result,
                None => return None,
            };
        }
    }

    Some(result)
}

#[cfg(test)]
mod test {
    use super::*;
    use option::*;
    use num::Float;

    #[test]
    fn from_str_issue7588() {
        let u : Option<u8> = from_str_radix_int("1000", 10);
        assert_eq!(u, None);
        let s : Option<i16> = from_str_radix_int("80000", 10);
        assert_eq!(s, None);
        let f : Option<f32> = from_str_radix_float("10000000000000000000000000000000000000000", 10);
        assert_eq!(f, Some(Float::infinity()))
        let fe : Option<f32> = from_str_radix_float("1e40", 10);
        assert_eq!(fe, Some(Float::infinity()))
    }

    #[test]
    fn test_from_str_radix_float() {
        let x1 : Option<f64> = from_str_radix_float("-123.456", 10);
        assert_eq!(x1, Some(-123.456));
        let x2 : Option<f32> = from_str_radix_float("123.456", 10);
        assert_eq!(x2, Some(123.456));
        let x3 : Option<f32> = from_str_radix_float("-0.0", 10);
        assert_eq!(x3, Some(-0.0));
        let x4 : Option<f32> = from_str_radix_float("0.0", 10);
        assert_eq!(x4, Some(0.0));
        let x4 : Option<f32> = from_str_radix_float("1.0", 10);
        assert_eq!(x4, Some(1.0));
        let x5 : Option<f32> = from_str_radix_float("-1.0", 10);
        assert_eq!(x5, Some(-1.0));
    }
}
