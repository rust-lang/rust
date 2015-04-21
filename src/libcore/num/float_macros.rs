// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(hidden)]

macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        use num::Float;
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6,
                "{} is not approximately equal to {}", *a, *b);
    })
}

macro_rules! from_str_radix_float_impl {
    ($T:ty) => {
        fn from_str_radix(src: &str, radix: u32)
                          -> Result<$T, ParseFloatError> {
            use num::FloatErrorKind::*;
            use num::ParseFloatError as PFE;

            // Special values
            match src {
                "inf"   => return Ok(Float::infinity()),
                "-inf"  => return Ok(Float::neg_infinity()),
                "NaN"   => return Ok(Float::nan()),
                _       => {},
            }

            let (is_positive, src) =  match src.slice_shift_char() {
                None             => return Err(PFE { kind: Empty }),
                Some(('-', ""))  => return Err(PFE { kind: Empty }),
                Some(('-', src)) => (false, src),
                Some((_, _))     => (true,  src),
            };

            // The significand to accumulate
            let mut sig = if is_positive { 0.0 } else { -0.0 };
            // Necessary to detect overflow
            let mut prev_sig = sig;
            let mut cs = src.chars().enumerate();
            // Exponent prefix and exponent index offset
            let mut exp_info = None::<(char, usize)>;

            // Parse the integer part of the significand
            for (i, c) in cs.by_ref() {
                match c.to_digit(radix) {
                    Some(digit) => {
                        // shift significand one digit left
                        sig = sig * (radix as $T);

                        // add/subtract current digit depending on sign
                        if is_positive {
                            sig = sig + ((digit as isize) as $T);
                        } else {
                            sig = sig - ((digit as isize) as $T);
                        }

                        // Detect overflow by comparing to last value, except
                        // if we've not seen any non-zero digits.
                        if prev_sig != 0.0 {
                            if is_positive && sig <= prev_sig
                                { return Ok(Float::infinity()); }
                            if !is_positive && sig >= prev_sig
                                { return Ok(Float::neg_infinity()); }

                            // Detect overflow by reversing the shift-and-add process
                            if is_positive && (prev_sig != (sig - digit as $T) / radix as $T)
                                { return Ok(Float::infinity()); }
                            if !is_positive && (prev_sig != (sig + digit as $T) / radix as $T)
                                { return Ok(Float::neg_infinity()); }
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
                            return Err(PFE { kind: Invalid });
                        },
                    },
                }
            }

            // If we are not yet at the exponent parse the fractional
            // part of the significand
            if exp_info.is_none() {
                let mut power = 1.0;
                for (i, c) in cs.by_ref() {
                    match c.to_digit(radix) {
                        Some(digit) => {
                            // Decrease power one order of magnitude
                            power = power / (radix as $T);
                            // add/subtract current digit depending on sign
                            sig = if is_positive {
                                sig + (digit as $T) * power
                            } else {
                                sig - (digit as $T) * power
                            };
                            // Detect overflow by comparing to last value
                            if is_positive && sig < prev_sig
                                { return Ok(Float::infinity()); }
                            if !is_positive && sig > prev_sig
                                { return Ok(Float::neg_infinity()); }
                            prev_sig = sig;
                        },
                        None => match c {
                            'e' | 'E' | 'p' | 'P' => {
                                exp_info = Some((c, i + 1));
                                break; // start of exponent
                            },
                            _ => {
                                return Err(PFE { kind: Invalid });
                            },
                        },
                    }
                }
            }

            // Parse and calculate the exponent
            let exp = match exp_info {
                Some((c, offset)) => {
                    let base = match c {
                        'E' | 'e' if radix == 10 => 10.0,
                        'P' | 'p' if radix == 16 => 2.0,
                        _ => return Err(PFE { kind: Invalid }),
                    };

                    // Parse the exponent as decimal integer
                    let src = &src[offset..];
                    let (is_positive, exp) = match src.slice_shift_char() {
                        Some(('-', src)) => (false, src.parse::<usize>()),
                        Some(('+', src)) => (true,  src.parse::<usize>()),
                        Some((_, _))     => (true,  src.parse::<usize>()),
                        None             => return Err(PFE { kind: Invalid }),
                    };

                    match (is_positive, exp) {
                        (true,  Ok(exp)) => base.powi(exp as i32),
                        (false, Ok(exp)) => 1.0 / base.powi(exp as i32),
                        (_, Err(_))      => return Err(PFE { kind: Invalid }),
                    }
                },
                None => 1.0, // no exponent
            };

            Ok(sig * exp)
        }
    }
}
