// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt::{Formatter, Result, LowerExp, UpperExp, Display, Debug};
use num::flt2dec;

// Common code of floating point Debug and Display.
fn float_to_decimal_common<T>(fmt: &mut Formatter, num: &T, negative_zero: bool) -> Result
    where T: flt2dec::DecodableFloat
{
    let force_sign = fmt.sign_plus();
    let sign = match (force_sign, negative_zero) {
        (false, false) => flt2dec::Sign::Minus,
        (false, true)  => flt2dec::Sign::MinusRaw,
        (true,  false) => flt2dec::Sign::MinusPlus,
        (true,  true)  => flt2dec::Sign::MinusPlusRaw,
    };

    let mut buf = [0; 1024]; // enough for f32 and f64
    let mut parts = [flt2dec::Part::Zero(0); 16];
    let formatted = if let Some(precision) = fmt.precision {
        flt2dec::to_exact_fixed_str(flt2dec::strategy::grisu::format_exact, *num, sign,
                                    precision, false, &mut buf, &mut parts)
    } else {
        flt2dec::to_shortest_str(flt2dec::strategy::grisu::format_shortest, *num, sign,
                                 0, false, &mut buf, &mut parts)
    };
    fmt.pad_formatted_parts(&formatted)
}

// Common code of floating point LowerExp and UpperExp.
fn float_to_exponential_common<T>(fmt: &mut Formatter, num: &T, upper: bool) -> Result
    where T: flt2dec::DecodableFloat
{
    let force_sign = fmt.sign_plus();
    let sign = match force_sign {
        false => flt2dec::Sign::Minus,
        true  => flt2dec::Sign::MinusPlus,
    };

    let mut buf = [0; 1024]; // enough for f32 and f64
    let mut parts = [flt2dec::Part::Zero(0); 16];
    let formatted = if let Some(precision) = fmt.precision {
        // 1 integral digit + `precision` fractional digits = `precision + 1` total digits
        flt2dec::to_exact_exp_str(flt2dec::strategy::grisu::format_exact, *num, sign,
                                  precision + 1, upper, &mut buf, &mut parts)
    } else {
        flt2dec::to_shortest_exp_str(flt2dec::strategy::grisu::format_shortest, *num, sign,
                                     (0, 0), upper, &mut buf, &mut parts)
    };
    fmt.pad_formatted_parts(&formatted)
}

macro_rules! floating {
    ($ty:ident) => (
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Debug for $ty {
            fn fmt(&self, fmt: &mut Formatter) -> Result {
                float_to_decimal_common(fmt, self, true)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl Display for $ty {
            fn fmt(&self, fmt: &mut Formatter) -> Result {
                float_to_decimal_common(fmt, self, false)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl LowerExp for $ty {
            fn fmt(&self, fmt: &mut Formatter) -> Result {
                float_to_exponential_common(fmt, self, false)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl UpperExp for $ty {
            fn fmt(&self, fmt: &mut Formatter) -> Result {
                float_to_exponential_common(fmt, self, true)
            }
        }
    )
}

floating! { f32 }
floating! { f64 }
