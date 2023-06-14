// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: --crate-type lib -Cdebug-assertions=no

#![feature(flt2dec)]

extern crate core;

use core::num::flt2dec;
use std::fmt::{Formatter, Result};

// EMIT_MIR funky_arms.float_to_exponential_common.ConstProp.diff
fn float_to_exponential_common<T>(fmt: &mut Formatter<'_>, num: &T, upper: bool) -> Result
where
    T: flt2dec::DecodableFloat,
{
    let force_sign = fmt.sign_plus();
    // A bug in const propagation (never reached master, but during dev of a PR) caused the
    // `sign = Minus` assignment to get propagated into all future reads of `sign`. This is
    // wrong because `sign` could also have `MinusPlus` value.
    let sign = match force_sign {
        false => flt2dec::Sign::Minus,
        true => flt2dec::Sign::MinusPlus,
    };

    if let Some(precision) = fmt.precision() {
        // 1 integral digit + `precision` fractional digits = `precision + 1` total digits
        float_to_exponential_common_exact(fmt, num, sign, precision as u32 + 1, upper)
    } else {
        float_to_exponential_common_shortest(fmt, num, sign, upper)
    }
}
#[inline(never)]
fn float_to_exponential_common_exact<T>(
    fmt: &mut Formatter<'_>,
    num: &T,
    sign: flt2dec::Sign,
    precision: u32,
    upper: bool,
) -> Result
where
    T: flt2dec::DecodableFloat,
{
    unimplemented!()
}

#[inline(never)]
fn float_to_exponential_common_shortest<T>(
    fmt: &mut Formatter<'_>,
    num: &T,
    sign: flt2dec::Sign,
    upper: bool,
) -> Result
where
    T: flt2dec::DecodableFloat,
{
    unimplemented!()
}
