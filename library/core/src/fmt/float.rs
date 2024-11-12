use crate::fmt::{Debug, Display, Formatter, LowerExp, Result, UpperExp};
use crate::mem::MaybeUninit;
use crate::num::{flt2dec, fmt as numfmt};

#[doc(hidden)]
trait GeneralFormat: PartialOrd {
    /// Determines if a value should use exponential based on its magnitude, given the precondition
    /// that it will not be rounded any further before it is displayed.
    fn already_rounded_value_should_use_exponential(&self) -> bool;
}

macro_rules! impl_general_format {
    ($($t:ident)*) => {
        $(impl GeneralFormat for $t {
            fn already_rounded_value_should_use_exponential(&self) -> bool {
                let abs = $t::abs(*self);
                (abs != 0.0 && abs < 1e-4) || abs >= 1e+16
            }
        })*
    }
}

impl_general_format! { f32 f64 }

// Don't inline this so callers don't use the stack space this function
// requires unless they have to.
#[inline(never)]
fn float_to_decimal_common_exact<T>(
    fmt: &mut Formatter<'_>,
    num: &T,
    sign: flt2dec::Sign,
    precision: usize,
) -> Result
where
    T: flt2dec::DecodableFloat,
{
    let mut buf: [MaybeUninit<u8>; 1024] = [MaybeUninit::uninit(); 1024]; // enough for f32 and f64
    let mut parts: [MaybeUninit<numfmt::Part<'_>>; 4] = [MaybeUninit::uninit(); 4];
    let formatted = flt2dec::to_exact_fixed_str(
        flt2dec::strategy::grisu::format_exact,
        *num,
        sign,
        precision,
        &mut buf,
        &mut parts,
    );
    // SAFETY: `to_exact_fixed_str` and `format_exact` produce only ASCII characters.
    unsafe { fmt.pad_formatted_parts(&formatted) }
}

// Don't inline this so callers that call both this and the above won't wind
// up using the combined stack space of both functions in some cases.
#[inline(never)]
fn float_to_decimal_common_shortest<T>(
    fmt: &mut Formatter<'_>,
    num: &T,
    sign: flt2dec::Sign,
    precision: usize,
) -> Result
where
    T: flt2dec::DecodableFloat,
{
    // enough for f32 and f64
    let mut buf: [MaybeUninit<u8>; flt2dec::MAX_SIG_DIGITS] =
        [MaybeUninit::uninit(); flt2dec::MAX_SIG_DIGITS];
    let mut parts: [MaybeUninit<numfmt::Part<'_>>; 4] = [MaybeUninit::uninit(); 4];
    let formatted = flt2dec::to_shortest_str(
        flt2dec::strategy::grisu::format_shortest,
        *num,
        sign,
        precision,
        &mut buf,
        &mut parts,
    );
    // SAFETY: `to_shortest_str` and `format_shortest` produce only ASCII characters.
    unsafe { fmt.pad_formatted_parts(&formatted) }
}

fn float_to_decimal_display<T>(fmt: &mut Formatter<'_>, num: &T) -> Result
where
    T: flt2dec::DecodableFloat,
{
    let force_sign = fmt.sign_plus();
    let sign = match force_sign {
        false => flt2dec::Sign::Minus,
        true => flt2dec::Sign::MinusPlus,
    };

    if let Some(precision) = fmt.precision {
        float_to_decimal_common_exact(fmt, num, sign, precision)
    } else {
        let min_precision = 0;
        float_to_decimal_common_shortest(fmt, num, sign, min_precision)
    }
}

// Don't inline this so callers don't use the stack space this function
// requires unless they have to.
#[inline(never)]
fn float_to_exponential_common_exact<T>(
    fmt: &mut Formatter<'_>,
    num: &T,
    sign: flt2dec::Sign,
    precision: usize,
    upper: bool,
) -> Result
where
    T: flt2dec::DecodableFloat,
{
    let mut buf: [MaybeUninit<u8>; 1024] = [MaybeUninit::uninit(); 1024]; // enough for f32 and f64
    let mut parts: [MaybeUninit<numfmt::Part<'_>>; 6] = [MaybeUninit::uninit(); 6];
    let formatted = flt2dec::to_exact_exp_str(
        flt2dec::strategy::grisu::format_exact,
        *num,
        sign,
        precision,
        upper,
        &mut buf,
        &mut parts,
    );
    // SAFETY: `to_exact_exp_str` and `format_exact` produce only ASCII characters.
    unsafe { fmt.pad_formatted_parts(&formatted) }
}

// Don't inline this so callers that call both this and the above won't wind
// up using the combined stack space of both functions in some cases.
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
    // enough for f32 and f64
    let mut buf: [MaybeUninit<u8>; flt2dec::MAX_SIG_DIGITS] =
        [MaybeUninit::uninit(); flt2dec::MAX_SIG_DIGITS];
    let mut parts: [MaybeUninit<numfmt::Part<'_>>; 6] = [MaybeUninit::uninit(); 6];
    let formatted = flt2dec::to_shortest_exp_str(
        flt2dec::strategy::grisu::format_shortest,
        *num,
        sign,
        (0, 0),
        upper,
        &mut buf,
        &mut parts,
    );
    // SAFETY: `to_shortest_exp_str` and `format_shortest` produce only ASCII characters.
    unsafe { fmt.pad_formatted_parts(&formatted) }
}

// Common code of floating point LowerExp and UpperExp.
fn float_to_exponential_common<T>(fmt: &mut Formatter<'_>, num: &T, upper: bool) -> Result
where
    T: flt2dec::DecodableFloat,
{
    let force_sign = fmt.sign_plus();
    let sign = match force_sign {
        false => flt2dec::Sign::Minus,
        true => flt2dec::Sign::MinusPlus,
    };

    if let Some(precision) = fmt.precision {
        // 1 integral digit + `precision` fractional digits = `precision + 1` total digits
        float_to_exponential_common_exact(fmt, num, sign, precision + 1, upper)
    } else {
        float_to_exponential_common_shortest(fmt, num, sign, upper)
    }
}

fn float_to_general_debug<T>(fmt: &mut Formatter<'_>, num: &T) -> Result
where
    T: flt2dec::DecodableFloat + GeneralFormat,
{
    let force_sign = fmt.sign_plus();
    let sign = match force_sign {
        false => flt2dec::Sign::Minus,
        true => flt2dec::Sign::MinusPlus,
    };

    if let Some(precision) = fmt.precision {
        // this behavior of {:.PREC?} predates exponential formatting for {:?}
        float_to_decimal_common_exact(fmt, num, sign, precision)
    } else {
        // since there is no precision, there will be no rounding
        if num.already_rounded_value_should_use_exponential() {
            let upper = false;
            float_to_exponential_common_shortest(fmt, num, sign, upper)
        } else {
            let min_precision = 1;
            float_to_decimal_common_shortest(fmt, num, sign, min_precision)
        }
    }
}

macro_rules! floating {
    ($($ty:ident)*) => {
        $(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Debug for $ty {
                fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
                    float_to_general_debug(fmt, self)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl Display for $ty {
                fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
                    float_to_decimal_display(fmt, self)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl LowerExp for $ty {
                fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
                    float_to_exponential_common(fmt, self, false)
                }
            }

            #[stable(feature = "rust1", since = "1.0.0")]
            impl UpperExp for $ty {
                fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
                    float_to_exponential_common(fmt, self, true)
                }
            }
        )*
    };
}

floating! { f32 f64 }

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for f16 {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:#06x}", self.to_bits())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for f128 {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:#034x}", self.to_bits())
    }
}
