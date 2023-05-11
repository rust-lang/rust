//! Representation of a float as the significant digits and exponent.

use crate::num::dec2flt::float::RawFloat;
use crate::num::dec2flt::fpu::set_precision;

#[rustfmt::skip]
const INT_POW10: [u64; 16] = [
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000,
    100000000000,
    1000000000000,
    10000000000000,
    100000000000000,
    1000000000000000,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Number {
    pub exponent: i64,
    pub mantissa: u64,
    pub negative: bool,
    pub many_digits: bool,
}

impl Number {
    /// Detect if the float can be accurately reconstructed from native floats.
    #[inline]
    fn is_fast_path<F: RawFloat>(&self) -> bool {
        F::MIN_EXPONENT_FAST_PATH <= self.exponent
            && self.exponent <= F::MAX_EXPONENT_DISGUISED_FAST_PATH
            && self.mantissa <= F::MAX_MANTISSA_FAST_PATH
            && !self.many_digits
    }

    /// The fast path algorithm using machine-sized integers and floats.
    ///
    /// This is extracted into a separate function so that it can be attempted before constructing
    /// a Decimal. This only works if both the mantissa and the exponent
    /// can be exactly represented as a machine float, since IEE-754 guarantees
    /// no rounding will occur.
    ///
    /// There is an exception: disguised fast-path cases, where we can shift
    /// powers-of-10 from the exponent to the significant digits.
    pub fn try_fast_path<F: RawFloat>(&self) -> Option<F> {
        // The fast path crucially depends on arithmetic being rounded to the correct number of bits
        // without any intermediate rounding. On x86 (without SSE or SSE2) this requires the precision
        // of the x87 FPU stack to be changed so that it directly rounds to 64/32 bit.
        // The `set_precision` function takes care of setting the precision on architectures which
        // require setting it by changing the global state (like the control word of the x87 FPU).
        let _cw = set_precision::<F>();

        if self.is_fast_path::<F>() {
            let mut value = if self.exponent <= F::MAX_EXPONENT_FAST_PATH {
                // normal fast path
                let value = F::from_u64(self.mantissa);
                if self.exponent < 0 {
                    value / F::pow10_fast_path((-self.exponent) as _)
                } else {
                    value * F::pow10_fast_path(self.exponent as _)
                }
            } else {
                // disguised fast path
                let shift = self.exponent - F::MAX_EXPONENT_FAST_PATH;
                let mantissa = self.mantissa.checked_mul(INT_POW10[shift as usize])?;
                if mantissa > F::MAX_MANTISSA_FAST_PATH {
                    return None;
                }
                F::from_u64(mantissa) * F::pow10_fast_path(F::MAX_EXPONENT_FAST_PATH as _)
            };
            if self.negative {
                value = -value;
            }
            Some(value)
        } else {
            None
        }
    }
}
