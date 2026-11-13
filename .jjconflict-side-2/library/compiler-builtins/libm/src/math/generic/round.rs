use super::{copysign, trunc_status};
use crate::support::{Float, MinInt};

#[inline]
pub fn round<F: Float>(x: F) -> F {
    let f0p5 = F::from_parts(false, F::EXP_BIAS - 1, F::Int::ZERO); // 0.5
    let f0p25 = F::from_parts(false, F::EXP_BIAS - 2, F::Int::ZERO); // 0.25

    trunc_status(x + copysign(f0p5 - f0p25 * F::EPSILON, x)).val
}
