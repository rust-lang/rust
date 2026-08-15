use super::super::{Float, MinInt};

const FP_ILOGBNAN: i32 = i32::MIN;
const FP_ILOGB0: i32 = FP_ILOGBNAN;

#[inline]
pub fn ilogb<F: Float>(x: F) -> i32 {
    let zero = F::Int::ZERO;
    let mut i = x.to_bits();
    let e = x.ex() as i32;

    if e == 0 {
        i <<= F::EXP_BITS + 1;
        if i == F::Int::ZERO {
            force_eval!(0.0 / 0.0);
            return FP_ILOGB0;
        }
        /* subnormal x */
        let mut e = -(F::EXP_BIAS as i32);
        while i >> (F::BITS - 1) == zero {
            e -= 1;
            i <<= 1;
        }
        e
    } else if e == F::EXP_SAT as i32 {
        force_eval!(0.0 / 0.0);
        if i << (F::EXP_BITS + 1) != zero {
            FP_ILOGBNAN
        } else {
            i32::MAX
        }
    } else {
        e - F::EXP_BIAS as i32
    }
}
