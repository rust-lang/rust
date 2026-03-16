use super::super::{CastFrom, Float};

#[inline]
pub fn frexp<F: Float>(x: F) -> (F, i32) {
    let mut ix = x.to_bits();
    let mut ee = x.ex() as i32;

    if ee == 0 {
        if x == F::ZERO {
            return (x, 0);
        }

        // Subnormals, needs to be normalized first
        ix &= F::SIG_MASK;
        (ee, ix) = F::normalize(ix);
        ix |= x.to_bits() & F::SIGN_MASK;
    } else if ee == F::EXP_SAT as i32 {
        // inf or  NaN
        return (x, 0);
    }

    let e = ee - (F::EXP_BIAS as i32 - 1);
    ix &= F::SIGN_MASK | F::SIG_MASK;
    ix |= F::Int::cast_from(F::EXP_BIAS - 1) << F::SIG_BITS;
    (F::from_bits(ix), e)
}
