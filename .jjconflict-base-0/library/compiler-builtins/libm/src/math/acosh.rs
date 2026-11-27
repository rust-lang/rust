use super::{Float, log, log1p, sqrt};

const LN2: f64 = 0.693147180559945309417232121458176568; /* 0x3fe62e42,  0xfefa39ef*/

/// Inverse hyperbolic cosine (f64)
///
/// Calculates the inverse hyperbolic cosine of `x`.
/// Is defined as `log(x + sqrt(x*x-1))`.
/// `x` must be a number greater than or equal to 1.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn acosh(x: f64) -> f64 {
    let ux = x.to_bits();

    /* x < 1 domain error is handled in the called functions */
    if (ux & !f64::SIGN_MASK) < 2_f64.to_bits() {
        /* |x| < 2, invalid if x < 1 */
        /* up to 2ulp error in [1,1.125] */
        let x_1 = x - 1.0;
        log1p(x_1 + sqrt(x_1 * x_1 + 2.0 * x_1))
    } else if ux < ((1 << 26) as f64).to_bits() {
        /* 2 <= x < 0x1p26 */
        log(2.0 * x - 1.0 / (x + sqrt(x * x - 1.0)))
    } else {
        /* x >= 0x1p26 or x <= -2 or nan */
        log(x) + LN2
    }
}
