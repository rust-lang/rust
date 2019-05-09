use super::{log1pf, logf, sqrtf};

const LN2: f32 = 0.693147180559945309417232121458176568;

/* acosh(x) = log(x + sqrt(x*x-1)) */
pub fn acoshf(x: f32) -> f32 {
    let u = x.to_bits();
    let a = u & 0x7fffffff;

    if a < 0x3f800000 + (1 << 23) {
        /* |x| < 2, invalid if x < 1 or nan */
        /* up to 2ulp error in [1,1.125] */
        return log1pf(x - 1.0 + sqrtf((x - 1.0) * (x - 1.0) + 2.0 * (x - 1.0)));
    }
    if a < 0x3f800000 + (12 << 23) {
        /* |x| < 0x1p12 */
        return logf(2.0 * x - 1.0 / (x + sqrtf(x * x - 1.0)));
    }
    /* x >= 0x1p12 */
    return logf(x) + LN2;
}
