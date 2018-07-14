use core::f64;

#[inline]
pub fn fdim(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        x
    } else if y.is_nan() {
        y
    } else {
        if x > y {
            x - y
        } else {
            0.0
        }
    }
}
