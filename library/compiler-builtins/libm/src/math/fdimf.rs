use core::f32;

#[inline]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fdimf(x: f32, y: f32) -> f32 {
    if x.is_nan() {
        x
    } else if y.is_nan() {
        y
    } else if x > y {
        x - y
    } else {
        0.0
    }
}
