use core::f32;

pub fn fdimf(x: f32, y: f32) -> f32 {
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
