use super::super::Float;

pub fn fdim<F: Float>(x: F, y: F) -> F {
    if x.is_nan() {
        x
    } else if y.is_nan() {
        y
    } else if x > y {
        x - y
    } else {
        F::ZERO
    }
}
