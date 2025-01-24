/// Calculate the remainder of `x / y`, the precise result of `x - trunc(x / y) * y`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmod(x: f64, y: f64) -> f64 {
    super::generic::fmod(x, y)
}
