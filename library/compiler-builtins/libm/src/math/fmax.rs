/// Return the greater of two arguments or, if either argument is NaN, the other argument.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmax(x: f64, y: f64) -> f64 {
    super::generic::fmax(x, y)
}
