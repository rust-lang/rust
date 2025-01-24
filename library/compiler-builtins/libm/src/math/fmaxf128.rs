/// Return the greater of two arguments or, if either argument is NaN, the other argument.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaxf128(x: f128, y: f128) -> f128 {
    super::generic::fmax(x, y)
}
