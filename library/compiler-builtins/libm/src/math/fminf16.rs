/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminf16(x: f16, y: f16) -> f16 {
    super::generic::fmin(x, y)
}
