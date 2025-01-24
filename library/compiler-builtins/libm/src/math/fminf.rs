/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminf(x: f32, y: f32) -> f32 {
    super::generic::fmin(x, y)
}
