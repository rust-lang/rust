/// Round `x` to the nearest integer, breaking ties toward even.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn rintf128(x: f128) -> f128 {
    super::generic::rint(x)
}
