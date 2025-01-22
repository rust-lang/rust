/// Round `x` to the nearest integer, breaking ties toward even.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn rintf16(x: f16) -> f16 {
    super::generic::rint(x)
}
