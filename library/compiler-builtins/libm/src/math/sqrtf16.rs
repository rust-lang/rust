/// The square root of `x` (f16).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn sqrtf16(x: f16) -> f16 {
    return super::generic::sqrt(x);
}
