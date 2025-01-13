/// Ceil (f16)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn ceilf16(x: f16) -> f16 {
    super::generic::ceil(x)
}
