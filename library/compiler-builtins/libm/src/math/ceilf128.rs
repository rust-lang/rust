/// Ceil (f128)
///
/// Finds the nearest integer greater than or equal to `x`.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn ceilf128(x: f128) -> f128 {
    super::generic::ceil(x)
}
