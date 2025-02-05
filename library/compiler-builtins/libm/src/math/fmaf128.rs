/// Fused multiply add (f128)
///
/// Computes `(x*y)+z`, rounded as one ternary operation (i.e. calculated with infinite precision).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaf128(x: f128, y: f128, z: f128) -> f128 {
    return super::generic::fma(x, y, z);
}
