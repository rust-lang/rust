/// Decompose a float into a normalized value within the range `[0.5, 1)`, and a power of 2.
///
/// That is, `x * 2^p` will represent the input value.
// Placeholder so we can have `frexpf16` in the `Float` trait.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn frexpf16(x: f16) -> (f16, i32) {
    super::generic::frexp(x)
}

/// Decompose a float into a normalized value within the range `[0.5, 1)`, and a power of 2.
///
/// That is, `x * 2^p` will represent the input value.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn frexpf(x: f32) -> (f32, i32) {
    super::generic::frexp(x)
}

/// Decompose a float into a normalized value within the range `[0.5, 1)`, and a power of 2.
///
/// That is, `x * 2^p` will represent the input value.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn frexp(x: f64) -> (f64, i32) {
    super::generic::frexp(x)
}

/// Decompose a float into a normalized value within the range `[0.5, 1)`, and a power of 2.
///
/// That is, `x * 2^p` will represent the input value.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn frexpf128(x: f128) -> (f128, i32) {
    super::generic::frexp(x)
}
