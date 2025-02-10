/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg(f16_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimumf16(x: f16, y: f16) -> f16 {
    super::generic::fminimum(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimum(x: f64, y: f64) -> f64 {
    super::generic::fminimum(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimumf(x: f32, y: f32) -> f32 {
    super::generic::fminimum(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `minimum`. The result orders -0.0 < 0.0.
#[cfg(f128_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimumf128(x: f128, y: f128) -> f128 {
    super::generic::fminimum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg(f16_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximumf16(x: f16, y: f16) -> f16 {
    super::generic::fmaximum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximumf(x: f32, y: f32) -> f32 {
    super::generic::fmaximum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximum(x: f64, y: f64) -> f64 {
    super::generic::fmaximum(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2019 `maximum`. The result orders -0.0 < 0.0.
#[cfg(f128_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximumf128(x: f128, y: f128) -> f128 {
    super::generic::fmaximum(x, y)
}
