/// Return the lesser of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `minimumNumber`. The result orders -0.0 < 0.0.
#[cfg(f16_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimum_numf16(x: f16, y: f16) -> f16 {
    super::generic::fminimum_num(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `minimumNumber`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimum_numf(x: f32, y: f32) -> f32 {
    super::generic::fminimum_num(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `minimumNumber`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimum_num(x: f64, y: f64) -> f64 {
    super::generic::fminimum_num(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `minimumNumber`. The result orders -0.0 < 0.0.
#[cfg(f128_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminimum_numf128(x: f128, y: f128) -> f128 {
    super::generic::fminimum_num(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `maximumNumber`. The result orders -0.0 < 0.0.
#[cfg(f16_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximum_numf16(x: f16, y: f16) -> f16 {
    super::generic::fmaximum_num(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `maximumNumber`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximum_numf(x: f32, y: f32) -> f32 {
    super::generic::fmaximum_num(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `maximumNumber`. The result orders -0.0 < 0.0.
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximum_num(x: f64, y: f64) -> f64 {
    super::generic::fmaximum_num(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, NaN.
///
/// This coincides with IEEE 754-2019 `maximumNumber`. The result orders -0.0 < 0.0.
#[cfg(f128_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaximum_numf128(x: f128, y: f128) -> f128 {
    super::generic::fmaximum_num(x, y)
}
