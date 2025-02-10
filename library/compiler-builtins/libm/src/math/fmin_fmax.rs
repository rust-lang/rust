/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `minNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg(f16_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminf16(x: f16, y: f16) -> f16 {
    super::generic::fmin(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `minNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminf(x: f32, y: f32) -> f32 {
    super::generic::fmin(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `minNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmin(x: f64, y: f64) -> f64 {
    super::generic::fmin(x, y)
}

/// Return the lesser of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `minNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg(f128_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fminf128(x: f128, y: f128) -> f128 {
    super::generic::fmin(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `maxNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg(f16_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaxf16(x: f16, y: f16) -> f16 {
    super::generic::fmax(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `maxNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaxf(x: f32, y: f32) -> f32 {
    super::generic::fmax(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `maxNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmax(x: f64, y: f64) -> f64 {
    super::generic::fmax(x, y)
}

/// Return the greater of two arguments or, if either argument is NaN, the other argument.
///
/// This coincides with IEEE 754-2011 `maxNum`. The result disregards signed zero (meaning if
/// the inputs are -0.0 and +0.0, either may be returned).
#[cfg(f128_enabled)]
#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn fmaxf128(x: f128, y: f128) -> f128 {
    super::generic::fmax(x, y)
}
