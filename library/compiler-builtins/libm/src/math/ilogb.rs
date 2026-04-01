/// Extract the binary exponent of `x`.
#[cfg(f16_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ilogbf16(x: f16) -> i32 {
    super::generic::ilogb(x)
}

/// Extract the binary exponent of `x`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ilogbf(x: f32) -> i32 {
    super::generic::ilogb(x)
}

/// Extract the binary exponent of `x`.
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ilogb(x: f64) -> i32 {
    super::generic::ilogb(x)
}

/// Extract the binary exponent of `x`.
#[cfg(f128_enabled)]
#[cfg_attr(assert_no_panic, no_panic::no_panic)]
pub fn ilogbf128(x: f128) -> i32 {
    super::generic::ilogb(x)
}
