//! Run-time feature detection on ARM Aarch32.
use runtime::bit;
use runtime::arch::HasFeature;

#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("neon", $unstable_detect_feature:path) => {
        $unstable_detect_feature($crate::__vendor_runtime::__Feature::neon{})
    };
    ("pmull", $unstable_detect_feature:path) => {
        $unstable_detect_feature($crate::__vendor_runtime::__Feature::pmull{})
    };
    ($t:tt, $unstable_detect_feature:path) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}

/// ARM CPU Feature enum. Each variant denotes a position in a bitset for a
/// particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum __Feature {
    /// ARM Advanced SIMD (NEON) - Aarch32
    neon,
    /// Polynomial Multiply
    pmull,
}

pub fn detect_features<T: HasFeature>(mut x: T) -> usize {
    let mut value: usize = 0;
    {
        let mut enable_feature = |f| {
            if x.has_feature(&f) {
                value = bit::set(value, f as u32);
            }
        };
        enable_feature(__Feature::neon);
        enable_feature(__Feature::pmull);
    }
    value
}
