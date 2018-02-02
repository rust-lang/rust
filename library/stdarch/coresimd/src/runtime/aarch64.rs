//! Run-time feature detection on ARM Aarch64.
use runtime::cache;
use runtime::arch::HasFeature;

#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("neon", $unstable_detect_feature:path) => {
        // FIXME: this should be removed once we rename Aarch64 neon to asimd
        $unstable_detect_feature($crate::__vendor_runtime::_Feature::asimd{})
    };
    ("asimd", $unstable_detect_feature:path) => {
        $unstable_detect_feature($crate::__vendor_runtime::__Feature::asimd{})
    };
    ("pmull", $unstable_detect_feature:path) => {
        $unstable_detect_feature($crate::__vendor_runtime::__Feature::pmull{})
    };
    ($t:tt, $unstable_detect_feature:path) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}

/// ARM Aarch64 CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum __Feature {
    /// ARM Advanced SIMD (ASIMD) - Aarch64
    asimd,
    /// Polynomial Multiply
    pmull,
}

pub fn detect_features<T: HasFeature>(mut x: T) -> cache::Initializer {
    let mut value = cache::Initializer::default();
    {
        let mut enable_feature = |f| {
            if x.has_feature(&f) {
                value.set(f as u32);
            }
        };
        enable_feature(__Feature::asimd);
        enable_feature(__Feature::pmull);
    }
    value
}
