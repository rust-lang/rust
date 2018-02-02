//! Run-time feature detection on PowerPC64.
use runtime::cache;
use runtime::arch::HasFeature;

#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("altivec", $unstable_detect_feature:path) => {
        $unstable_detect_feature($crate::__vendor_runtime::__Feature::altivec{})
    };
    ("vsx", $unstable_detect_feature:path) => {
        $unstable_detect_feature($crate::__vendor_runtime::__Feature::vsx{})
    };
    ("power8", $unstable_detect_feature:path) => {
        $unstable_detect_feature($crate::__vendor_runtime::__Feature::power8{})
    };
    ($t:tt, $unstable_detect_feature:path) => { compile_error!(concat!("unknown PowerPC target feature: ", $t)) };
}

/// PowerPC CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum __Feature {
    /// Altivec
    altivec,
    /// VSX
    vsx,
    /// Power8
    power8,
}

pub fn detect_features<T: HasFeature>(mut x: T) -> cache::Initializer {
    let mut value = cache::Initializer::default();
    {
        let mut enable_feature = |f| {
            if x.has_feature(&f) {
                value.set(f as u32);
            }
        };
        enable_feature(__Feature::altivec);
        enable_feature(__Feature::vsx);
        enable_feature(__Feature::power8);
    }
    value
}
