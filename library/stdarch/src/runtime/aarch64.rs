//! Run-time feature detection on ARM Aarch64.
use super::{bit, linux};

#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("neon") => {
        // FIXME: this should be removed once we rename Aarch64 neon to asimd
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::asimd{})
    };
    ("asimd") => {
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::asimd{})
    };
    ("pmull") => {
        $crate::vendor::__unstable_detect_feature($crate::vendor::__Feature::pmull{})
    };
    ($t:tt) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
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

pub fn detect_features<T: linux::FeatureQuery>(mut x: T) -> usize {
    let value: usize = 0;
    {
        let mut enable_feature = |f| {
            if x.has_feature(&f) {
                bit::set(value, f as u32);
            }
        };
        enable_feature(__Feature::asimd);
        enable_feature(__Feature::pmull);
    }
    value
}

impl linux::FeatureQuery for linux::CpuInfo {
    fn has_feature(&mut self, x: &__Feature) -> bool {
        use self::__Feature::*;
        match *x {
            asimd => self.field("Features").has("asimd"),
            pmull => self.field("Features").has("pmull"),
        }
    }
}
