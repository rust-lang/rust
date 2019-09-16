//! Run-time feature detection for ARM on FreeBSD

use super::auxvec;
use crate::detect::{cache, Feature};

/// Try to read the features from the auxiliary vector
pub(crate) fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    let enable_feature = |value: &mut cache::Initializer, f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    if let Ok(auxv) = auxvec::auxv() {
        enable_feature(&mut value, Feature::neon, auxv.hwcap & 0x00001000 != 0);
        enable_feature(&mut value, Feature::pmull, auxv.hwcap2 & 0x00000002 != 0);
        return value;
    }
    value
}
