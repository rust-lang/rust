//! Run-time feature detection for Aarch64 on FreeBSD.

use super::super::aarch64::detect_features;
use crate::detect::{cache, Feature};

/// Performs run-time feature detection.
#[inline]
pub fn check_for(x: Feature) -> bool {
    cache::test(x as u32, detect_features)
}

#[cfg(test)]
mod tests {
    #[test]
    fn dump() {
        println!("asimd: {:?}", is_aarch64_feature_detected!("asimd"));
        println!("pmull: {:?}", is_aarch64_feature_detected!("pmull"));
        println!("fp: {:?}", is_aarch64_feature_detected!("fp"));
        println!("fp16: {:?}", is_aarch64_feature_detected!("fp16"));
        println!("sve: {:?}", is_aarch64_feature_detected!("sve"));
        println!("crc: {:?}", is_aarch64_feature_detected!("crc"));
        println!("crypto: {:?}", is_aarch64_feature_detected!("crypto"));
        println!("lse: {:?}", is_aarch64_feature_detected!("lse"));
        println!("rdm: {:?}", is_aarch64_feature_detected!("rdm"));
        println!("rcpc: {:?}", is_aarch64_feature_detected!("rcpc"));
        println!("dotprod: {:?}", is_aarch64_feature_detected!("dotprod"));
    }
}
