//! Run-time feature detection for Aarch64 on FreeBSD.

pub(crate) use super::super::aarch64::detect_features;

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
        println!("lse: {:?}", is_aarch64_feature_detected!("lse"));
        println!("rdm: {:?}", is_aarch64_feature_detected!("rdm"));
        println!("rcpc: {:?}", is_aarch64_feature_detected!("rcpc"));
        println!("dotprod: {:?}", is_aarch64_feature_detected!("dotprod"));
        println!("tme: {:?}", is_aarch64_feature_detected!("tme"));
    }
}
