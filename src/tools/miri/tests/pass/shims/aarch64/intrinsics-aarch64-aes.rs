// We're testing aarch64 AES target specific features.
//@only-target: aarch64
//@compile-flags: -C target-feature=+neon,+aes
//@run-native

use std::arch::aarch64::*;
use std::arch::is_aarch64_feature_detected;

fn main() {
    assert!(is_aarch64_feature_detected!("neon"));
    assert!(is_aarch64_feature_detected!("aes"));

    unsafe {
        test_vmull_p64();
        test_vmull_high_p64();
    }
}

#[target_feature(enable = "neon,aes")]
unsafe fn test_vmull_p64() {
    assert_eq!(vmull_p64(0, 0), 0);
    assert_eq!(vmull_p64(0, 0xffffffffffffffff), 0);
    assert_eq!(vmull_p64(1, 1), 1);
    assert_eq!(vmull_p64(1, 0x8000000000000000), 0x8000000000000000);

    assert_eq!(vmull_p64(0b11, 0b11), 0b101);

    // Check with the same inputs that are used in the x86_64 pclmulqdq test.
    assert_eq!(
        vmull_p64(0x7fffffffffffffff, 0xdd358416f52ecd34),
        (2704901987789626761u128 << 64) | 13036940098130298092u128,
    );
}

#[target_feature(enable = "neon,aes")]
unsafe fn test_vmull_high_p64() {
    // The lower (first) element is ignored.
    let a = vcombine_p64(vcreate_p64(123), vcreate_p64(0b11));
    let b = vcombine_p64(vcreate_p64(456), vcreate_p64(0b11));
    assert_eq!(vmull_high_p64(a, b), 0b101);

    // Check with the same inputs that are used in the x86_64 pclmulqdq test.
    let a = vcombine_p64(vcreate_p64(0), vcreate_p64(0x7fffffffffffffff));
    let b = vcombine_p64(vcreate_p64(0), vcreate_p64(0xdd358416f52ecd34));
    assert_eq!(vmull_high_p64(a, b), (2704901987789626761u128 << 64) | 13036940098130298092u128);
}
