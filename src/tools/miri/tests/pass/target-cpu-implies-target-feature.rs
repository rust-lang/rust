// Test that target-cpu implies the correct target features
//@only-target: x86_64
//@compile-flags: -C target-cpu=x86-64-v4

fn main() {
    assert!(is_x86_feature_detected!("avx512bw"));
    assert!(is_x86_feature_detected!("avx512cd"));
    assert!(is_x86_feature_detected!("avx512dq"));
    assert!(is_x86_feature_detected!("avx512f"));
    assert!(is_x86_feature_detected!("avx512vl"));
    assert!(!is_x86_feature_detected!("avx512vpopcntdq"));
}
