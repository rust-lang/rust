// Test that target-cpu implies the correct target features
//@only-target: x86_64
//@compile-flags: -C target-cpu=x86-64-v4

fn main() {
    assert!(cfg!(target_feature = "avx2"));
    assert!(cfg!(target_feature = "avx512bw"));
    assert!(cfg!(target_feature = "avx512cd"));
    assert!(cfg!(target_feature = "avx512dq"));
    assert!(cfg!(target_feature = "avx512f"));
    assert!(cfg!(target_feature = "avx512vl"));
    assert!(is_x86_feature_detected!("avx512bw"));

    assert!(cfg!(not(target_feature = "avx512vpopcntdq")));
    assert!(!is_x86_feature_detected!("avx512vpopcntdq"));
}
