// run-pass

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    // stable target features:
    is_x86_feature_detected!("pclmulqdq");
    is_x86_feature_detected!("rdrand");
    is_x86_feature_detected!("rdseed");
    is_x86_feature_detected!("tsc");
    is_x86_feature_detected!("sse");
    is_x86_feature_detected!("sse2");
    is_x86_feature_detected!("sse3");
    is_x86_feature_detected!("ssse3");
    is_x86_feature_detected!("sse4.1");
    is_x86_feature_detected!("sse4.2");
    is_x86_feature_detected!("sha");
    is_x86_feature_detected!("avx");
    is_x86_feature_detected!("avx2");
    is_x86_feature_detected!("fma");
    is_x86_feature_detected!("bmi1");
    is_x86_feature_detected!("bmi2");
    is_x86_feature_detected!("lzcnt");
    is_x86_feature_detected!("popcnt");
    is_x86_feature_detected!("fxsr");
    is_x86_feature_detected!("xsave");
    is_x86_feature_detected!("xsaveopt");
    is_x86_feature_detected!("xsaves");
    is_x86_feature_detected!("xsavec");
    is_x86_feature_detected!("adx");
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}
