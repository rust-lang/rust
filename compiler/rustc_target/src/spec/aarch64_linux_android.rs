use crate::spec::{Target, TargetOptions};

// See https://developer.android.com/ndk/guides/abis.html#arm64-v8a
// for target ABI requirements.

pub fn target() -> Target {
    let mut base = super::android_base::opts();
    base.max_atomic_width = Some(128);
    // As documented in http://developer.android.com/ndk/guides/cpu-features.html
    // the neon (ASIMD) and FP must exist on all android aarch64 targets.
    base.features = "+neon,+fp-armv8".to_string();
    Target {
        llvm_target: "aarch64-linux-android".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: TargetOptions { unsupported_abis: super::arm_base::unsupported_abis(), ..base },
    }
}
