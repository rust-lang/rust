use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::android_base::opts();
    // https://developer.android.com/ndk/guides/abis.html#armeabi
    base.features = "+strict-align,+v5te".to_string();
    base.max_atomic_width = Some(32);

    Target {
        llvm_target: "arm-linux-androideabi".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        options: TargetOptions { unsupported_abis: super::arm_base::unsupported_abis(), ..base },
    }
}
