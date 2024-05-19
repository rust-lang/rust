use crate::spec::{base, SanitizerSet, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "arm-linux-androideabi".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            // https://developer.android.com/ndk/guides/abis.html#armeabi
            features: "+strict-align,+v5te".into(),
            supported_sanitizers: SanitizerSet::ADDRESS,
            max_atomic_width: Some(32),
            ..base::android::opts()
        },
    }
}
