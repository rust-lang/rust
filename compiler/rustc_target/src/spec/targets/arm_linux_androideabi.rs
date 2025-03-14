use crate::spec::{FloatAbi, SanitizerSet, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "arm-linux-androideabi".into(),
        metadata: TargetMetadata {
            description: Some("Armv6 Android".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            // https://developer.android.com/ndk/guides/abis.html#armeabi
            features: "+strict-align,+v5te".into(),
            supported_sanitizers: SanitizerSet::ADDRESS,
            max_atomic_width: Some(32),
            ..base::android::opts()
        },
    }
}
