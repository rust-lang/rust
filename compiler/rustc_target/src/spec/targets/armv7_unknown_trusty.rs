use crate::spec::{
    FloatAbi, LinkSelfContainedDefault, PanicStrategy, RelroLevel, Target, TargetMetadata,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        // It's important we use "gnueabi" and not "musleabi" here. LLVM uses it
        // to determine the calling convention and float ABI, and it doesn't
        // support the "musleabi" value.
        llvm_target: "armv7-unknown-unknown-gnueabi".into(),
        metadata: TargetMetadata {
            description: Some("Armv7-A Trusty".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            features: "+v7,+thumb2,+soft-float,-neon".into(),
            max_atomic_width: Some(64),
            mcount: "\u{1}mcount".into(),
            os: "trusty".into(),
            link_self_contained: LinkSelfContainedDefault::InferredForMusl,
            dynamic_linking: false,
            executables: true,
            crt_static_default: true,
            crt_static_respected: true,
            relro_level: RelroLevel::Full,
            panic_strategy: PanicStrategy::Abort,
            position_independent_executables: true,
            static_position_independent_executables: true,

            ..Default::default()
        },
    }
}
