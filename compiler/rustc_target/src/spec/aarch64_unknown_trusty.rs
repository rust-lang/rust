// Trusty OS target for AArch64.

use super::{
    crt_objects::LinkSelfContainedDefault, PanicStrategy, RelroLevel, Target, TargetOptions,
};

pub fn target() -> Target {
    Target {
        llvm_target: "aarch64-unknown-unknown-musl".into(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+neon,+fp-armv8,+reserve-x18".into(),
            executables: true,
            max_atomic_width: Some(128),
            panic_strategy: PanicStrategy::Abort,
            os: "trusty".into(),
            position_independent_executables: true,
            static_position_independent_executables: true,
            crt_static_default: true,
            crt_static_respected: false,
            dynamic_linking: false,
            link_self_contained: LinkSelfContainedDefault::Musl,
            relro_level: RelroLevel::Full,
            mcount: "\u{1}_mcount".into(),
            ..Default::default()
        },
    }
}
