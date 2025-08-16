use crate::spec::{
    Cc, LinkerFlavor, Lld, PanicStrategy, RelroLevel, StackProbeType, Target, TargetMetadata,
    TargetOptions,
};

const LINKER_SCRIPT: &str = include_str!("./aarch64_nintendo_switch_freestanding_linker_script.ld");

/// A base target for Nintendo Switch devices using a pure LLVM toolchain.
pub(crate) fn target() -> Target {
    Target {
        llvm_target: "aarch64-unknown-none".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Nintendo Switch, Horizon".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+v8a,+neon,+crypto,+crc".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            link_script: Some(LINKER_SCRIPT.into()),
            os: "horizon".into(),
            vendor: "nintendo".into(),
            max_atomic_width: Some(128),
            stack_probes: StackProbeType::Inline,
            panic_strategy: PanicStrategy::Abort,
            position_independent_executables: true,
            dynamic_linking: true,
            relro_level: RelroLevel::Off,
            ..Default::default()
        },
    }
}
