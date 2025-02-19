// Generic ARMv7-A target for bare-metal code - floating point disabled
//
// This is basically the `armv7-unknown-linux-gnueabi` target with some changes
// (listed below) to bring it closer to the bare-metal `thumb` & `aarch64`
// targets:
//
// - `TargetOptions.features`: added `+strict-align`. rationale: unaligned
// memory access is disabled on boot on these cores
// - linker changed to LLD. rationale: C is not strictly needed to build
// bare-metal binaries (the `gcc` linker has the advantage that it knows where C
// libraries and crt*.o are but it's not much of an advantage here); LLD is also
// faster
// - `panic_strategy` set to `abort`. rationale: matches `thumb` targets
// - `relocation-model` set to `static`; also no PIE, no relro and no dynamic
// linking. rationale: matches `thumb` targets

use crate::spec::{
    Cc, FloatAbi, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    let opts = TargetOptions {
        abi: "eabi".into(),
        llvm_floatabi: Some(FloatAbi::Soft),
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        features: "+v7,+thumb2,+soft-float,-neon,+strict-align".into(),
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        max_atomic_width: Some(64),
        panic_strategy: PanicStrategy::Abort,
        emit_debug_gdb_scripts: false,
        c_enum_min_bits: Some(8),
        ..Default::default()
    };
    Target {
        llvm_target: "armv7a-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv7-A".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: opts,
    }
}
