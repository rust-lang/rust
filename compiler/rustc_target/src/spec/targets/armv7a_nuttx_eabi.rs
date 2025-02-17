// Targets Cortex-A7/A8/A9 processors (ARMv7-A) running NuttX
//
// This target assumes that the device does NOT have a FPU (Floating Point Unit)
// and will use software floating point operations. This matches the NuttX EABI
// configuration without hardware floating point support.

use crate::spec::{
    Cc, FloatAbi, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions, cvs,
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
        families: cvs!["unix"],
        os: "nuttx".into(),
        ..Default::default()
    };
    Target {
        llvm_target: "armv7a-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("ARMv7-A Cortex-A with NuttX".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: opts,
    }
}
