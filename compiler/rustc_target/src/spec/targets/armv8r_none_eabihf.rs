// Targets the Little-endian Cortex-R52 processor (ARMv8-R)

use crate::spec::{
    Cc, FloatAbi, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions,
};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "armv8r-none-eabihf".into(),
        metadata: TargetMetadata {
            description: Some("Bare Armv8-R, hardfloat".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            abi: "eabihf".into(),
            llvm_floatabi: Some(FloatAbi::Hard),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            relocation_model: RelocModel::Static,
            panic_strategy: PanicStrategy::Abort,
            // Armv8-R requires a minimum set of floating-point features equivalent to:
            // fp-armv8, SP-only, with 16 DP (32 SP) registers
            // LLVM defines Armv8-R to include these features automatically.
            //
            // The Cortex-R52 supports these default features and optionally includes:
            // neon-fp-armv8, SP+DP, with 32 DP registers
            //
            // Reference:
            // Arm Cortex-R52 Processor Technical Reference Manual
            // - Chapter 15 Advanced SIMD and floating-point support
            max_atomic_width: Some(64),
            emit_debug_gdb_scripts: false,
            // GCC defaults to 8 for arm-none here.
            c_enum_min_bits: Some(8),
            ..Default::default()
        },
    }
}
