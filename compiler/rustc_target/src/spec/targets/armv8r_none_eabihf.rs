// Targets the Little-endian Cortex-R52 processor (ARMv8-R)

use crate::spec::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "armv8r-none-eabihf".into(),
        description: None,
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),

        options: TargetOptions {
            abi: "eabihf".into(),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            relocation_model: RelocModel::Static,
            panic_strategy: PanicStrategy::Abort,
            // The Cortex-R52 has two variants with respect to floating-point support:
            // 1. fp-armv8, SP-only, with 16 DP (32 SP) registers
            // 2. neon-fp-armv8, SP+DP, with 32 DP registers
            // Use the lesser of these two options as the default, as it will produce code
            // compatible with either variant.
            //
            // Reference:
            // Arm Cortex-R52 Processor Technical Reference Manual
            // - Chapter 15 Advanced SIMD and floating-point support
            features: "+fp-armv8,-fp64,-d32".into(),
            max_atomic_width: Some(64),
            emit_debug_gdb_scripts: false,
            // GCC defaults to 8 for arm-none here.
            c_enum_min_bits: Some(8),
            ..Default::default()
        },
    }
}
