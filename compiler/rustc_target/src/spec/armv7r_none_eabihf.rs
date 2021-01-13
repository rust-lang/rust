// Targets the Little-endian Cortex-R4F/R5F processor (ARMv7-R)

use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "armv7r-unknown-none-eabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
            executables: true,
            linker: Some("rust-lld".to_owned()),
            relocation_model: RelocModel::Static,
            panic_strategy: PanicStrategy::Abort,
            features: "+vfp3,-d32,-fp16".to_string(),
            max_atomic_width: Some(32),
            unsupported_abis: super::arm_base::unsupported_abis(),
            emit_debug_gdb_scripts: false,
            ..Default::default()
        },
    }
}
