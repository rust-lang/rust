// Generic ARMv7-A target for bare-metal code - floating point enabled (assumes
// FPU is present and emits FPU instructions)
//
// This is basically the `armv7-unknown-linux-gnueabihf` target with some
// changes (list in `armv7a_none_eabi.rs`) to bring it closer to the bare-metal
// `thumb` & `aarch64` targets.

use super::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let opts = TargetOptions {
        vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        linker: Some("rust-lld".to_owned()),
        features: "+v7,+vfp3,-d32,+thumb2,-neon,+strict-align".to_string(),
        executables: true,
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        max_atomic_width: Some(64),
        panic_strategy: PanicStrategy::Abort,
        unsupported_abis: super::arm_base::unsupported_abis(),
        emit_debug_gdb_scripts: false,
        ..Default::default()
    };
    Target {
        llvm_target: "armv7a-none-eabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        options: opts,
    }
}
