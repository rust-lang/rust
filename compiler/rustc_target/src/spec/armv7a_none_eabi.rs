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
// - `target_os` set to `none`. rationale: matches `thumb` targets
// - `target_{env,vendor}` set to an empty string. rationale: matches `thumb`
// targets
// - `panic_strategy` set to `abort`. rationale: matches `thumb` targets
// - `relocation-model` set to `static`; also no PIE, no relro and no dynamic
// linking. rationale: matches `thumb` targets

use super::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let opts = TargetOptions {
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        linker: Some("rust-lld".to_owned()),
        features: "+v7,+thumb2,+soft-float,-neon,+strict-align".to_string(),
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
        llvm_target: "armv7a-none-eabi".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        options: opts,
    }
}
