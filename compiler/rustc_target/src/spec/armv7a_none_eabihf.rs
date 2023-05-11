// Generic ARMv7-A target for bare-metal code - floating point enabled (assumes
// FPU is present and emits FPU instructions)
//
// This is basically the `armv7-unknown-linux-gnueabihf` target with some
// changes (list in `armv7a_none_eabi.rs`) to bring it closer to the bare-metal
// `thumb` & `aarch64` targets.

use super::{Cc, LinkerFlavor, Lld, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let opts = TargetOptions {
        abi: "eabihf".into(),
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
        linker: Some("rust-lld".into()),
        features: "+v7,+vfp3,-d32,+thumb2,-neon,+strict-align".into(),
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        max_atomic_width: Some(64),
        panic_strategy: PanicStrategy::Abort,
        emit_debug_gdb_scripts: false,
        // GCC and Clang default to 8 for arm-none here
        c_enum_min_bits: Some(8),
        ..Default::default()
    };
    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: opts,
    }
}
