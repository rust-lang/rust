// Generic AArch64 target for bare-metal code - Floating point enabled
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.
//
// For example, `-C target-cpu=cortex-a53`.

use super::{LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    let opts = TargetOptions {
        linker: Some("rust-lld".to_owned()),
        features: "+strict-align,+neon,+fp-armv8".to_string(),
        executables: true,
        relocation_model: RelocModel::Static,
        disable_redzone: true,
        linker_is_gnu: true,
        max_atomic_width: Some(128),
        panic_strategy: PanicStrategy::Abort,
        unsupported_abis: super::arm_base::unsupported_abis(),
        ..Default::default()
    };
    Target {
        llvm_target: "aarch64-unknown-none".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 64,
        target_c_int_width: "32".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        options: opts,
    }
}
