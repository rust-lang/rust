// Generic x86-64 target for bare-metal code - Floating point disabled
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.

use super::{
    CodeModel, LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, RelroLevel, StackProbeType,
    Target, TargetOptions,
};

pub fn target() -> Target {
    let opts = TargetOptions {
        cpu: "x86-64".to_string(),
        max_atomic_width: Some(64),
        // don't use probe-stack=inline-asm until rust#83139 and rust#84667 are resolved
        stack_probes: StackProbeType::Call,
        position_independent_executables: true,
        static_position_independent_executables: true,
        relro_level: RelroLevel::Full,
        relocation_model: RelocModel::Pic,
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        linker: Some("rust-lld".to_owned()),
        features:
            "-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-3dnow,-3dnowa,-avx,-avx2,+soft-float"
                .to_string(),
        executables: true,
        disable_redzone: true,
        panic_strategy: PanicStrategy::Abort,
        code_model: Some(CodeModel::Kernel),
        ..Default::default()
    };
    Target {
        llvm_target: "x86_64-unknown-none-elf".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),
        options: opts,
    }
}
