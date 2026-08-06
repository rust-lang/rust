use crate::spec::{
    Arch, Cc, LinkerFlavor, Lld, LlvmAbi, PanicStrategy, RelocModel, Target, TargetMetadata,
    TargetOptions,
};

// Bare-metal RV32IMFC for cores that have hardware single-precision float (the `F`
// extension, with the `ilp32f` ABI) but NO atomic ('a') extension.
// This is `riscv32imafc-unknown-none-elf` MINUS the atomic extension, handled the
// same way the in-tree `riscv32imc-unknown-none-elf` handles a no-`a` core:
// `+forced-atomics` makes atomic load/store lower to plain ld/st (sound on a single
// hart) while `atomic_cas = false` keeps RMW/CAS off — downstream crates use a
// critical-section polyfill (e.g. portable-atomic) for those. No lr.w/sc.w/amo* are
// ever emitted, so it does not trap on a core without the A extension.
pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        llvm_target: "riscv32".into(),
        metadata: TargetMetadata {
            description: Some(
                "Bare RISC-V (RV32IMFC ISA, hardware single-float, no atomics)".into(),
            ),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        arch: Arch::RiscV32,

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            cpu: "generic-rv32".into(),
            max_atomic_width: Some(32),
            atomic_cas: false,
            features: "+m,+f,+c,+forced-atomics".into(),
            llvm_abiname: LlvmAbi::Ilp32f,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            ..Default::default()
        },
    }
}
