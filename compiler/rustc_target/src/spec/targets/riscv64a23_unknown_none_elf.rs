use crate::spec::{
    Arch, Cc, CodeModel, LinkerFlavor, Lld, LlvmAbi, PanicStrategy, RelocModel, SanitizerSet,
    Target, TargetMetadata, TargetOptions,
};

/// Bare-metal RISC-V target following the RVA23U64 mandatory profile.
///
/// RVA23U64 is the 2023 application-class mandatory profile for 64-bit
/// RISC-V user-mode code.  It includes:
///
///   - `I`, `M`, `A`, `F`, `D`, `C` (standard IMAFC + compressed)
///   - `Zicsr`, `Zifencei`  (CSR access and instruction-stream fence)
///   - `Zba`, `Zbb`, `Zbs`  (bit-manipulation, Bitmanip subset)
///   - `Zic64b`             (64-byte cache-block operations)
///   - `Ziccamoa`           (atomics at cache-block granularity)
///   - `Ziccif`             (instruction-fetch ordering)
///   - `Zicclsm`            (misaligned load/store support)
///   - `Ziccrse`            (reservation-set events)
///   - `Za64rs`             (64-byte reservation-set size)
///   - `Zawrs`              (WRS.STO / WRS.NTO wait-on-reservation-set)
///   - `Zfhmin`             (half-precision load/store and convert)
///   - `Zkt`                (constant-time instructions)
///   - `Zvbb`, `Zvfhmin`    (vector bit-manip and half-precision vector ops)
///   - `Zvkn`               (NIST crypto suite for RVV)
///   - `Zvkt`               (vector constant-time instructions)
///   - `V` (and `Zvl128b`)  (standard 128-bit+ vector extension)
///
/// This target is intended for capability-native, bare-metal kernel and
/// runtime development on modern server-class RISC-V hardware (e.g. the
/// Dominion and Feox projects).  It uses the same conservative link model
/// as `riscv64gc-unknown-none-elf` (static PIC, medium code model, abort
/// on panic) so that it integrates cleanly with bare-metal linker scripts.
pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        metadata: TargetMetadata {
            description: Some("Bare RISC-V (RVA23U64 mandatory profile)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        llvm_target: "riscv64".into(),
        pointer_width: 64,
        arch: Arch::RiscV64,

        options: TargetOptions {
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            linker: Some("rust-lld".into()),
            llvm_abiname: LlvmAbi::Lp64d,
            // Use the LLVM RVA23U64 feature string so that the backend can
            // select the correct instruction scheduling model and
            // auto-vectorisation decisions.
            cpu: "generic-rv64".into(),
            features: "+rva23u64".into(),
            max_atomic_width: Some(64),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            code_model: Some(CodeModel::Medium),
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            supported_sanitizers: SanitizerSet::KERNELADDRESS | SanitizerSet::SHADOWCALLSTACK,
            ..Default::default()
        },
    }
}
