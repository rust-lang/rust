use crate::spec::{PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions, cvs};

pub(crate) fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        llvm_target: "riscv32".into(),
        metadata: TargetMetadata {
            description: Some("RISC-V ESP-IDF".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        arch: "riscv32".into(),

        options: TargetOptions {
            families: cvs!["unix"],
            os: "espidf".into(),
            env: "newlib".into(),
            vendor: "espressif".into(),
            linker: Some("riscv32-esp-elf-gcc".into()),
            cpu: "generic-rv32".into(),

            // While the RiscV32IMC architecture does not natively support atomics, ESP-IDF does support
            // the __atomic* and __sync* GCC builtins, so setting `max_atomic_width` to `Some(32)`
            // and `atomic_cas` to `true` will cause the compiler to emit libcalls to these builtins.
            //
            // Support for atomics is necessary for the Rust STD library, which is supported by the ESP-IDF framework.
            max_atomic_width: Some(32),
            atomic_cas: true,

            features: "+m,+c".into(),
            llvm_abiname: "ilp32".into(),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            ..Default::default()
        },
    }
}
