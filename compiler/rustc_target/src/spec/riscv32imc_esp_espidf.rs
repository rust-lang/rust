use crate::spec::{LinkerFlavor, PanicStrategy, RelocModel};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".to_string(),
        llvm_target: "riscv32".to_string(),
        pointer_width: 32,
        arch: "riscv32".to_string(),

        options: TargetOptions {
            families: vec!["unix".to_string()],
            os: "espidf".to_string(),
            env: "newlib".to_string(),
            vendor: "espressif".to_string(),
            linker_flavor: LinkerFlavor::Gcc,
            linker: Some("riscv32-esp-elf-gcc".to_string()),
            cpu: "generic-rv32".to_string(),

            // While the RiscV32IMC architecture does not natively support atomics, ESP-IDF does support
            // the __atomic* and __sync* GCC builtins, so setting `max_atomic_width` to `Some(64)`
            // and `atomic_cas` to `true` will cause the compiler to emit libcalls to these builtins.
            //
            // Support for atomics is necessary for the Rust STD library, which is supported by the ESP-IDF framework.
            max_atomic_width: Some(64),
            atomic_cas: true,

            features: "+m,+c".to_string(),
            executables: true,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            emit_debug_gdb_scripts: false,
            eh_frame_header: false,
            ..Default::default()
        },
    }
}
