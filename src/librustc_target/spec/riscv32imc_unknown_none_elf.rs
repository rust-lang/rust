use crate::spec::{LinkerFlavor, LldFlavor, PanicStrategy,
           Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".to_string(),
        llvm_target: "riscv32".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: "unknown".to_string(),
        arch: "riscv32".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions {
            linker: Some("rust-lld".to_string()),
            cpu: "generic-rv32".to_string(),
            // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=86005
            max_atomic_width: None, //Some(32),
            atomic_cas: false,
            features: "+m,+c".to_string(),
            executables: true,
            panic_strategy: PanicStrategy::Abort,
            relocation_model: "static".to_string(),
            emit_debug_gdb_scripts: false,
            abi_blacklist: super::riscv_base::abi_blacklist(),
            eliminate_frame_pointer: false,
            .. Default::default()
        },
    })
}
