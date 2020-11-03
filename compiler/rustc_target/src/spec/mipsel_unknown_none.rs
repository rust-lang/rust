//! Bare MIPS32r2, little endian, softfloat, O32 calling convention
//!
//! Can be used for MIPS M4K core (e.g. on PIC32MX devices)

use crate::spec::abi::Abi;
use crate::spec::{LinkerFlavor, LldFlavor, RelocModel};
use crate::spec::{PanicStrategy, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "mipsel-unknown-none".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64".to_string(),
        arch: "mips".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),

        options: TargetOptions {
            cpu: "mips32r2".to_string(),
            features: "+mips32r2,+soft-float,+noabicalls".to_string(),
            max_atomic_width: Some(32),
            executables: true,
            linker: Some("rust-lld".to_owned()),
            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            unsupported_abis: vec![
                Abi::Stdcall,
                Abi::Fastcall,
                Abi::Vectorcall,
                Abi::Thiscall,
                Abi::Win64,
                Abi::SysV64,
            ],
            emit_debug_gdb_scripts: false,
            ..Default::default()
        },
    }
}
