use crate::spec::{abi::Abi, LinkerFlavor, PanicStrategy, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "xtensa-none-elf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32".to_string(),
        arch: "xtensa".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            cpu: "esp8266".to_string(),
            linker: Some("xtensa-esp32-elf-ld".to_string()),

            max_atomic_width: Some(32),

            // Because these devices have very little resources having an
            // unwinder is too onerous so we default to "abort" because the
            // "unwind" strategy is very rare.
            panic_strategy: PanicStrategy::Abort,

            // Similarly, one almost always never wants to use relocatable
            // code because of the extra costs it involves.
            relocation_model: "static".to_string(),

            // See the thumb_base.rs file for an explanation of this value
            emit_debug_gdb_scripts: false,

            abi_blacklist: vec![
                Abi::Stdcall,
                Abi::Fastcall,
                Abi::Vectorcall,
                Abi::Thiscall,
                Abi::Win64,
                Abi::SysV64,
            ],

            ..Default::default()
        },
    })
}
