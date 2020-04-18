use super::{LinkerFlavor, PanicStrategy, Target, TargetOptions};

pub fn target() -> Result<Target, String> {
    let opts = TargetOptions {
        linker: Some("aarch64-none-elf-gcc".to_owned()),
        features: "+a53,+strict-align,+crc".to_string(),
        executables: true,
        relocation_model: "pic".to_string(),
        disable_redzone: true,
        linker_is_gnu: true,
        max_atomic_width: Some(128),
        panic_strategy: PanicStrategy::Unwind,
        abi_blacklist: super::arm_base::abi_blacklist(),
        target_family: Some("unix".to_string()),
        position_independent_executables: true,
        has_elf_tls: false,
        trap_unreachable: true,
        emit_debug_gdb_scripts: true,
        requires_uwtable: true,
        ..Default::default()
    };
    Ok(Target {
        llvm_target: "aarch64-unknown-none".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "horizon".to_string(),
        target_env: "newlib".to_string(),
        target_vendor: "libnx".to_string(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: opts,
    })
}
