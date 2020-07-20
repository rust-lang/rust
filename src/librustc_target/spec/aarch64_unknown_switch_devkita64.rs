use super::{LinkArgs, LinkerFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

// DevkitA64 has custom linker requirements.
const LINKER_SCRIPT: &str = include_str!("./aarch64_unknown_switch_devkita64_script.ld");

pub fn target() -> Result<Target, String> {
    let mut link_args = LinkArgs::new();
    link_args.insert(
        LinkerFlavor::Gcc,
        vec![
            "-march=armv8-a".to_string(),
            "-mtune=cortex-a57".to_string(),
            "-mtp=soft".to_string(),
            "-nodefaultlibs".to_string(),
            "-nostdlib".to_string(),
            "-nostartfiles".to_string(),
            "-L/opt/devkitpro/portlibs/switch/lib".to_string(),
            "-L/opt/devkitpro/libnx/lib".to_string(),
            "-L/opt/devkitpro/devkitA64/lib/gcc/aarch64-none-elf/10.1.0/pic".to_string(),
            "-L/opt/devkitpro/devkitA64/aarch64-none-elf/lib/pic".to_string(),
            "-Wl,--start-group".to_string(),
            "-lgcc".to_string(),
            "-lc".to_string(),
            "-lnx".to_string(),
            "-lsysbase".to_string(),
            "-lm".to_string(),
            "-l:crtbegin.o".to_string(),
            "-l:crtend.o".to_string(),
            "-l:crti.o".to_string(),
            "-l:crtn.o".to_string(),
            "-Wl,--end-group".to_string(),
            "-fPIE".to_string(),
            "-pie".to_string(),
            "-Wl,-z,text".to_string(),
            "-Wl,-z,muldefs".to_string(),
            "-Wl,--export-dynamic".to_string(),
            "-Wl,--eh-frame-hdr".to_string(),
        ],
    );
    let opts = TargetOptions {
        linker: Some("aarch64-none-elf-gcc".to_owned()),
        features: "+a53,+strict-align,+crc,+read-tp-soft".to_string(),
        executables: true,
        relocation_model: RelocModel::Pic,
        disable_redzone: true,
        linker_is_gnu: true,
        max_atomic_width: Some(128),
        panic_strategy: PanicStrategy::Unwind,
        unsupported_abis: super::arm_base::unsupported_abis(),
        target_family: Some("unix".to_string()),
        position_independent_executables: true,
        has_elf_tls: true,
        trap_unreachable: true,
        emit_debug_gdb_scripts: true,
        requires_uwtable: true,
        post_link_args: link_args,
        link_script: Some(LINKER_SCRIPT.to_string()),
        ..Default::default()
    };
    Ok(Target {
        llvm_target: "aarch64-unknown-none".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "switch".to_string(),
        target_env: "devkita64".to_string(),
        target_vendor: "unknown".to_string(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: opts,
    })
}
