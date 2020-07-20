use super::{LinkerFlavor, PanicStrategy, Target, TargetOptions, RelocModel, LinkArgs};

// DevkitA64 has custom linker requirements.
const LINKER_SCRIPT: &str = include_str!("./aarch64_unknown_switch_devkita64_script.ld");

pub fn target() -> Result<Target, String> {
    let mut pre_link_args = LinkArgs::new();
    pre_link_args.insert(LinkerFlavor::Gcc, vec![
        // from https://github.com/switchbrew/switch-examples/blob/master/templates/application/Makefile#L50
        "-march=armv8-a".to_string(),
        "-mtune=cortex-a57".to_string(),
        "-mtp=soft".to_string(),
        // from https://github.com/switchbrew/switch-examples/blob/master/templates/application/Makefile#L68
        "-L/opt/devkitpro/portlibs/switch/lib".to_string(),
        "-L/opt/devkitpro/libnx/lib".to_string(),
        // from https://github.com/switchbrew/libnx/blob/master/nx/switch.specs
        "-nostartfiles".to_string(),
        "-l:crti.o".to_string(),
        "-l:crtbegin.o".to_string(),
    ]);
    let mut late_link_args = LinkArgs::new();
    late_link_args.insert(LinkerFlavor::Gcc, vec![
        // from https://github.com/switchbrew/switch-examples/blob/master/templates/application/Makefile#L62
        "-lnx".to_string(),
    ]);
    let mut post_link_args = LinkArgs::new();
    post_link_args.insert(LinkerFlavor::Gcc, vec![
        // from https://github.com/switchbrew/libnx/blob/master/nx/switch.specs
        "-Wl,-z,text".to_string(),
        "-Wl,-z,nodynamic-undefined-weak".to_string(),
        "-Wl,--build-id=sha1".to_string(),
        "-Wl,--nx-module-name".to_string(),
        // from https://github.com/switchbrew/libnx/blob/master/nx/switch.specs (implicit)
        "-l:crtend.o".to_string(),
        // this shouldn't be necessary (it isn't used in C), maybe it's needed because libc explicitly links `-lc`?
        "-Wl,-z,muldefs".to_string(),
        // force an _DYNAMIC symbol to be added even in static binaries (needed by crt0)
        "-Wl,--export-dynamic".to_string(),
    ]);
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
        static_position_independent_executables: true,
        has_elf_tls: true,
        crt_static_default: true,
        trap_unreachable: true,
        emit_debug_gdb_scripts: true,
        requires_uwtable: true,
        pre_link_args,
        late_link_args,
        post_link_args,
        no_default_libraries: false,
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
