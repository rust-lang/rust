use crate::spec::{LinkerFlavor, Target, TargetOptions};

/// A base target for AVR devices using the GNU toolchain.
///
/// Requires GNU avr-gcc and avr-binutils on the host system.
pub fn target(target_cpu: String) -> Target {
    Target {
        arch: "avr".to_string(),
        data_layout: "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8".to_string(),
        llvm_target: "avr-unknown-unknown".to_string(),
        pointer_width: 16,
        options: TargetOptions {
            target_c_int_width: "16".to_string(),
            target_os: "unknown".to_string(),
            cpu: target_cpu.clone(),
            exe_suffix: ".elf".to_string(),

            linker: Some("avr-gcc".to_owned()),
            dynamic_linking: false,
            executables: true,
            linker_is_gnu: true,
            has_rpath: false,
            position_independent_executables: false,
            eh_frame_header: false,
            pre_link_args: vec![(
                LinkerFlavor::Gcc,
                vec![
                    format!("-mmcu={}", target_cpu),
                    // We want to be able to strip as much executable code as possible
                    // from the linker command line, and this flag indicates to the
                    // linker that it can avoid linking in dynamic libraries that don't
                    // actually satisfy any symbols up to that point (as with many other
                    // resolutions the linker does). This option only applies to all
                    // following libraries so we're sure to pass it as one of the first
                    // arguments.
                    "-Wl,--as-needed".to_string(),
                ],
            )]
            .into_iter()
            .collect(),
            late_link_args: vec![(LinkerFlavor::Gcc, vec!["-lgcc".to_owned()])]
                .into_iter()
                .collect(),
            max_atomic_width: Some(0),
            atomic_cas: false,
            ..TargetOptions::default()
        },
    }
}
