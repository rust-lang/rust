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
            c_int_width: "16".to_string(),
            cpu: target_cpu.clone(),
            exe_suffix: ".elf".to_string(),

            linker: Some("avr-gcc".to_owned()),
            executables: true,
            eh_frame_header: false,
            pre_link_args: [(LinkerFlavor::Gcc, vec![format!("-mmcu={}", target_cpu)])]
                .into_iter()
                .collect(),
            late_link_args: [(LinkerFlavor::Gcc, vec!["-lgcc".to_owned()])].into_iter().collect(),
            max_atomic_width: Some(0),
            atomic_cas: false,
            ..TargetOptions::default()
        },
    }
}
