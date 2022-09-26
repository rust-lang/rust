use crate::spec::{LinkerFlavor, RelocModel, Target, TargetOptions};

/// A base target for AVR devices using the GNU toolchain.
///
/// Requires GNU avr-gcc and avr-binutils on the host system.
/// FIXME: Remove the second parameter when const string concatenation is possible.
pub fn target(target_cpu: &'static str, mmcu: &'static str) -> Target {
    Target {
        arch: "avr".into(),
        data_layout: "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8".into(),
        llvm_target: "avr-unknown-unknown".into(),
        pointer_width: 16,
        options: TargetOptions {
            c_int_width: "16".into(),
            cpu: target_cpu.into(),
            exe_suffix: ".elf".into(),

            linker: Some("avr-gcc".into()),
            eh_frame_header: false,
            pre_link_args: TargetOptions::link_args(LinkerFlavor::Gcc, &[mmcu]),
            late_link_args: TargetOptions::link_args(LinkerFlavor::Gcc, &["-lgcc"]),
            max_atomic_width: Some(0),
            atomic_cas: false,
            relocation_model: RelocModel::Static,
            ..TargetOptions::default()
        },
    }
}
