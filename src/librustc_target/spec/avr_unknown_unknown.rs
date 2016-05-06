use crate::spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "avr-unknown-unknown".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "16".to_string(),
        data_layout: "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8-a:8".to_string(),
        arch: "avr".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        target_os: "unknown".to_string(),
        target_env: "".to_string(),
        target_vendor: "unknown".to_string(),
        target_c_int_width: 16.to_string(),
        options: super::none_base::opts(),
    })
}
