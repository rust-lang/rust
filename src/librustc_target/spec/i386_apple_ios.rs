use crate::spec::{LinkerFlavor, Target, TargetOptions, TargetResult};
use super::apple_ios_base::{opts, Arch};

pub fn target() -> TargetResult {
    let base = opts(Arch::I386)?;
    Ok(Target {
        llvm_target: "i386-apple-ios".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128".to_string(),
        arch: "x86".to_string(),
        target_os: "ios".to_string(),
        target_env: String::new(),
        target_vendor: "apple".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: TargetOptions {
            max_atomic_width: Some(64),
            stack_probes: true,
            .. base
        }
    })
}
