//! Targets the Nintendo Game Boy Advance (GBA).
//!
//! The GBA is a handheld game device from 2001. Inside, the CPU is an ARM7TDMI.
//! That's in the ARMv4T architecture family.
//!
//! Technically the device has no OS, however we're going to copy the
//! `mipsel_sony_psp` target setup and set the OS string to be "GBA". Other than
//! the setting of the `target_os` and `target_vendor` values, this target is a
//! fairly standard configuration for `thumbv4t`

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "thumbv4t-none-eabi".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "gba".to_string(),
        target_env: String::new(),
        target_vendor: "nintendo".to_string(),
        arch: "arm".to_string(),
        data_layout: "TODO".to_string(),
        linker_flavor: LinkerFlavor::Ld,
        options: TargetOptions {
            // TODO
            ..TargetOptions::default()
        },
    })
}
