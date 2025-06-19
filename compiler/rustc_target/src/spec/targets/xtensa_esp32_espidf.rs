use rustc_abi::Endian;

use crate::spec::base::xtensa;
use crate::spec::{Target, TargetMetadata, TargetOptions, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "xtensa-none-elf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-v1:8:8-i64:64-i128:128-n32".into(),
        arch: "xtensa".into(),
        metadata: TargetMetadata { description: None, tier: None, host_tools: None, std: None },

        options: TargetOptions {
            endian: Endian::Little,
            c_int_width: 32,
            families: cvs!["unix"],
            os: "espidf".into(),
            env: "newlib".into(),
            vendor: "espressif".into(),

            executables: true,
            cpu: "esp32".into(),
            linker: Some("xtensa-esp32-elf-gcc".into()),

            // The esp32 only supports native 32bit atomics.
            max_atomic_width: Some(32),
            atomic_cas: true,

            ..xtensa::opts()
        },
    }
}
