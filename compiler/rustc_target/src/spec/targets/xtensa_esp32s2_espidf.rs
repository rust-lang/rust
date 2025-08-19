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
            cpu: "esp32s2".into(),
            linker: Some("xtensa-esp32s2-elf-gcc".into()),

            // See https://github.com/espressif/rust-esp32-example/issues/3#issuecomment-861054477
            //
            // While the ESP32-S2 chip does not natively support atomics, ESP-IDF does support
            // the __atomic* and __sync* compiler builtins. Setting `max_atomic_width` and `atomic_cas`
            // and `atomic_cas: true` will cause the compiler to emit libcalls to these builtins. On the
            // ESP32-S2, these are guaranteed to be lock-free.
            //
            // Support for atomics is necessary for the Rust STD library, which is supported by ESP-IDF.
            max_atomic_width: Some(32),
            atomic_cas: true,

            ..xtensa::opts()
        },
    }
}
