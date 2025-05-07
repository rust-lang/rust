use crate::spec::base::xtensa;
use crate::spec::{Target, TargetMetadata, TargetOptions};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "xtensa-none-elf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-v1:8:8-i64:64-i128:128-n32".into(),
        arch: "xtensa".into(),
        metadata: TargetMetadata {
            description: Some("Xtensa ESP32-S3".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },

        options: TargetOptions {
            vendor: "espressif".into(),
            cpu: "esp32s3".into(),
            linker: Some("xtensa-esp32s3-elf-gcc".into()),
            max_atomic_width: Some(32),
            atomic_cas: true,
            ..xtensa::opts()
        },
    }
}
