use rustc_abi::Endian;

use crate::spec::{StackProbeType, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "aarch64_be-unknown-netbsd".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 NetBSD (big-endian)".into()),
            tier: Some(3),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "E-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            mcount: "__mcount".into(),
            max_atomic_width: Some(128),
            stack_probes: StackProbeType::Inline,
            endian: Endian::Big,
            ..base::netbsd::opts()
        },
    }
}
