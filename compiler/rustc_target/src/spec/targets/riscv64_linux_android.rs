use std::borrow::Cow;

use crate::spec::{base, CodeModel, SanitizerSet, SplitDebuginfo, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "riscv64-linux-android".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        arch: "riscv64".into(),
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv64".into(),
            features: "+m,+a,+f,+d,+c,+zba,+zbb,+zbs,+v".into(),
            llvm_abiname: "lp64d".into(),
            supported_sanitizers: SanitizerSet::ADDRESS,
            max_atomic_width: Some(64),
            supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Off]),
            ..base::android::opts()
        },
    }
}
