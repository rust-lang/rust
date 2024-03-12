use std::borrow::Cow;

use crate::spec::{base, CodeModel, SplitDebuginfo, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "riscv32-unknown-linux-gnu".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        arch: "riscv32".into(),
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv32".into(),
            features: "+m,+a,+f,+d,+c".into(),
            llvm_abiname: "ilp32d".into(),
            max_atomic_width: Some(32),
            supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Off]),
            ..base::linux_gnu::opts()
        },
    }
}
