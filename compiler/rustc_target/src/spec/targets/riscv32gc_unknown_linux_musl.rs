use std::borrow::Cow;

use crate::spec::{CodeModel, SplitDebuginfo, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "riscv32-unknown-linux-musl".into(),
        metadata: TargetMetadata {
            description: Some(
                "RISC-V Linux (kernel 5.4, musl 1.2.3 + RISCV32 support patches".into(),
            ),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-i64:64-n32-S128".into(),
        arch: "riscv32".into(),
        options: TargetOptions {
            code_model: Some(CodeModel::Medium),
            cpu: "generic-rv32".into(),
            features: "+m,+a,+f,+d,+c,+zicsr,+zifencei".into(),
            llvm_abiname: "ilp32d".into(),
            max_atomic_width: Some(32),
            supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Off]),
            // FIXME(compiler-team#422): musl targets should be dynamically linked by default.
            crt_static_default: true,
            ..base::linux_musl::opts()
        },
    }
}
