use crate::abi::Endian;
use crate::spec::{Target, base};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "bpfeb".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("BPF (big endian)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        data_layout: "E-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        pointer_width: 64,
        arch: "bpf".into(),
        options: base::bpf::opts(Endian::Big),
    }
}
