use crate::spec::Target;
use crate::{abi::Endian, spec::base};

pub fn target() -> Target {
    Target {
        llvm_target: "bpfel".into(),
        description: None,
        data_layout: "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128".into(),
        pointer_width: 64,
        arch: "bpf".into(),
        options: base::bpf::opts(Endian::Little),
    }
}
