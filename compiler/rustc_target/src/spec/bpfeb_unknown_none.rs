use crate::spec::Target;
use crate::{abi::Endian, spec::bpf_base};

pub fn target() -> Target {
    Target {
        llvm_target: "bpfeb".to_string(),
        data_layout: "E-m:e-p:64:64-i64:64-i128:128-n32:64-S128".to_string(),
        pointer_width: 64,
        arch: "bpf".to_string(),
        options: bpf_base::opts(Endian::Big),
    }
}
