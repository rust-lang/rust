use super::apple_sdk_base::{opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let base = opts("tvos", Arch::X86_64);
    Target {
        llvm_target: "x86_64-apple-tvos".to_string(),
        pointer_width: 64,
        data_layout: "e-m:o-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "x86_64".to_string(),
        options: TargetOptions { max_atomic_width: Some(64), stack_probes: true, ..base },
    }
}
