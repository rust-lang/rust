use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.max_atomic_width = Some(128);
    base.supports_xray = true;
    base.features = "+v8a".into();

    Target {
        llvm_target: "aarch64-unknown-linux-musl".into(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions { mcount: "\u{1}_mcount".into(), ..base },
    }
}
