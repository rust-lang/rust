use crate::spec::{base, Target};

pub fn target() -> Target {
    let mut base = base::teeos::opts();
    base.features = "+strict-align,+neon,+fp-armv8".into();
    base.max_atomic_width = Some(128);
    base.linker = Some("aarch64-linux-gnu-ld".into());

    Target {
        llvm_target: "aarch64-unknown-none".into(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
