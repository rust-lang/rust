use crate::spec::{Target, TargetOptions};

use super::SanitizerSet;

pub fn target() -> Target {
    let mut base = super::linux_musl_base::opts();
    base.env = "ohos".into();
    base.crt_static_default = false;
    base.max_atomic_width = Some(128);

    Target {
        // LLVM 15 doesn't support OpenHarmony yet, use a linux target instead.
        llvm_target: "aarch64-unknown-linux-musl".into(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+reserve-x18".into(),
            mcount: "\u{1}_mcount".into(),
            force_emulated_tls: true,
            supported_sanitizers: SanitizerSet::ADDRESS
                | SanitizerSet::CFI
                | SanitizerSet::LEAK
                | SanitizerSet::MEMORY
                | SanitizerSet::MEMTAG
                | SanitizerSet::THREAD
                | SanitizerSet::HWADDRESS,
            ..base
        },
    }
}
