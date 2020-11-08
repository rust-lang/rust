use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let mut base = super::cloudabi_base::opts();
    base.cpu = "cortex-a8".to_string();
    base.max_atomic_width = Some(64);
    base.features = "+v7,+vfp3,+neon".to_string();
    base.unsupported_abis = super::arm_base::unsupported_abis();
    base.linker = Some("armv7-unknown-cloudabi-eabihf-cc".to_string());

    Target {
        llvm_target: "armv7-unknown-cloudabi-eabihf".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        options: TargetOptions { target_mcount: "\u{1}mcount".to_string(), ..base },
    }
}
