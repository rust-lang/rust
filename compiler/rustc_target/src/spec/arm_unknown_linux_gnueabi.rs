use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "arm-unknown-linux-gnueabi".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        options: TargetOptions {
            abi: "eabi".to_string(),
            features: "+strict-align,+v6".to_string(),
            max_atomic_width: Some(64),
            mcount: "\u{1}__gnu_mcount_nc".to_string(),
            ..super::linux_gnu_base::opts()
        },
    }
}
