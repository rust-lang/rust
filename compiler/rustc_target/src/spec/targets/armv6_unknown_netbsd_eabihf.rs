use crate::spec::{base, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "armv6-unknown-netbsdelf-eabihf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabihf".into(),
            // FIXME: remove env when cfg_target_abi becomes stable
            env: "eabihf".into(),
            features: "+v6,+vfp2,-d32".into(),
            max_atomic_width: Some(64),
            mcount: "__mcount".into(),
            ..base::netbsd::opts()
        },
    }
}
