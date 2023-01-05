use super::apple_base::{opts, Arch};
use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let arch = Arch::Armv7k;
    Target {
        llvm_target: "armv7k-apple-watchos".into(),
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-Fi8-i64:64-a:0:32-n32-S128".into(),
        arch: arch.target_arch(),
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".into(),
            max_atomic_width: Some(64),
            forces_embed_bitcode: true,
            dynamic_linking: false,
            position_independent_executables: true,
            // These arguments are not actually invoked - they just have
            // to look right to pass App Store validation.
            bitcode_llvm_cmdline: "-triple\0\
                armv7k-apple-watchos3.0.0\0\
                -emit-obj\0\
                -disable-llvm-passes\0\
                -target-abi\0\
                darwinpcs\0\
                -Os\0"
                .into(),
            ..opts("watchos", arch)
        },
    }
}
