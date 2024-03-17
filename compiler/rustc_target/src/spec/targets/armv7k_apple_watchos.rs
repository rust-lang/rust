use crate::spec::base::apple::{opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::Armv7k;
    const OS: &'static str = "watchos";
    const ABI: TargetAbi = TargetAbi::Normal;

    Target {
        llvm_target: "armv7k-apple-watchos".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-Fi8-i64:64-a:0:32-n32-S128".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".into(),
            max_atomic_width: Some(64),
            dynamic_linking: false,
            position_independent_executables: true,
            ..opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)))
        },
    }
}
