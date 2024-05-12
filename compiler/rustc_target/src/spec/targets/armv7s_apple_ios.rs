use crate::spec::base::apple::{ios_llvm_target, opts, pre_link_args, Arch, TargetAbi};
use crate::spec::{MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    const ARCH: Arch = Arch::Armv7s;
    const OS: &'static str = "ios";
    const ABI: TargetAbi = TargetAbi::Normal;

    Target {
        llvm_target: MaybeLazy::lazy(|| ios_llvm_target(ARCH)),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 32,
        data_layout: "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32".into(),
        arch: ARCH.target_arch(),
        options: TargetOptions {
            features: "+v7,+vfp4,+neon".into(),
            max_atomic_width: Some(64),
            ..opts(OS, ARCH, ABI, MaybeLazy::lazy(|| pre_link_args(OS, ARCH, ABI)))
        },
    }
}
