use crate::spec::{base, LinkerFlavor, Lld, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    base.max_atomic_width = Some(128);
    base.features = "+v8a,+neon,+fp-armv8".into();
    base.late_link_args = TargetOptions::link_args(
        LinkerFlavor::Msvc(Lld::No),
        &["/machine:arm64ec", "softintrin.lib"],
    );

    Target {
        llvm_target: "arm64ec-pc-windows-msvc".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "arm64ec".into(),
        options: base,
    }
}
