use crate::spec::{LinkerFlavor, Lld, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    base.max_atomic_width = Some(128);
    base.features = "+v8a,+neon,+fp-armv8".into();

    // MSVC emits a warning about code that may trip "Cortex-A53 MPCore processor bug #843419" (see
    // https://developer.arm.com/documentation/epm048406/latest) which is sometimes emitted by LLVM.
    // Since Arm64 Windows 10+ isn't supported on that processor, it's safe to disable the warning.
    base.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/arm64hazardfree"]);

    Target {
        llvm_target: "aarch64-pc-windows-msvc".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 Windows MSVC".into()),
            tier: Some(2),
            host_tools: Some(true),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
                .into(),
        arch: "aarch64".into(),
        options: base,
    }
}
