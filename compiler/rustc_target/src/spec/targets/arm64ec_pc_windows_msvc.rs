use crate::spec::{FramePointer, LinkerFlavor, Lld, Target, TargetMetadata, add_link_args, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_msvc::opts();
    base.max_atomic_width = Some(128);
    base.features = "+v8a,+neon,+fp-armv8".into();
    add_link_args(
        &mut base.late_link_args,
        LinkerFlavor::Msvc(Lld::No),
        &["/machine:arm64ec", "softintrin.lib"],
    );

    // Microsoft recommends enabling frame pointers on Arm64 Windows.
    // From https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170#integer-registers
    // "The frame pointer (x29) is required for compatibility with fast stack walking used by ETW
    // and other services. It must point to the previous {x29, x30} pair on the stack."
    base.frame_pointer = FramePointer::NonLeaf;

    Target {
        llvm_target: "arm64ec-pc-windows-msvc".into(),
        metadata: TargetMetadata {
            description: Some("Arm64EC Windows MSVC".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
                .into(),
        arch: "arm64ec".into(),
        options: base,
    }
}
