use crate::spec::{FramePointer, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::windows_gnullvm::opts();
    base.max_atomic_width = Some(128);
    base.features = "+v8a,+neon,+fp-armv8".into();
    base.linker = Some("aarch64-w64-mingw32-clang".into());

    // Microsoft recommends enabling frame pointers on Arm64 Windows.
    // From https://learn.microsoft.com/en-us/cpp/build/arm64-windows-abi-conventions?view=msvc-170#integer-registers
    // "The frame pointer (x29) is required for compatibility with fast stack walking used by ETW
    // and other services. It must point to the previous {x29, x30} pair on the stack."
    base.frame_pointer = FramePointer::NonLeaf;

    Target {
        llvm_target: "aarch64-pc-windows-gnu".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 MinGW (Windows 10+), LLVM ABI".into()),
            tier: Some(2),
            host_tools: Some(false),
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
