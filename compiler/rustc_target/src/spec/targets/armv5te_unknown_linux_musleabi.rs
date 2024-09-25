use crate::spec::{Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    Target {
        // FIXME: this comment below does not seem applicable?
        // It's important we use "gnueabihf" and not "musleabihf" here. LLVM
        // uses it to determine the calling convention and float ABI, and LLVM
        // doesn't support the "musleabihf" value.
        llvm_target: "armv5te-unknown-linux-gnueabi".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Armv5TE Linux with musl 1.2.3".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: TargetOptions {
            abi: "eabi".into(),
            features: "+soft-float,+strict-align".into(),
            // Atomic operations provided by compiler-builtins
            max_atomic_width: Some(32),
            mcount: "\u{1}mcount".into(),
            has_thumb_interworking: true,
            ..base::linux_musl::opts()
        },
    }
}
