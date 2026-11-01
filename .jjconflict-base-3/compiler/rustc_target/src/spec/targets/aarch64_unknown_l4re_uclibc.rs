use crate::spec::{Arch, Cc, LinkerFlavor, Target, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::l4re::opts();

    let extra_link_args = &["-zmax-page-size=0x1000", "-zcommon-page-size=0x1000"];
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::Yes), extra_link_args);
    base.add_pre_link_args(LinkerFlavor::Unix(Cc::No), extra_link_args);

    Target {
        llvm_target: "aarch64-unknown-l4re-uclibc".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Arm64 L4Re".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 64,
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: Arch::AArch64,
        options: TargetOptions {
            features: "+v8a".into(),
            mcount: "__mcount".into(),
            max_atomic_width: Some(128),
            ..base
        }
    }
}
