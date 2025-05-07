// This defines the aarch64 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options.

use crate::spec::{LinkerFlavor, Lld, Target, TargetMetadata, base};

pub(crate) fn target() -> Target {
    let mut base = base::uefi_msvc::opts();

    base.max_atomic_width = Some(128);
    base.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/machine:arm64"]);
    base.features = "+v8a".into();

    Target {
        llvm_target: "aarch64-unknown-windows".into(),
        metadata: TargetMetadata {
            description: Some("ARM64 UEFI".into()),
            tier: Some(2),
            host_tools: Some(false),
            std: None, // ?
        },
        pointer_width: 64,
        data_layout:
            "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
                .into(),
        arch: "aarch64".into(),
        options: base,
    }
}
