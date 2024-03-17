// This defines the aarch64 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options.

use crate::spec::{base, LinkerFlavor, Lld, MaybeLazy, Target, TargetOptions};

pub fn target() -> Target {
    let mut base = base::uefi_msvc::opts();

    base.max_atomic_width = Some(128);
    base.pre_link_args = MaybeLazy::lazy(|| {
        TargetOptions::link_args(LinkerFlavor::Msvc(Lld::No), &["/machine:arm64"])
    });
    base.features = "+v8a".into();

    Target {
        llvm_target: "aarch64-unknown-windows".into(),
        metadata: crate::spec::TargetMetadata {
            description: None,
            tier: None,
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout: "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
