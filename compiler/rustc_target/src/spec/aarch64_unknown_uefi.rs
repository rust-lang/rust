// This defines the aarch64 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options.

use super::uefi_msvc_base;
use crate::spec::{LinkerFlavor, Lld, Target};

pub fn target() -> Target {
    let mut base = uefi_msvc_base::opts();

    base.max_atomic_width = Some(128);
    base.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/machine:arm64"]);
    base.features = "+v8a".into();

    Target {
        llvm_target: "aarch64-unknown-windows".into(),
        pointer_width: 64,
        data_layout: "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128".into(),
        arch: "aarch64".into(),
        options: base,
    }
}
