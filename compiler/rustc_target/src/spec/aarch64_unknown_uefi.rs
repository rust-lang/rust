// This defines the aarch64 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options.

use super::uefi_msvc_base;
use crate::spec::{LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut base = uefi_msvc_base::opts();

    base.max_atomic_width = Some(64);

    let pre_link_args_msvc = vec!["/machine:arm64".to_string()];

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().extend(pre_link_args_msvc.clone());
    base.pre_link_args
        .get_mut(&LinkerFlavor::Lld(LldFlavor::Link))
        .unwrap()
        .extend(pre_link_args_msvc);

    Target {
        llvm_target: "aarch64-unknown-windows".to_string(),
        pointer_width: 64,
        data_layout: "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: base,
    }
}
