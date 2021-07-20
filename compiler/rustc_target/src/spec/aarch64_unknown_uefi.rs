// This defines the amd64 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options. On aarch64 systems UEFI systems always run in long-mode,
// have the interrupt-controller pre-configured and force a single-CPU execution.
//
// The win64 ABI is used. It differs from the sysv64 ABI, so we must use a windows target with
// LLVM. "aarch64-unknown-windows" is used to get the minimal subset of windows-specific features.

use super::uefi_msvc_base;
use crate::spec::{CodeModel, LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut base = uefi_msvc_base::opts();

    base.max_atomic_width = Some(64);

    let pre_link_args_msvc = vec!["/machine:arm64".to_string()];

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().extend(pre_link_args_msvc.clone());
    base.pre_link_args
        .get_mut(&LinkerFlavor::Lld(LldFlavor::Link))
        .unwrap()
        .extend(pre_link_args_msvc);

    // UEFI systems run without a host OS, hence we cannot assume any code locality. We must tell
    // LLVM to expect code to reference any address in the address-space. The "large" code-model
    // places no locality-restrictions, so it fits well here.
    base.code_model = Some(CodeModel::Large);

    Target {
        llvm_target: "aarch64-unknown-windows".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        options: base,
    }
}
