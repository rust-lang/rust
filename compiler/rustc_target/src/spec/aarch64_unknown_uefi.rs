// Copyright 2021 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This defines the aarch64 target for UEFI systems as described in the UEFI specification. See the
// uefi-base module for generic UEFI options. On aarch64 systems (mostly called "arm64" in the spec)
// UEFI systems always run in long-mode, have the interrupt-controller pre-configured and force a
// single-CPU execution.
// The win64 ABI is used. It differs from the sysv64 ABI, so we must use a windows target with
// LLVM. "aarch64-unknown-windows" is used to get the minimal subset of windows-specific features.

use crate::spec::{CodeModel, LinkerFlavor, LldFlavor, Target};

pub fn target() -> Target {
    let mut base = super::uefi_msvc_base::opts();
    base.cpu = "aarch64";
    base.max_atomic_width = Some(64);

    // We disable MMX and SSE for now. UEFI does not prevent these from being used, but there have
    // been reports to GRUB that some firmware does not initialize the FP exception handlers
    // properly. Therefore, using FP coprocessors will end you up at random memory locations when
    // you throw FP exceptions.
    // To be safe, we disable them for now and force soft-float. This can be revisited when we
    // have more test coverage. Disabling FP served GRUB well so far, so it should be good for us
    // as well.
    base.features = "-mmx,-sse,+soft-float".to_string();

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

    Ok(Target {
        llvm_target: "aarch64-unknown-windows".to_string(),
        pointer_width: 64,
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),

        options: base,
    })
}
