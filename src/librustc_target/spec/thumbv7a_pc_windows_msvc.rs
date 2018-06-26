// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkerFlavor, Target, TargetOptions, TargetResult, PanicStrategy};

pub fn target() -> TargetResult {
    let mut base = super::windows_msvc_base::opts();

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(
        "/LIBPATH:C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.11.25503\\lib\\arm".to_string());

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(
        "/LIBPATH:C:\\Program Files (x86)\\Windows Kits\\10\\lib\\10.0.17134.0\\ucrt\\arm".to_string());

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(
        "/LIBPATH:C:\\Program Files (x86)\\Windows Kits\\10\\lib\\10.0.17134.0\\um\\arm".to_string());

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(
        "/MACHINE:ARM".to_string());

    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(
        "/INCREMENTAL:NO".to_string());

    // Prevent error LNK2013: BRANCH24(T) fixup overflow
    base.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().push(
        "/OPT:NOLBR".to_string());

    base.panic_strategy = PanicStrategy::Abort;

    Ok(Target {
        llvm_target: "thumbv7a-pc-windows-msvc".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:w-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "windows".to_string(),
        target_env: "msvc".to_string(),
        target_vendor: "pc".to_string(),
        linker_flavor: LinkerFlavor::Msvc,

        options: TargetOptions {
            features: "+v7,+thumb-mode,+vfp3,+d16,+thumb2,+neon".to_string(),
            cpu: "generic".to_string(),
            max_atomic_width: Some(64),
            abi_blacklist: super::arm_base::abi_blacklist(),
            .. base
        }
    })
}
