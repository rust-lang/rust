// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Generic AArch64 target for bare-metal code
//
// Can be used in conjunction with the `target-feature` and
// `target-cpu` compiler flags to opt-in more hardware-specific
// features.
//
// For example, `-C target-cpu=cortex-a53`.

use super::{LldFlavor, LinkerFlavor, Target, TargetOptions, PanicStrategy};

pub fn target() -> Result<Target, String> {
    let opts = TargetOptions {
        linker: Some("rust-lld".to_owned()),
        executables: true,
        relocation_model: "static".to_string(),
        disable_redzone: true,
        linker_is_gnu: true,
        max_atomic_width: Some(128),
        panic_strategy: PanicStrategy::Abort,
        abi_blacklist: super::arm_base::abi_blacklist(),
        .. Default::default()
    };
    Ok(Target {
        llvm_target: "aarch64-unknown-none".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "none".to_string(),
        target_env: "".to_string(),
        target_vendor: "".to_string(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        linker_flavor: LinkerFlavor::Lld(LldFlavor::Ld),
        options: opts,
    })
}
