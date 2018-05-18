// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Targets the Cortex-R4F/R5F processor (ARMv7-R)

use std::default::Default;
use spec::{LinkerFlavor, PanicStrategy, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "armebv7r-none-eabihf".to_string(),
        target_endian: "big".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        arch: "arm".to_string(),
        target_os: "none".to_string(),
        target_env: "".to_string(),
        target_vendor: "".to_string(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            executables: true,
            relocation_model: "static".to_string(),
            panic_strategy: PanicStrategy::Abort,
            features: "+v7,+vfp3,+d16,+fp-only-sp".to_string(),
            max_atomic_width: Some(32),
            abi_blacklist: super::arm_base::abi_blacklist(),
            emit_debug_gdb_scripts: false,
            .. Default::default()
        },
    })
}
