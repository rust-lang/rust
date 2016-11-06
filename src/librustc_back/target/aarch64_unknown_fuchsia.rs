// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::{Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::fuchsia_base::opts();
    base.max_atomic_width = Some(128);

    Ok(Target {
        llvm_target: "aarch64-unknown-fuchsia".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        data_layout: "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".to_string(),
        arch: "aarch64".to_string(),
        target_os: "fuchsia".to_string(),
        target_env: "".to_string(),
        target_vendor: "unknown".to_string(),
        options: TargetOptions {
            abi_blacklist: super::arm_base::abi_blacklist(),
            .. base
        },
    })
}
