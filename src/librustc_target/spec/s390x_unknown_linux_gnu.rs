// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use spec::{LinkerFlavor, Target, TargetResult};

pub fn target() -> TargetResult {
    let mut base = super::linux_base::opts();
    // z10 is the oldest CPU supported by LLVM
    base.cpu = "z10".to_string();
    // FIXME: The data_layout string below and the ABI implementation in
    // cabi_s390x.rs are for now hard-coded to assume the no-vector ABI.
    // Pass the -vector feature string to LLVM to respect this assumption.
    base.features = "-vector".to_string();
    base.max_atomic_width = Some(64);
    // see #36994
    base.exe_allocation_crate = None;
    base.min_global_align = Some(16);

    Ok(Target {
        llvm_target: "s390x-unknown-linux-gnu".to_string(),
        target_endian: "big".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64".to_string(),
        arch: "s390x".to_string(),
        target_os: "linux".to_string(),
        target_env: "gnu".to_string(),
        target_vendor: "unknown".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
