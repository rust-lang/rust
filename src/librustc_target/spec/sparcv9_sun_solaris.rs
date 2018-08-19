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
    let mut base = super::solaris_base::opts();
    base.pre_link_args.insert(LinkerFlavor::Gcc, vec!["-m64".to_string()]);
    // llvm calls this "v9"
    base.cpu = "v9".to_string();
    base.max_atomic_width = Some(64);
    base.exe_allocation_crate = None;

    Ok(Target {
        llvm_target: "sparcv9-sun-solaris".to_string(),
        target_endian: "big".to_string(),
        target_pointer_width: "64".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "E-m:e-i64:64-n32:64-S128".to_string(),
        // Use "sparc64" instead of "sparcv9" here, since the former is already
        // used widely in the source base.  If we ever needed ABI
        // differentiation from the sparc64, we could, but that would probably
        // just be confusing.
        arch: "sparc64".to_string(),
        target_os: "solaris".to_string(),
        target_env: "".to_string(),
        target_vendor: "sun".to_string(),
        linker_flavor: LinkerFlavor::Gcc,
        options: base,
    })
}
