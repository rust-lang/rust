// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::Target;
use target::TargetOptions;
use std::default::Default;

pub fn target() -> Target {
    Target {
        llvm_target: "i686-unknown-haiku".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128".to_string(),
        arch: "x86".to_string(),
        target_os: "haiku".to_string(),
        target_env: "".to_string(),
        target_vendor: "unknown".to_string(),
        options: TargetOptions {
            linker: "cc".to_string(),
            dynamic_linking: true,
            executables: true,
            has_rpath: true,
            .. Default::default()
        },
    }
}
