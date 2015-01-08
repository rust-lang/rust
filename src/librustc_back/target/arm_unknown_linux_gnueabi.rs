// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::{Target, TargetOptions};

pub fn target() -> Target {
    let base = super::linux_base::opts();
    Target {
        data_layout: "e-p:32:32:32\
                      -i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64\
                      -f32:32:32-f64:64:64\
                      -v64:64:64-v128:64:128\
                      -a:0:64-n32".to_string(),
        llvm_target: "arm-unknown-linux-gnueabi".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        arch: "arm".to_string(),
        target_os: "linux".to_string(),

        options: TargetOptions {
            features: "+v6".to_string(),
            .. base
        },
    }
}
