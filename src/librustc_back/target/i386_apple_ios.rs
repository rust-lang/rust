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

pub fn target() -> Target {
    Target {
        data_layout: "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16\
                      -i32:32:32-i64:32:64\
                      -f32:32:32-f64:32:64-v64:64:64\
                      -v128:128:128-a:0:64-f80:128:128\
                      -n8:16:32".to_string(),
        llvm_target: "i386-apple-ios".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        arch: "x86".to_string(),
        target_os: "ios".to_string(),

        options: super::apple_base::opts()
    }
}
