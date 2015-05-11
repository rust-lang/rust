// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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
    let mut base = super::windows_msvc_base::opts();
    base.cpu = "x86-64".to_string();

    Target {
        // This is currently in sync with the specification for
        // x86_64-pc-windows-gnu but there's a comment in that file questioning
        // whether this is valid or not. Sounds like the two should stay in sync
        // at least for now.
        data_layout: "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                      f32:32:32-f64:64:64-v64:64:64-v128:128:128-a:0:64-\
                      s0:64:64-f80:128:128-n8:16:32:64-S128".to_string(),
        llvm_target: "x86_64-pc-windows-msvc".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "64".to_string(),
        arch: "x86_64".to_string(),
        target_os: "windows".to_string(),
        target_env: "msvc".to_string(),
        options: base,
    }
}
