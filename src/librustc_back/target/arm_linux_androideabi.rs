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
    let mut base = super::linux_base::opts();
    base.features = "+v7".into_string();
    // Many of the symbols defined in compiler-rt are also defined in libgcc.  Android
    // linker doesn't like that by default.
    base.pre_link_args.push("-Wl,--allow-multiple-definition".into_string());
    // FIXME #17437 (and #17448): Android doesn't support position dependant executables anymore.
    base.position_independent_executables = false;

    Target {
        data_layout: "e-p:32:32:32\
                      -i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64\
                      -f32:32:32-f64:64:64\
                      -v64:64:64-v128:64:128\
                      -a0:0:64-n32".into_string(),
        llvm_target: "arm-linux-androideabi".into_string(),
        target_endian: "little".into_string(),
        target_word_size: "32".into_string(),
        arch: "arm".into_string(),
        target_os: "android".into_string(),
        options: base,
    }
}
