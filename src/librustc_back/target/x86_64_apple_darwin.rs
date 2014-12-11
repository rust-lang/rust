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
    let mut base = super::apple_base::opts();
    base.eliminate_frame_pointer = false;
    base.pre_link_args.push("-m64".into_string());

    Target {
        data_layout: "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
                      f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-\
                      s0:64:64-f80:128:128-n8:16:32:64".into_string(),
        llvm_target: "x86_64-apple-darwin".into_string(),
        target_endian: "little".into_string(),
        target_word_size: "64".into_string(),
        arch: "x86_64".into_string(),
        target_os: "macos".into_string(),
        options: base,
    }
}
