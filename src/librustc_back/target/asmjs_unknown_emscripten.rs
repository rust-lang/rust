// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{Target, TargetOptions};

pub fn target() -> Target {
    let opts = TargetOptions {
        linker: "emcc".to_string(),
        ar: "emar".to_string(),

        dynamic_linking: false,
        executables: true,
        exe_suffix: ".js".to_string(),
        no_compiler_rt: true,
        linker_is_gnu: true,
        allow_asm: false,
        archive_format: "gnu".to_string(),
        obj_is_bitcode: true,
        .. Default::default()
    };
    Target {
        llvm_target: "asmjs-unknown-emscripten".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_os: "emscripten".to_string(),
        target_env: "".to_string(),
        target_vendor: "unknown".to_string(),
        arch: "asmjs".to_string(),
        options: opts,
    }
}
