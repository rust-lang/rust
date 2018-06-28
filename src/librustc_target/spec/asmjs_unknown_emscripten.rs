// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{LinkArgs, LinkerFlavor, Target, TargetOptions};

pub fn target() -> Result<Target, String> {
    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Em,
                vec!["-s".to_string(),
                     "ERROR_ON_UNDEFINED_SYMBOLS=1".to_string(),
                     "-s".to_string(),
                     "ABORTING_MALLOC=0".to_string(),
                     "-s".to_string(),
                     "WASM=0".to_string()]);

    let opts = TargetOptions {
        dynamic_linking: false,
        executables: true,
        exe_suffix: ".js".to_string(),
        linker_is_gnu: true,
        allow_asm: false,
        obj_is_bitcode: true,
        is_like_emscripten: true,
        max_atomic_width: Some(32),
        post_link_args: args,
        target_family: Some("unix".to_string()),
        codegen_backend: "emscripten".to_string(),
        .. Default::default()
    };
    Ok(Target {
        llvm_target: "asmjs-unknown-emscripten".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "emscripten".to_string(),
        target_env: "".to_string(),
        target_vendor: "unknown".to_string(),
        data_layout: "e-p:32:32-i64:64-v128:32:128-n32-S128".to_string(),
        arch: "asmjs".to_string(),
        linker_flavor: LinkerFlavor::Em,
        options: opts,
    })
}
