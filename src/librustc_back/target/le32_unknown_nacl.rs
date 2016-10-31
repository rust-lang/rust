// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    let opts = TargetOptions {
        linker: "pnacl-clang".to_string(),
        ar: "pnacl-ar".to_string(),

        pre_link_args: vec!["--pnacl-exceptions=sjlj".to_string(),
                            "--target=le32-unknown-nacl".to_string(),
                            "-Wl,--start-group".to_string()],
        post_link_args: vec!["-Wl,--end-group".to_string()],
        dynamic_linking: false,
        executables: true,
        exe_suffix: ".pexe".to_string(),
        linker_is_gnu: true,
        allow_asm: false,
        max_atomic_width: Some(32),
        .. Default::default()
    };
    Ok(Target {
        llvm_target: "le32-unknown-nacl".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_os: "nacl".to_string(),
        target_env: "newlib".to_string(),
        target_vendor: "unknown".to_string(),
        data_layout: "e-i64:64:64-p:32:32:32-v128:32:32".to_string(),
        arch: "le32".to_string(),
        options: opts,
    })
}
