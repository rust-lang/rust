// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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
        linker: "pnacl-clang".to_string(),
        ar: "pnacl-ar".to_string(),

        pre_link_args: vec!("--pnacl-exceptions=sjlj".to_string(),
                            "-Wl,--start-group".to_string()),
        post_link_args: vec!("-Wl,--end-group".to_string()),
        dynamic_linking: false,
        executables: true,
        morestack: false,
        exe_suffix: ".pexe".to_string(),
        no_compiler_rt: true,
        linker_is_gnu: true,
        is_like_pnacl: true,
        no_asm: true,
        lto_supported: false, // `pnacl-ld` runs "LTO".
        .. Default::default()
    };
    Target {
        data_layout: "e-i1:8:8-i8:8:8-i16:16:16-i32:32:32-\
                      i64:64:64-f32:32:32-f64:64:64-p:32:32:32-v128:32:32".to_string(),
        // Pretend that we are ARM for name mangling and assembly conventions.
        // https://code.google.com/p/nativeclient/issues/detail?id=2554
        llvm_target: "armv7a-none-nacl-gnueabi".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_os: "nacl".to_string(),
        target_env: "".to_string(),
        arch: "le32".to_string(),
        options: opts,
    }
}
