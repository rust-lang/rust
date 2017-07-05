// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate build_helper;
extern crate cmake;

use std::env;
use build_helper::sanitizer_lib_boilerplate;

use cmake::Config;

fn main() {
    if let Some(llvm_config) = env::var_os("LLVM_CONFIG") {
        let native = match sanitizer_lib_boilerplate("tsan") {
            Ok(native) => native,
            _ => return,
        };

        Config::new(&native.src_dir)
            .define("COMPILER_RT_BUILD_SANITIZERS", "ON")
            .define("COMPILER_RT_BUILD_BUILTINS", "OFF")
            .define("COMPILER_RT_BUILD_XRAY", "OFF")
            .define("LLVM_CONFIG_PATH", llvm_config)
            .out_dir(&native.out_dir)
            .build_target("tsan")
            .build();
    }
    println!("cargo:rerun-if-env-changed=LLVM_CONFIG");
}
