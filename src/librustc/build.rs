// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CFG_LIBDIR_RELATIVE");
    println!("cargo:rerun-if-env-changed=CFG_COMPILER_HOST_TRIPLE");
    println!("cargo:rerun-if-env-changed=RUSTC_VERIFY_LLVM_IR");

    if env::var_os("RUSTC_VERIFY_LLVM_IR").is_some() {
        println!("cargo:rustc-cfg=always_verify_llvm_ir");
    }
}
