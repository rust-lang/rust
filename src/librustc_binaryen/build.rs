// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate cc;
extern crate cmake;

use std::env;

use cmake::Config;

fn main() {
    let target = env::var("TARGET").unwrap();

    // Bring in `__emutls_get_address` which is apparently needed for now
    if target.contains("pc-windows-gnu") {
        println!("cargo:rustc-link-lib=gcc_eh");
        println!("cargo:rustc-link-lib=pthread");
    }

    Config::new("../binaryen")
        .define("BUILD_STATIC_LIB", "ON")
        .build_target("binaryen")
        .build();

    // I couldn't figure out how to link just one of these, so link everything.
    println!("cargo:rustc-link-lib=static=asmjs");
    println!("cargo:rustc-link-lib=static=binaryen");
    println!("cargo:rustc-link-lib=static=cfg");
    println!("cargo:rustc-link-lib=static=emscripten-optimizer");
    println!("cargo:rustc-link-lib=static=ir");
    println!("cargo:rustc-link-lib=static=passes");
    println!("cargo:rustc-link-lib=static=support");
    println!("cargo:rustc-link-lib=static=wasm");

    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}/build/lib", out_dir);

    // Add in our own little shim along with some extra files that weren't
    // included in the main build.
    let mut cfg = cc::Build::new();
    cfg.file("BinaryenWrapper.cpp")
        .file("../binaryen/src/wasm-linker.cpp")
        .file("../binaryen/src/wasm-emscripten.cpp")
        .include("../binaryen/src")
        .cpp_link_stdlib(None)
        .warnings(false)
        .cpp(true);

    if !target.contains("msvc") {
        cfg.flag("-std=c++11");
    }
    cfg.compile("binaryen_wrapper");
}
