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
extern crate gcc;

use std::env;
use std::path::PathBuf;
use std::process::Command;

use build_helper::output;

// NOTE This pretty much looks the same as librustc_llvm build.rs. Refer to that file for an
// explanation of what this function does.

fn main() {
    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    let is_crossed = target != host;

    let llvm_config = env::var_os("LLVM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if let Some(dir) = env::var_os("CARGO_TARGET_DIR")
                .map(PathBuf::from) {
                    let to_test = dir.parent()
                        .unwrap()
                        .parent()
                        .unwrap()
                        .join(&target)
                        .join("llvm/bin/llvm-config");
                    if Command::new(&to_test).output().is_ok() {
                        return to_test;
                    }
                }
            PathBuf::from("llvm-config")
        });

    println!("cargo:rerun-if-changed={}", llvm_config.display());

    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--cxxflags");
    let cxxflags = output(&mut cmd);

    let mut cfg = gcc::Config::new();
    for flag in cxxflags.split_whitespace() {
        // Ignore flags like `-m64` when we're doing a cross build
        if is_crossed && flag.starts_with("-m") {
            continue;
        }
        cfg.flag(flag);
    }

    cfg.file("../rustlld/RustWrapper.cpp")
        .cpp(true)
        .cpp_link_stdlib(None)
        .compile("librustlld.a");

    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--ldflags");
    for lib in output(&mut cmd).split_whitespace() {
        if lib.starts_with("-LIBPATH:") {
            println!("cargo:rustc-link-search=native={}", &lib[9..]);
        } else if is_crossed {
            if lib.starts_with("-L") {
                println!("cargo:rustc-link-search=native={}",
                         lib[2..].replace(&host, &target));
            }
        } else if lib.starts_with("-l") {
            println!("cargo:rustc-link-lib={}", &lib[2..]);
        } else if lib.starts_with("-L") {
            println!("cargo:rustc-link-search=native={}", &lib[2..]);
        }
    }

    if !target.contains("msvc") {
        if let Some(s) = env::var_os("LLVM_STATIC_STDCPP") {
            assert!(!cxxflags.contains("stdlib=libc++"));
            let path = PathBuf::from(s);
            println!("cargo:rustc-link-search=native={}",
                     path.parent().unwrap().display());
            println!("cargo:rustc-link-lib=static=stdc++");
        } else if cxxflags.contains("stdlib=libc++") {
            println!("cargo:rustc-link-lib=c++");
        } else {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }

    // NOTE After this point comes LLD specific stuff
    //
    // We need to link to these libraries to be able to use lld functions via ffi
    // These libraries depend on llvm libraries but we don't link this crate to those libraries
    // because that can result in duplicate linking to static libraries, instead rustc_llvm will
    // link to those libraries for us.
    //
    // To elaborate on this last point: which libraries we must link to come from the output of
    // llvm-config. llvm-config maps "components" to -l flags. rustc_llvm is already using this
    // method to figure out which -l flags we need to use llvm via ffi. If we do the same thing here
    // we risk ending up static llvm libraries being linked up more than once in the final rustc
    // binary -- and this last part can cause runtime failures of rustc like
    //
    //     : CommandLine Error: Option 'color' registered more than once!
    //     LLVM ERROR: inconsistency in registered CommandLine options
    //
    println!("cargo:rustc-link-lib=static=lldConfig");
    println!("cargo:rustc-link-lib=static=lldELF");
}
