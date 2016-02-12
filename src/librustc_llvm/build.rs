// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate gcc;
extern crate build_helper;

use std::process::Command;
use std::env;
use std::path::PathBuf;

use build_helper::output;

fn main() {
    println!("cargo:rustc-cfg=cargobuild");

    let target = env::var("TARGET").unwrap();
    let llvm_config = env::var_os("LLVM_CONFIG").map(PathBuf::from)
                           .unwrap_or_else(|| {
        match env::var_os("CARGO_TARGET_DIR").map(PathBuf::from) {
            Some(dir) => {
                let to_test = dir.parent().unwrap().parent().unwrap()
                                 .join(&target).join("llvm/bin/llvm-config");
                if Command::new(&to_test).output().is_ok() {
                    return to_test
                }
            }
            None => {}
        }
        PathBuf::from("llvm-config")
    });

    println!("cargo:rerun-if-changed={}", llvm_config.display());

    let optional_components = ["x86", "arm", "aarch64", "mips", "powerpc",
                               "pnacl"];

    // FIXME: surely we don't need all these components, right? Stuff like mcjit
    //        or interpreter the compiler itself never uses.
    let required_components = &["ipo", "bitreader", "bitwriter", "linker",
                                "asmparser", "mcjit", "interpreter",
                                "instrumentation"];

    let components = output(Command::new(&llvm_config).arg("--components"));
    let mut components = components.split_whitespace().collect::<Vec<_>>();
    components.retain(|c| {
        optional_components.contains(c) || required_components.contains(c)
    });

    for component in required_components {
        if !components.contains(component) {
            panic!("require llvm component {} but wasn't found", component);
        }
    }

    for component in components.iter() {
        println!("cargo:rustc-cfg=llvm_component=\"{}\"", component);
    }

    // Link in our own LLVM shims, compiled with the same flags as LLVM
    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--cxxflags");
    let cxxflags = output(&mut cmd);
    let mut cfg = gcc::Config::new();
    for flag in cxxflags.split_whitespace() {
        cfg.flag(flag);
    }
    cfg.file("../rustllvm/ExecutionEngineWrapper.cpp")
       .file("../rustllvm/PassWrapper.cpp")
       .file("../rustllvm/RustWrapper.cpp")
       .file("../rustllvm/ArchiveWrapper.cpp")
       .cpp(true)
       .cpp_link_stdlib(None) // we handle this below
       .compile("librustllvm.a");

    // Link in all LLVM libraries
    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--libs").arg("--system-libs").args(&components[..]);
    for lib in output(&mut cmd).split_whitespace() {
        let name = if lib.starts_with("-l") {
            &lib[2..]
        } else if lib.starts_with("-") {
            &lib[1..]
        } else {
            continue
        };

        // Don't need or want this library, but LLVM's CMake build system
        // doesn't provide a way to disable it, so filter it here even though we
        // may or may not have built it. We don't reference anything from this
        // library and it otherwise may just pull in extra dependencies on
        // libedit which we don't want
        if name == "LLVMLineEditor" {
            continue
        }

        let kind = if name.starts_with("LLVM") {"static"} else {"dylib"};
        println!("cargo:rustc-link-lib={}={}", kind, name);
    }

    // LLVM ldflags
    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--ldflags");
    for lib in output(&mut cmd).split_whitespace() {
        if lib.starts_with("-l") {
            println!("cargo:rustc-link-lib={}", &lib[2..]);
        } else if lib.starts_with("-L") {
            println!("cargo:rustc-link-search=native={}", &lib[2..]);
        }
    }

    // C++ runtime library
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
}
