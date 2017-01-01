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
use std::path::{PathBuf, Path};

use build_helper::output;

fn detect_llvm_link(llvm_config: &Path) -> (&'static str, Option<&'static str>) {
    let mut version_cmd = Command::new(llvm_config);
    version_cmd.arg("--version");
    let version_output = output(&mut version_cmd);
    let mut parts = version_output.split('.').take(2)
        .filter_map(|s| s.parse::<u32>().ok());
    if let (Some(major), Some(minor)) = (parts.next(), parts.next()) {
        if major > 3 || (major == 3 && minor >= 9) {
            // Force the link mode we want, preferring static by default, but
            // possibly overridden by `configure --enable-llvm-link-shared`.
            if env::var_os("LLVM_LINK_SHARED").is_some() {
                return ("dylib", Some("--link-shared"));
            } else {
                return ("static", Some("--link-static"));
            }
        } else if major == 3 && minor == 8 {
            // Find out LLVM's default linking mode.
            let mut mode_cmd = Command::new(llvm_config);
            mode_cmd.arg("--shared-mode");
            if output(&mut mode_cmd).trim() == "shared" {
                return ("dylib", None);
            } else {
                return ("static", None);
            }
        }
    }
    ("static", None)
}

fn main() {
    println!("cargo:rustc-cfg=cargobuild");

    let target = env::var("TARGET").expect("TARGET was not set");
    let llvm_config = env::var_os("LLVM_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if let Some(dir) = env::var_os("CARGO_TARGET_DIR").map(PathBuf::from) {
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

    // Test whether we're cross-compiling LLVM. This is a pretty rare case
    // currently where we're producing an LLVM for a different platform than
    // what this build script is currently running on.
    //
    // In that case, there's no guarantee that we can actually run the target,
    // so the build system works around this by giving us the LLVM_CONFIG for
    // the host platform. This only really works if the host LLVM and target
    // LLVM are compiled the same way, but for us that's typically the case.
    //
    // We *want* detect this cross compiling situation by asking llvm-config
    // what it's host-target is. If that's not the TARGET, then we're cross
    // compiling. Unfortunately `llvm-config` seems either be buggy, or we're
    // misconfiguring it, because the `i686-pc-windows-gnu` build of LLVM will
    // report itself with a `--host-target` of `x86_64-pc-windows-gnu`. This
    // tricks us into thinking we're doing a cross build when we aren't, so
    // havoc ensues.
    //
    // In any case, if we're cross compiling, this generally just means that we
    // can't trust all the output of llvm-config becaues it might be targeted
    // for the host rather than the target. As a result a bunch of blocks below
    // are gated on `if !is_crossed`
    let target = env::var("TARGET").expect("TARGET was not set");
    let host = env::var("HOST").expect("HOST was not set");
    let is_crossed = target != host;

    let optional_components =
        ["x86", "arm", "aarch64", "mips", "powerpc", "pnacl", "systemz", "jsbackend", "msp430",
         "sparc", "nvptx"];

    // FIXME: surely we don't need all these components, right? Stuff like mcjit
    //        or interpreter the compiler itself never uses.
    let required_components = &["ipo",
                                "bitreader",
                                "bitwriter",
                                "linker",
                                "asmparser",
                                "mcjit",
                                "interpreter",
                                "instrumentation"];

    let components = output(Command::new(&llvm_config).arg("--components"));
    let mut components = components.split_whitespace().collect::<Vec<_>>();
    components.retain(|c| optional_components.contains(c) || required_components.contains(c));

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
        // Ignore flags like `-m64` when we're doing a cross build
        if is_crossed && flag.starts_with("-m") {
            continue;
        }
        cfg.flag(flag);
    }

    for component in &components[..] {
        let mut flag = String::from("-DLLVM_COMPONENT_");
        flag.push_str(&component.to_uppercase());
        cfg.flag(&flag);
    }

    if env::var_os("LLVM_RUSTLLVM").is_some() {
        cfg.flag("-DLLVM_RUSTLLVM");
    }

    println!("cargo:rerun-if-changed=../rustllvm/PassWrapper.cpp");
    println!("cargo:rerun-if-changed=../rustllvm/RustWrapper.cpp");
    println!("cargo:rerun-if-changed=../rustllvm/ArchiveWrapper.cpp");
    cfg.file("../rustllvm/PassWrapper.cpp")
       .file("../rustllvm/RustWrapper.cpp")
       .file("../rustllvm/ArchiveWrapper.cpp")
       .cpp(true)
       .cpp_link_stdlib(None) // we handle this below
       .compile("librustllvm.a");

    let (llvm_kind, llvm_link_arg) = detect_llvm_link(&llvm_config);

    // Link in all LLVM libraries, if we're uwring the "wrong" llvm-config then
    // we don't pick up system libs because unfortunately they're for the host
    // of llvm-config, not the target that we're attempting to link.
    let mut cmd = Command::new(&llvm_config);
    cmd.arg("--libs");

    if let Some(link_arg) = llvm_link_arg {
        cmd.arg(link_arg);
    }

    if !is_crossed {
        cmd.arg("--system-libs");
    }
    cmd.args(&components[..]);

    for lib in output(&mut cmd).split_whitespace() {
        let name = if lib.starts_with("-l") {
            &lib[2..]
        } else if lib.starts_with("-") {
            &lib[1..]
        } else if Path::new(lib).exists() {
            // On MSVC llvm-config will print the full name to libraries, but
            // we're only interested in the name part
            let name = Path::new(lib).file_name().unwrap().to_str().unwrap();
            name.trim_right_matches(".lib")
        } else if lib.ends_with(".lib") {
            // Some MSVC libraries just come up with `.lib` tacked on, so chop
            // that off
            lib.trim_right_matches(".lib")
        } else {
            continue;
        };

        // Don't need or want this library, but LLVM's CMake build system
        // doesn't provide a way to disable it, so filter it here even though we
        // may or may not have built it. We don't reference anything from this
        // library and it otherwise may just pull in extra dependencies on
        // libedit which we don't want
        if name == "LLVMLineEditor" {
            continue;
        }

        let kind = if name.starts_with("LLVM") {
            llvm_kind
        } else {
            "dylib"
        };
        println!("cargo:rustc-link-lib={}={}", kind, name);
    }

    // LLVM ldflags
    //
    // If we're a cross-compile of LLVM then unfortunately we can't trust these
    // ldflags (largely where all the LLVM libs are located). Currently just
    // hack around this by replacing the host triple with the target and pray
    // that those -L directories are the same!
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

    // OpenBSD has a particular C++ runtime library name
    let stdcppname = if target.contains("openbsd") {
        "estdc++"
    } else {
        "stdc++"
    };

    // C++ runtime library
    if !target.contains("msvc") {
        if let Some(s) = env::var_os("LLVM_STATIC_STDCPP") {
            assert!(!cxxflags.contains("stdlib=libc++"));
            let path = PathBuf::from(s);
            println!("cargo:rustc-link-search=native={}",
                     path.parent().unwrap().display());
            println!("cargo:rustc-link-lib=static={}", stdcppname);
        } else if cxxflags.contains("stdlib=libc++") {
            println!("cargo:rustc-link-lib=c++");
        } else {
            println!("cargo:rustc-link-lib={}", stdcppname);
        }
    }
}
