// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Sanity checking performed by rustbuild before actually executing anything.
//!
//! This module contains the implementation of ensuring that the build
//! environment looks reasonable before progressing. This will verify that
//! various programs like git and python exist, along with ensuring that all C
//! compilers for cross-compiling are found.
//!
//! In theory if we get past this phase it's a bug if a build fails, but in
//! practice that's likely not true!

use std::collections::HashSet;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::process::Command;

use build_helper::output;

use Build;

pub fn check(build: &mut Build) {
    let mut checked = HashSet::new();
    let path = env::var_os("PATH").unwrap_or(OsString::new());
    // On Windows, quotes are invalid characters for filename paths, and if
    // one is present as part of the PATH then that can lead to the system
    // being unable to identify the files properly. See
    // https://github.com/rust-lang/rust/issues/34959 for more details.
    if cfg!(windows) {
        if path.to_string_lossy().contains("\"") {
            panic!("PATH contains invalid character '\"'");
        }
    }
    let mut need_cmd = |cmd: &OsStr| {
        if !checked.insert(cmd.to_owned()) {
            return
        }
        for path in env::split_paths(&path).map(|p| p.join(cmd)) {
            if fs::metadata(&path).is_ok() ||
               fs::metadata(path.with_extension("exe")).is_ok() {
                return
            }
        }
        panic!("\n\ncouldn't find required command: {:?}\n\n", cmd);
    };

    // If we've got a git directory we're gona need git to update
    // submodules and learn about various other aspects.
    if fs::metadata(build.src.join(".git")).is_ok() {
        need_cmd("git".as_ref());
    }

    // We need cmake, but only if we're actually building LLVM
    for host in build.config.host.iter() {
        if let Some(config) = build.config.target_config.get(host) {
            if config.llvm_config.is_some() {
                continue
            }
        }
        need_cmd("cmake".as_ref());
        if build.config.ninja {
            need_cmd("ninja".as_ref())
        }
        break
    }

    need_cmd("python".as_ref());

    // We're gonna build some custom C code here and there, host triples
    // also build some C++ shims for LLVM so we need a C++ compiler.
    for target in build.config.target.iter() {
        need_cmd(build.cc(target).as_ref());
        if let Some(ar) = build.ar(target) {
            need_cmd(ar.as_ref());
        }
    }
    for host in build.config.host.iter() {
        need_cmd(build.cxx(host).as_ref());
    }

    // Externally configured LLVM requires FileCheck to exist
    let filecheck = build.llvm_filecheck(&build.config.build);
    if !filecheck.starts_with(&build.out) && !filecheck.exists() && build.config.codegen_tests {
        panic!("filecheck executable {:?} does not exist", filecheck);
    }

    for target in build.config.target.iter() {
        // Either can't build or don't want to run jemalloc on these targets
        if target.contains("rumprun") ||
           target.contains("bitrig") ||
           target.contains("openbsd") ||
           target.contains("msvc") ||
           target.contains("emscripten") {
            build.config.use_jemalloc = false;
        }

        // Can't compile for iOS unless we're on OSX
        if target.contains("apple-ios") &&
           !build.config.build.contains("apple-darwin") {
            panic!("the iOS target is only supported on OSX");
        }

        // Make sure musl-root is valid if specified
        if target.contains("musl") && !target.contains("mips") {
            match build.config.musl_root {
                Some(ref root) => {
                    if fs::metadata(root.join("lib/libc.a")).is_err() {
                        panic!("couldn't find libc.a in musl dir: {}",
                               root.join("lib").display());
                    }
                    if fs::metadata(root.join("lib/libunwind.a")).is_err() {
                        panic!("couldn't find libunwind.a in musl dir: {}",
                               root.join("lib").display());
                    }
                }
                None => {
                    panic!("when targeting MUSL the build.musl-root option \
                            must be specified in config.toml")
                }
            }
        }

        if target.contains("msvc") {
            // There are three builds of cmake on windows: MSVC, MinGW, and
            // Cygwin. The Cygwin build does not have generators for Visual
            // Studio, so detect that here and error.
            let out = output(Command::new("cmake").arg("--help"));
            if !out.contains("Visual Studio") {
                panic!("
cmake does not support Visual Studio generators.

This is likely due to it being an msys/cygwin build of cmake,
rather than the required windows version, built using MinGW
or Visual Studio.

If you are building under msys2 try installing the mingw-w64-x86_64-cmake
package instead of cmake:

$ pacman -R cmake && pacman -S mingw-w64-x86_64-cmake
");
            }
        }

        if target.contains("arm-linux-android") {
            need_cmd("adb".as_ref());
        }
    }

    for host in build.flags.host.iter() {
        if !build.config.host.contains(host) {
            panic!("specified host `{}` is not in the ./configure list", host);
        }
    }
    for target in build.flags.target.iter() {
        if !build.config.target.contains(target) {
            panic!("specified target `{}` is not in the ./configure list",
                   target);
        }
    }

    let run = |cmd: &mut Command| {
        cmd.output().map(|output| {
            String::from_utf8_lossy(&output.stdout)
                   .lines().next().unwrap()
                   .to_string()
        })
    };
    build.gdb_version = run(Command::new("gdb").arg("--version")).ok();
    build.lldb_version = run(Command::new("lldb").arg("--version")).ok();
    if build.lldb_version.is_some() {
        build.lldb_python_dir = run(Command::new("lldb").arg("-P")).ok();
    }
}
