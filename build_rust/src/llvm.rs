// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Configure and compile LLVM. Also provide paths to the LLVM tools
//! for other parts of the build system.

use std::process::Command;
use std::path::{Path, PathBuf};
use std::ffi::OsString;
use build_state::*;
use configure::ConfigArgs;
use cc::Triple;
use log::Tee;

/// Type of the LLVM build.
#[derive(Clone, Copy)]
pub enum LLVMBuildType {
    Release,                    // implies no assertions
    ReleaseAsserts,
    Debug,                      // implies assertions on
    DebugNoAsserts
}

impl LLVMBuildType {
    fn to_cmake_build_type(self) -> &'static str {
        match self {
            LLVMBuildType::Release | LLVMBuildType::ReleaseAsserts
                => "-DCMAKE_BUILD_TYPE=Release",
            LLVMBuildType::Debug | LLVMBuildType::DebugNoAsserts
                => "-DCMAKE_BUILD_TYPE=Debug"
        }
    }

    fn to_cmake_assert_var(self) -> &'static str {
        match self {
            LLVMBuildType::ReleaseAsserts => "-DLLVM_ENABLE_ASSERTIONS=ON",
            LLVMBuildType::DebugNoAsserts => "-DLLVM_ENABLE_ASSERTIONS=OFF",
            _ => ""             // use the default
        }
    }
}

/// Provides paths to the LLVM tools and other related information
pub struct LLVMTools {
    llvm_build_artifacts_dir : PathBuf
}

impl LLVMTools {
    pub fn new(args : &ConfigArgs, triple : &Triple) -> LLVMTools {
        // msvc build seems to put all build artifacts under Debug
        // regardless of build type
        let build_dir = if triple.is_msvc() {
            llvm_build_dir(args, triple).join("Debug")
        } else {
            llvm_build_dir(args, triple)
        };
        LLVMTools {
            llvm_build_artifacts_dir : build_dir
        }
    }

    pub fn path_to_llvm_config(&self) -> PathBuf {
        self.llvm_build_artifacts_dir.join("bin").join("llvm-config")
    }

    fn path_to_llc(&self) -> PathBuf {
        self.llvm_build_artifacts_dir.join("bin").join("llc")
    }

    pub fn path_to_llvm_libs(&self) -> PathBuf {
        self.llvm_build_artifacts_dir.join("lib")
    }

    pub fn llc_cmd(&self, target : &Triple, src : &Path, obj : &Path)
                   -> Command {
        let mut cmd = Command::new(&self.path_to_llc());
        cmd.arg("-filetype=obj")
            .arg(&format!("-mtriple={}", target))
            .arg("-relocation-model=pic")
            .arg("-o").arg(obj).arg(src);
        cmd
    }

    pub fn get_llvm_cxxflags(&self) -> BuildState<Vec<String>> {
        let output = try!(Command::new(&self.path_to_llvm_config())
                          .arg("--cxxflags").output());
        let cxxflags = try!(String::from_utf8(output.stdout));
        continue_with(cxxflags.trim().split(' ')
                      .filter(|s| *s != "")
                      .map(|s| s.into()).collect())
    }
}

fn llvm_src_dir(args : &ConfigArgs) -> PathBuf {
    args.src_dir().join("llvm")
}

fn llvm_build_dir(args : &ConfigArgs, triple : &Triple) -> PathBuf {
    args.target_build_dir(triple).join("llvm")
}

fn cmake_makefile_target(triple : &Triple) -> &'static str {
    if triple.is_msvc() {
        "Visual Studio 12"
    } else if triple.is_mingw() {
        "MinGW Makefiles"
    } else {
        "Unix Makefiles"
    }
}

// TODO : Add cross-compile
fn llvm_config_args(cfg : &ConfigArgs, target : &Triple) -> Vec<OsString> {
    vec![
        "-G".into(),
        cmake_makefile_target(target).into(),
        cfg.llvm_build_type().to_cmake_build_type().into(),
        cfg.llvm_build_type().to_cmake_assert_var().into(),
        "-DLLVM_ENABLE_TERMINFO=OFF".into(),
        "-DLLVM_ENABLE_ZLIB=OFF".into(),
        "-DLLVM_ENABLE_FFI=OFF".into(),
        "-DLLVM_BUILD_DOCS=OFF".into(),
        llvm_src_dir(cfg).into_os_string() ]
}

fn configure_llvm_for_target(args : &ConfigArgs,
                             target : &Triple) -> BuildState<()> {
    println!("Configuring llvm for target triple {}...", target);
    let build_dir = llvm_build_dir(args, target);
    let logger = args.get_logger(target, "configure_llvm");
    Command::new("cmake")
        .args(&llvm_config_args(args, target))
        .current_dir(&build_dir)
        .tee(&logger)
}

fn build_llvm_for_target(cfg : &ConfigArgs,
                         target : &Triple) -> BuildState<()> {
    println!("Building llvm for target triple {}...", target);
    let logger = cfg.get_logger(target, "make_llvm");
    let mut arg : Vec<OsString> = vec![ "--build".into(), ".".into() ];
    // msbuild doesn't support -jnproc and will use all cores by default
    if !target.is_msvc() {
        arg.push("--".into());
        arg.push(cfg.jnproc());
    }
    Command::new("cmake")
        .args(&arg)
        .current_dir(&llvm_build_dir(cfg, target))
        .tee(&logger)
}

pub fn configure_llvm(cfg : &ConfigArgs) -> BuildState<()> {
    let build = cfg.build_triple();
    let host = cfg.host_triple();
    if build == host {
        configure_llvm_for_target(cfg, build)
    } else {
        try!(configure_llvm_for_target(cfg, build));
        configure_llvm_for_target(cfg, host)
    }
}

pub fn build_llvm(cfg : &ConfigArgs) -> BuildState<()> {
    let build = cfg.build_triple();
    let host = cfg.host_triple();
    if build == host {
        build_llvm_for_target(cfg, build)
    } else {
        try!(build_llvm_for_target(cfg, build));
        build_llvm_for_target(cfg, host)
    }
}
