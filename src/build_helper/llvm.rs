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
use config::Config;

/// Provides paths to the LLVM tools and libraries
pub struct LLVMTools {
    llvm_build_artifacts_dir : PathBuf,
    llvm_src_dir : PathBuf
}

impl LLVMTools {
    pub fn new(cfg : &Config) -> LLVMTools {
        LLVMTools {
            llvm_build_artifacts_dir : cfg.llvm_build_artifacts_dir(),
            llvm_src_dir : cfg.src_dir().join("llvm")
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

    pub fn llc_cmd(&self, target : &str, src : &Path, obj : &Path)
                   -> Command {
        let mut cmd = Command::new(&self.path_to_llc());
        cmd.arg("-filetype=obj")
            .arg(&format!("-mtriple={}", target))
            .arg("-relocation-model=pic")
            .arg("-o").arg(obj).arg(src);
        cmd
    }

    pub fn get_llvm_cxxflags(&self) -> Vec<String> {
        let output = Command::new(&self.path_to_llvm_config())
            .arg("--cxxflags").output().expect("llvm-config --cxxflags");
        let cxxflags = String::from_utf8(output.stdout).unwrap();
        cxxflags.split_whitespace().map(|s| s.into()).collect()
    }

    pub fn llvm_src_dir(&self) -> &Path {
        &self.llvm_src_dir
    }
}
