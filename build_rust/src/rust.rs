// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Bootstrap a working stage2 compiler from the stage0 snapshot.
//! This means that for a given `i` where `i` runs from 0 to 2, we
//! use the existing stage`i` compiler to compile the stage`i+1`
//! compiler. The build artifacts are then promoted into the
//! stage`i+1` directory which will then be used to compile the next
//! stage.

use std::process::Command;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use build_state::*;
use configure::ConfigArgs;
use rt::{rt_build_dir, llvmdeps_path, llvmdef_path};
use cc::Triple;
use log::Tee;

const RUST_LIBS : &'static [&'static str]
    = &["core", "libc", "rand", "alloc", "rustc_unicode", "collections",
        "rustc_bitflags", "std", "arena", "flate", "getopts", "graphviz",
        "log", "term", "serialize", "fmt_macros", "syntax", "rbml",
        "rustc_llvm", "rustc_back", "rustc_data_structures", "rustc",
        "rustc_borrowck", "rustc_typeck", "rustc_resolve", "rustc_trans",
        "rustc_privacy", "rustc_lint", "rustc_driver"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Stage0, Stage1, Stage2
}

impl Stage {
    fn to_str(self) -> &'static str {
        match self {
            Stage::Stage0 => "stage0",
            Stage::Stage1 => "stage1",
            Stage::Stage2 => "stage2"
        }
    }
}

/// Specifies the host and target triples and build directories
/// of a given stage
struct StageInfo {
    stage : Stage,
    compiler_host : Triple,
    target_triple : Triple,
    host_build_dir : PathBuf
}

impl StageInfo {
    fn stage0(cfg : &ConfigArgs) -> StageInfo {
        let build = cfg.build_triple();
        StageInfo {
            stage : Stage::Stage0,
            compiler_host : build.clone(),
            target_triple : build.clone(),
            host_build_dir : cfg.target_build_dir(build)
        }
    }

    fn stage1(cfg : &ConfigArgs) -> StageInfo {
        let build = cfg.build_triple();
        let host = cfg.host_triple();
        StageInfo {
            stage : Stage::Stage1,
            compiler_host : build.clone(),
            target_triple : host.clone(),
            host_build_dir : cfg.target_build_dir(build)
        }
    }

    fn stage2(cfg : &ConfigArgs, target : &Triple) -> StageInfo {
        let host = cfg.host_triple();
        StageInfo {
            stage : Stage::Stage2,
            compiler_host : host.clone(),
            target_triple : target.clone(),
            host_build_dir : cfg.target_build_dir(host)
        }
    }

    fn is_stage0(&self) -> bool {
        self.stage == Stage::Stage0
    }

    fn is_stage1(&self) -> bool {
        self.stage == Stage::Stage1
    }

    fn is_stage2(&self) -> bool {
        self.stage == Stage::Stage2
    }

    fn build_dir(&self) -> PathBuf {
        self.host_build_dir.join(self.stage.to_str())
    }

    fn bin_dir(&self) -> PathBuf {
        self.build_dir().join("bin")
    }

    fn lib_dir(&self) -> PathBuf {
        if self.compiler_host.is_windows() {
            self.build_dir().join("bin")
        } else {
            self.build_dir().join("lib")
        }
    }

    fn rustlib_dir(&self) -> PathBuf {
        self.lib_dir().join("rustlib").join(&self.target_triple)
    }

    fn rustlib_bin(&self) -> PathBuf {
        self.rustlib_dir().join("bin")
    }

    fn rustlib_lib(&self) -> PathBuf {
        self.rustlib_dir().join("lib")
    }
}

/// Invoke the rustc compiler to compile a library or an executable
struct RustBuilder {
    rustc_path : PathBuf,
    src_dir : PathBuf,
    llvm_deps_file : PathBuf,
    llvm_def_file : PathBuf,
    ld_library_path : PathBuf,
    cfgs : Vec<&'static str>,
    prefer_dynamic : bool,
    no_landing_pads : bool,
    target_triple : Triple,
    link_dirs : Vec<PathBuf>,
    git_hash : String,
    extra_args : Vec<&'static str>
}

impl RustBuilder {
    fn new(cfg : &ConfigArgs, sinfo : &StageInfo) -> RustBuilder {
        let mut cfgs : Vec<&'static str> = vec![];
        if !cfg.is_debug_build() {
            cfgs.push("rtopt");
            cfgs.push("ndebug");
        }
        if !sinfo.target_triple.is_windows() {
            cfgs.push("jemalloc");
        }
        cfgs.push(sinfo.stage.to_str());

        let rustc_path =
            if cfg.use_local_rustc() && sinfo.is_stage0() {
                PathBuf::from("rustc")
            } else {
                sinfo.bin_dir().join("rustc")
            };

        let link_dirs = vec![
            rt_build_dir(cfg, &sinfo.compiler_host),
            cfg.llvm_tools(&sinfo.compiler_host).path_to_llvm_libs()];

        RustBuilder {
            rustc_path : rustc_path,
            src_dir : cfg.src_dir(),
            llvm_deps_file : llvmdeps_path(cfg, &sinfo.compiler_host),
            llvm_def_file : llvmdef_path(cfg, &sinfo.compiler_host),
            ld_library_path : sinfo.lib_dir(),
            cfgs : cfgs,
            prefer_dynamic : true,
            no_landing_pads : true,
            target_triple : sinfo.target_triple.clone(),
            link_dirs : link_dirs,
            git_hash : cfg.get_git_hash(),
            extra_args : vec![ "-O", "-W", "warnings" ]
        }
    }

    fn compile_cmd(&self) -> Command {
        let mut cmd = Command::new(&self.rustc_path);
        cmd.env("CFG_COMPILER_HOST_TRIPLE", &self.target_triple);
        cmd.env("CFG_LLVM_LINKAGE_FILE", &self.llvm_deps_file);
        cmd.env("LD_LIBRARY_PATH", &self.ld_library_path);
        cmd.args(&self.extra_args);
        for s in &self.cfgs {
            cmd.arg("--cfg").arg(s);
        }
        if self.prefer_dynamic {
            cmd.arg("-C").arg("prefer-dynamic");
        }
        if self.no_landing_pads {
            cmd.arg("-Z").arg("no-landing-pads");
        }
        cmd.arg("--target").arg(&self.target_triple);
        for d in &self.link_dirs {
            cmd.arg("-L").arg(d);
        }
        cmd.arg("-C").arg(&format!("extra-filename=-{}", self.git_hash));
        cmd
    }

    fn rustc_lib_cmd(&self, lib : &str, out_dir : &Path) -> Command {
        let src_path = self.src_dir
            .join(&format!("lib{}", lib)).join("lib.rs");
        let mut cmd = self.compile_cmd();
        if lib == "rustc_llvm" && self.target_triple.is_msvc() {
            cmd.arg("-C");
            let mut s = OsString::new();
            s.push("link-args=-DEF:");
            s.push(&self.llvm_def_file);
            cmd.arg(&s);
        }
        cmd.arg("--out-dir").arg(out_dir);
        cmd.arg(&src_path);
        cmd
    }

    fn compile_exe_cmd(&self, out : &Path) -> Command {
        let mut cmd = self.compile_cmd();
        cmd.arg("-o").arg(out);
        let src = self.src_dir.join("driver").join("driver.rs");
        cmd.arg(&src);
        cmd
    }

    fn rustc_exe_cmd(&self, out : &Path) -> Command {
        let mut cmd = self.compile_exe_cmd(out);
        cmd.arg("--cfg").arg("rustc");
        cmd
    }

    fn rustdoc_exe_cmd(&self, out : &Path) -> Command {
        let mut cmd = self.compile_exe_cmd(out);
        cmd.arg("--cfg").arg("rustdoc");
        cmd
    }
}

// Copy the runtime libraries from the <triple>/rt directory into
// stagei/lib/rustlib/<triple>/lib
fn copy_rt_libraries(cfg : &ConfigArgs, sinfo : &StageInfo)
                     -> BuildState<()> {
    use std::fs::copy;
    let libs = if sinfo.target_triple.is_windows() {
        vec!["compiler-rt"]
    } else {
        vec!["compiler-rt", "morestack"]
    };
    let from_dir = rt_build_dir(cfg, &sinfo.target_triple);
    let to_dir = sinfo.rustlib_lib();
    for lib in &libs {
        let filename = sinfo.target_triple.with_lib_ext(lib);
        let from = from_dir.join(&filename);
        let to = to_dir.join(&filename);
        try!(copy(&from, &to).map_err(|e| {
            format!("Failed to copy {:?} to {:?}: {}", from, to, e)
        }));
    }
    continue_build()
}

fn build_stage(cfg : &ConfigArgs, sinfo : &StageInfo) -> BuildState<()> {
    try!(copy_rt_libraries(cfg, sinfo));
    let compiler = RustBuilder::new(cfg, sinfo);
    let out_dir = sinfo.rustlib_lib();
    let logger = cfg.get_logger(&sinfo.compiler_host,
                                sinfo.stage.to_str());
    for lib in RUST_LIBS {
        println!("Building {} library lib{} for target triple {}...",
                 sinfo.stage.to_str(), lib, sinfo.target_triple);
        try!(compiler.rustc_lib_cmd(lib, &out_dir).tee(&logger));
    }
    if !sinfo.is_stage2() {
        println!("Building {} rustc for target triple {}...",
                 sinfo.stage.to_str(), sinfo.target_triple);
        let rustc_exe = sinfo.rustlib_bin()
            .join(sinfo.target_triple.with_exe_ext("rustc"));
        try!(compiler.rustc_exe_cmd(&rustc_exe).tee(&logger));
    }
    if sinfo.is_stage1() {
        let rustdoc_libs = vec!["test", "rustdoc"];
        for lib in &rustdoc_libs {
            println!("Building stage1 library lib{} for target triple {}...",
                     lib, sinfo.target_triple);
            try!(compiler.rustc_lib_cmd(lib, &out_dir).tee(&logger));
        }
        println!("Building stage1 rustdoc for target triple {}...",
                 sinfo.target_triple);
        let rustdoc_exe = sinfo.rustlib_bin()
            .join(sinfo.target_triple.with_exe_ext("rustdoc"));
        try!(compiler.rustdoc_exe_cmd(&rustdoc_exe).tee(&logger));
    }
    continue_build()
}

// Promote the stagei artifacts built under
// stage`i`/lib/rustlib/<triple>/lib directory into stage`i+1`/lib
fn promote_to_next_stage(sinfo : &StageInfo, snext : &StageInfo)
                         -> BuildState<()> {
    use std::fs::{read_dir, copy};
    println!("Promoting {} to {}...",
             sinfo.stage.to_str(), snext.stage.to_str());
    let mut copylist : Vec<(PathBuf, PathBuf)> = vec![];
    let exe = sinfo.target_triple.with_exe_ext("rustc");
    copylist.push((sinfo.rustlib_bin().join(&exe),
                   snext.bin_dir().join(&exe)));
    if sinfo.is_stage1() {
        let rustdoc = sinfo.target_triple.with_exe_ext("rustdoc");
        copylist.push((sinfo.rustlib_bin().join(&rustdoc),
                       snext.bin_dir().join(&rustdoc)));
    }
    for entry in try!(read_dir(&sinfo.rustlib_lib())
                      .map_err(|e| format!("Failed to read dir {:?}: {}",
                                           sinfo.rustlib_lib(), e))) {
        let entry = try!(entry).path();
        if let Some(ext) = entry.extension() {
            if OsStr::new(sinfo.target_triple.dylib_ext()) == ext {
                let filename = entry.file_name().unwrap();
                let to = snext.lib_dir().join(filename);
                copylist.push((entry.clone(), to));
            }
        }
    }
    for &(ref from, ref to) in &copylist {
        try!(copy(&from, &to).map_err(|e| {
            format!("Failed to copy {:?} to {:?}: {}", from, to, e)
        }));
    }
    continue_build()
}

fn create_dirs(sinfo : &StageInfo) {
    use std::fs::create_dir_all;
    let _ = create_dir_all(sinfo.bin_dir());
    let _ = create_dir_all(sinfo.lib_dir());
    let _ = create_dir_all(sinfo.rustlib_bin());
    let _ = create_dir_all(sinfo.rustlib_lib());
}

pub fn build_rust(cfg : &ConfigArgs) -> BuildState<()> {
    let stage0 = StageInfo::stage0(cfg);
    let stage1 = StageInfo::stage1(cfg);
    let stage2 : Vec<_> = cfg.target_triples().iter()
        .map(|target| StageInfo::stage2(cfg, target)).collect();
    create_dirs(&stage0);
    create_dirs(&stage1);
    let _ : Vec<_> = stage2.iter().map(|sinfo| create_dirs(&sinfo)).collect();

    if !cfg.no_bootstrap() {
        try!(build_stage(cfg, &stage0));
        try!(promote_to_next_stage(&stage0, &stage1));
    }
    try!(build_stage(cfg, &stage1));
    try!(promote_to_next_stage(&stage1, &stage2[0]));
    for sinfo in &stage2 {
        try!(build_stage(cfg, &sinfo));
    }
    continue_build()
}
