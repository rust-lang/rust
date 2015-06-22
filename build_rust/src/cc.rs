// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Invoke the appropriate toolchain to build and assemble a static
//! library. Different flavours of the toolchain (gnu vs. msvc) are
//! abstracted behind the trait `Toolchain`, and the build system
//! will select the appropriate one based on the build configuration
//! and the platform.

use std::fmt;
use std::ffi::{OsStr, OsString};
use std::path::{PathBuf, Path};
use std::process::Command;
use llvm::LLVMTools;
use build_state::*;
use configure::ConfigArgs;
use log::{Tee, Logger};

/// Specify a target triple, in the format `arch-vendor-os-abi`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Triple {
    triple : String
}

impl Triple {
    pub fn new(triple : &str) -> Result<Triple, String> {
        let v : Vec<&str>= triple.split('-').map(|s| s).collect();
        if v.len() < 3 {
            Err(format!("Invalid target triple {}.", triple))
        } else {
            Ok(Triple { triple : triple.into() })
        }
    }

    pub fn arch(&self) -> &str {
        self.triple.split('-').nth(0).unwrap()
    }

    pub fn os(&self) -> &str {
        self.triple.split('-').nth(2).unwrap()
    }

    pub fn abi(&self) -> Option<&str> {
        self.triple.split('-').nth(3)
    }

    pub fn is_i686(&self) -> bool {
        self.arch() == "i686"
    }

    pub fn is_x86_64(&self) -> bool {
        self.arch() == "x86_64"
    }

    pub fn is_windows(&self) -> bool {
        self.os() == "windows"
    }

    pub fn is_mingw(&self) -> bool {
        self.is_windows() && self.abi() == Some("gnu")
    }

    pub fn is_msvc(&self) -> bool {
        self.abi() == Some("msvc")
    }

    pub fn is_linux(&self) -> bool {
        self.os() == "linux"
    }

    pub fn is_darwin(&self) -> bool {
        self.os() == "darwin"
    }

    /// Append the extension for the executables in this platform
    pub fn with_exe_ext(&self, name : &str) -> String {
        if self.is_windows() {
            format!("{}.exe", name)
        } else {
            format!("{}", name)
        }
    }

    /// Append the extension for the executables in this platform
    pub fn with_lib_ext(&self, name : &str) -> String {
        if self.is_msvc() {
            format!("{}.lib", name)
        } else {
            format!("lib{}.a", name)
        }
    }

    /// Get the file extension for the dynamic libraries
    /// in this platform.
    pub fn dylib_ext(&self) -> &'static str {
        if self.is_windows() {
            "dll"
        } else if self.is_darwin() {
            "dylib"
        } else {
            "so"
        }
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.triple)
    }
}

impl AsRef<OsStr> for Triple {
    fn as_ref(&self) -> &OsStr {
        &OsStr::new(&self.triple)
    }
}

impl AsRef<Path> for Triple {
    fn as_ref(&self) -> &Path {
        &Path::new(&self.triple)
    }
}

impl<'a> From<&'a Triple> for &'a str {
    fn from(t : &'a Triple) -> &'a str {
        &t.triple
    }
}

impl<'a> From<&'a Triple> for String {
    fn from(t : &'a Triple) -> String {
        t.triple.clone()
    }
}

/// Compile and assemble a static library. This will invoke the
/// appropriate toolchain command (or LLVM tool) to compile the
/// given list of files into object files and then invoke ar (or
/// equivalent) to assemble them into a static library.
///
/// If a directory is given instead of a file, all supported files
/// under that directory will be compiled.
///
/// Supported file types are:
///
/// `*.ll` --- LLVM byte code, will invoke llc
///
/// `*.S`, `*.c`, `*.cc`, `*.cpp` --- will invoke appropriate
/// toolchain command
pub struct StaticLib {
    toolchain : Box<Toolchain>,
    llvm_tools : LLVMTools,
    include_directories: Vec<PathBuf>,
    files: Vec<PathBuf>,
    root_src_dir : PathBuf,
    root_build_dir : PathBuf,
    llvm_cxxflags : bool
}

impl StaticLib {
    /// This builder is finished with the `compile` function.
    fn new(toolchain : Box<Toolchain>, llvm_tools : LLVMTools)
           -> StaticLib {
        StaticLib {
            toolchain : toolchain,
            llvm_tools : llvm_tools,
            include_directories: Vec::new(),
            files: Vec::new(),
            root_src_dir : PathBuf::from("."),
            root_build_dir : PathBuf::from("."),
            llvm_cxxflags : false
        }
    }

    /// Set the source directory
    pub fn set_src_dir<P : AsRef<Path>>(&mut self, dir : P)
                                        -> &mut StaticLib {
        self.root_src_dir = dir.as_ref().into();
        self
    }

    /// Set the destination directory where the final artifact is written
    pub fn set_build_dir<P : AsRef<Path>>(&mut self, dir : P)
                                          -> &mut StaticLib {
        self.root_build_dir = dir.as_ref().into();
        self
    }

    /// Add a directory to the `-I` or include path for headers
    pub fn include_dirs<P: AsRef<Path>>(&mut self, dir: &[P])
                                        -> &mut StaticLib {
        let _ : Vec<()> = dir.iter().map(|p| p.as_ref().into())
            .map(|p| self.include_directories.push(p))
            .collect();
        self
    }

    /// Add a group of files which will be compiled
    pub fn files<P : AsRef<Path>>(&mut self, files : &[P])
                                  -> &mut StaticLib {
        let _ : Vec<()> = files.iter().map(|p| p.as_ref().into())
            .map(|p| self.files.push(p))
            .collect();
        self
    }

    /// If set, will invoke llvm-config to add the appropriate cxxflags
    pub fn set_llvm_cxxflags(&mut self) -> &mut StaticLib {
        self.llvm_cxxflags = true;
        self
    }

    /// Run the compiler, generating the file `output`
    pub fn compile(&self, out_lib: &str, logger : &Logger)
                   -> BuildState<()> {
        use std::fs::{read_dir, create_dir_all};
        let mut src_files : Vec<PathBuf> = vec![];
        for path in &self.files {
            if let Ok(dir) = read_dir(self.root_src_dir.join(path)) {
                let _ : Vec<_> = dir.map(|ent| ent.map(|f| {
                    if let Some(ext) = f.path().extension() {
                        if ext == "ll" || ext == "cc" || ext == "cpp"
                            || ext == "c" || ext == "S" {
                                let file = path.join(
                                    f.path().file_name().unwrap());
                                src_files.push(file);
                            }
                    }
                })).collect();
            } else {
                src_files.push(path.into());
            }
        }

        let mut objects = Vec::new();
        for file in &src_files {
            let src = self.root_src_dir.join(file);
            let obj = self.root_build_dir.join(file).with_extension("o");
            let _ = create_dir_all(&obj.parent().unwrap()); // ignore errors
            let mut cmd = self.compiler_cmd(&src, &obj);
            if self.llvm_cxxflags {
                let cxxflags = try!(self.llvm_tools.get_llvm_cxxflags());
                cmd.args(&cxxflags);
            }
            try!(cmd.tee(logger));
            objects.push(obj);
        }

        let output = self.root_build_dir
            .join(self.toolchain.target_triple().with_lib_ext(out_lib));
        self.toolchain.ar_cmd(&objects, &output).tee(logger)
    }

    fn compiler_cmd(&self, src : &Path, obj : &Path) -> Command {
        let inc_dirs = &self.include_directories;
        let ext = src.extension().expect(
            &format!("Source {:?} file has no extension.", src));
        if ext == "ll" {
            self.llvm_tools.llc_cmd(
                self.toolchain.target_triple(), src, obj)
        } else if ext == "cc" || ext == "cpp" {
            self.toolchain.cxx_cmd(src, obj, inc_dirs)
        } else {
            self.toolchain.cc_cmd(src, obj, inc_dirs)
        }
    }
}

/// Define the abstract interface that a toolchain implemenation
/// should support.
pub trait Toolchain {
    fn target_triple(&self) -> &Triple;
    fn cc_cmd(&self, src_file : &Path, obj_file : &Path,
              inc_dirs : &[PathBuf]) -> Command;
    fn cxx_cmd(&self, src_file : &Path, obj_file : &Path,
               inc_dirs : &[PathBuf]) -> Command;
    fn ar_cmd(&self, obj_files : &[PathBuf], output : &Path) -> Command;
}

/// Gnu-flavoured toolchain. Supports both gcc and clang.
struct GccishToolchain {
    target_triple : Triple,
    cc_cmd : String,
    cxx_cmd : String,
    ar_cmd : String
}

impl GccishToolchain {
    fn cross_gcc(triple : &Triple) -> GccishToolchain {
        GccishToolchain {
            target_triple : triple.clone(),
            cc_cmd : format!("{}-gcc", triple),
            cxx_cmd : format!("{}-g++", triple),
            ar_cmd : "ar".into()
        }
    }

    fn native_gcc(triple : &Triple) -> GccishToolchain {
        GccishToolchain {
            target_triple : triple.clone(),
            cc_cmd : "gcc".into(),
            cxx_cmd : "g++".into(),
            ar_cmd : "ar".into()
        }
    }

    fn clang(triple : &Triple) -> GccishToolchain {
        GccishToolchain {
            target_triple : triple.clone(),
            cc_cmd : "clang".into(),
            cxx_cmd : "clang++".into(),
            ar_cmd : "ar".into()
        }
    }

    fn add_args(&self, cmd : &mut Command, src : &Path, obj : &Path,
                inc_dirs : &[PathBuf]) {
        cmd.arg("-c").arg("-ffunction-sections").arg("-fdata-sections");

        let target = &self.target_triple;
        if target.is_windows() {
            cmd.arg("-mwin32");
        }

        if target.is_i686() {
            cmd.arg("-m32");
        } else if target.is_x86_64() {
            cmd.arg("-m64");
        }

        if !target.is_i686() {
            cmd.arg("-fPIC");
        }

        if target.is_darwin() {
            // for some reason clang on darwin doesn't seem to define this
            cmd.arg("-DCHAR_BIT=8");
        }

        for directory in inc_dirs {
            cmd.arg("-I").arg(directory);
        }

        cmd.arg("-o").arg(obj).arg(src);
    }
}

impl Toolchain for GccishToolchain {
    fn target_triple(&self) -> &Triple {
        &self.target_triple
    }

    fn cc_cmd(&self, src_file : &Path, obj_file : &Path,
              inc_dirs : &[PathBuf]) -> Command {
        let mut cmd = Command::new(&self.cc_cmd);
        self.add_args(&mut cmd, src_file, obj_file, inc_dirs);
        return cmd;
    }

    fn cxx_cmd(&self, src_file : &Path, obj_file : &Path,
               inc_dirs : &[PathBuf]) -> Command {
        let mut cmd = Command::new(&self.cxx_cmd);
        self.add_args(&mut cmd, src_file, obj_file, inc_dirs);
        cmd.arg("-fno-rtti");
        return cmd;
    }

    fn ar_cmd(&self, obj_files : &[PathBuf], output : &Path) -> Command {
        let mut cmd = Command::new(&self.ar_cmd);
        cmd.arg("crus");
        cmd.arg(&output);
        cmd.args(obj_files);
        cmd
    }
}

/// MSVC toolchain
struct MsvcToolchain {
    target_triple : Triple
}

impl MsvcToolchain {
    fn new(triple : &Triple) -> MsvcToolchain {
        MsvcToolchain {
            target_triple : triple.clone()
        }
    }
}

impl Toolchain for MsvcToolchain {
    fn target_triple(&self) -> &Triple {
        &self.target_triple
    }

    fn cc_cmd(&self, src_file : &Path, obj_file : &Path,
              inc_dirs : &[PathBuf]) -> Command {
        let mut cmd = Command::new("cl");

        cmd.arg("-nologo").arg("-c").arg(src_file);
        let _ : Vec<_> = inc_dirs.iter().map(|p| {
            let mut s = OsString::new();
            s.push("-I");
            s.push(&p);
            cmd.arg(&s);
        }).collect();

        let mut out = OsString::new();
        out.push("-Fo");
        out.push(obj_file);
        cmd.arg(&out);

        cmd
    }

    fn cxx_cmd(&self, src_file : &Path, obj_file : &Path,
               inc_dirs : &[PathBuf]) -> Command {
        self.cc_cmd(src_file, obj_file, inc_dirs)
    }

    fn ar_cmd(&self, obj_files : &[PathBuf], output : &Path) -> Command {
        let mut cmd = Command::new("lib");
        cmd.arg("-nologo");
        let mut s = OsString::new();
        s.push("-OUT:");
        s.push(&output);
        cmd.arg(&s);
        cmd.args(obj_files);
        cmd
    }
}

pub fn build_static_lib(cfg : &ConfigArgs, triple : &Triple)
                        -> StaticLib {
    let toolchain : Box<Toolchain> = if triple.is_darwin() {
        Box::new(GccishToolchain::clang(triple))
    } else if triple.is_mingw() {
        Box::new(GccishToolchain::native_gcc(triple))
    } else if triple.is_msvc() {
        Box::new(MsvcToolchain::new(triple))
    } else {
        Box::new(GccishToolchain::cross_gcc(triple))
    };
    StaticLib::new(toolchain, cfg.llvm_tools(triple))
}
