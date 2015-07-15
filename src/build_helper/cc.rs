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

use std::ffi::OsString;
use std::path::{PathBuf, Path};
use std::process::Command;
use llvm::LLVMTools;
use config::Config;
use run::Run;

/// Append the extension for the executables in this platform
pub fn with_exe_ext(triple : &str, name : &str) -> String {
    if triple.contains("windows") {
        format!("{}.exe", name)
    } else {
        format!("{}", name)
    }
}

/// Append the extension for the static libraries in this platform
pub fn with_lib_ext(triple : &str, name : &str) -> String {
    if triple.contains("msvc") {
        format!("{}.lib", name)
    } else {
        format!("lib{}.a", name)
    }
}

/// Get the file extension for the dynamic libraries in this platform.
pub fn dylib_ext(triple : &str) -> &'static str {
    if triple.contains("windows") {
        "dll"
    } else if triple.contains("darwin") {
        "dylib"
    } else {
        "so"
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
/// `*.ll` --- LLVM byte code, will invoke `llc`
///
/// `*.S`, `*.c`, `*.cc`, `*.cpp` --- will invoke toolchain command
/// defined by trait `Toolchain`.
pub struct StaticLib {
    toolchain : Box<Toolchain>,
    llvm_tools : LLVMTools,
    include_directories: Vec<PathBuf>,
    files: Vec<PathBuf>,
    src_dir : PathBuf,
    build_dir : PathBuf,
    with_llvm : bool
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
            src_dir : PathBuf::new(),
            build_dir : PathBuf::new(),
            with_llvm : false
        }
    }

    /// Set the source directory
    pub fn set_src_dir<P : AsRef<Path>>(&mut self, dir : P)
                                        -> &mut StaticLib {
        self.src_dir = dir.as_ref().into();
        self
    }

    /// Set the destination directory where the final artifact is written
    pub fn set_build_dir<P : AsRef<Path>>(&mut self, dir : P)
                                          -> &mut StaticLib {
        self.build_dir = dir.as_ref().into();
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

    /// If set, will invoke llvm-config to add the appropriate compiler flags
    pub fn with_llvm(&mut self) -> &mut StaticLib {
        self.with_llvm = true;
        self
    }

    /// Run the compiler, generating the file `output`
    pub fn compile(&self, out_lib: &str) {
        use std::fs::{read_dir, create_dir_all};
        let mut src_files : Vec<PathBuf> = vec![];
        for path in &self.files {
            if let Ok(dir) = read_dir(self.src_dir.join(path)) {
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
            let src = self.src_dir.join(file);
            let obj = self.build_dir.join(file).with_extension("o");
            let _ = create_dir_all(&obj.parent().unwrap()); // ignore errors
            let mut cmd = self.compiler_cmd(&src, &obj);
            if self.with_llvm {
               let cxxflags = self.llvm_tools.get_llvm_cxxflags();
               cmd.args(&cxxflags);
            }
            cmd.run();
            objects.push(obj);
        }

        let output = self.build_dir
            .join(with_lib_ext(self.toolchain.target_triple(), out_lib));
        self.toolchain.ar_cmd(&objects, &output).run();
        println!("cargo:rustc-link-search=native={}", self.build_dir.display());
        if self.with_llvm {
            println!("cargo:rustc-link-search=native={}",
                     self.llvm_tools.path_to_llvm_libs().display());
        }
    }

    fn compiler_cmd(&self, src : &Path, obj : &Path) -> Command {
        let mut inc_dirs : Vec<PathBuf> = self.include_directories.iter()
            .map(|d| self.src_dir.join(d)).collect();
        if self.with_llvm {
            inc_dirs.push(self.llvm_tools.llvm_src_dir().join("include"));
        }
        let ext = src.extension().expect(
            &format!("Source {:?} file has no extension.", src));
        if ext == "ll" {
            self.llvm_tools.llc_cmd(
                self.toolchain.target_triple(), src, obj)
        } else if ext == "cc" || ext == "cpp" {
            self.toolchain.cxx_cmd(src, obj, &inc_dirs)
        } else {
            self.toolchain.cc_cmd(src, obj, &inc_dirs)
        }
    }
}

/// Define the abstract interface that a toolchain implemenation
/// should support.
pub trait Toolchain {
    fn target_triple(&self) -> &str;
    fn cc_cmd(&self, src_file : &Path, obj_file : &Path,
              inc_dirs : &[PathBuf]) -> Command;
    fn cxx_cmd(&self, src_file : &Path, obj_file : &Path,
               inc_dirs : &[PathBuf]) -> Command;
    fn ar_cmd(&self, obj_files : &[PathBuf], output : &Path) -> Command;
}

/// Gnu-flavoured toolchain. Support both gcc and clang.
pub struct GccishToolchain {
    target_triple : String,
    pub cc_cmd : String,
    cxx_cmd : String,
    pub ar_cmd : String
}

impl GccishToolchain {
    pub fn new(target : &str) -> GccishToolchain {
        if target.contains("darwin") {
            GccishToolchain::clang(target)
        } else if target.contains("windows") && target.contains("gnu") {
            GccishToolchain::native_gcc(target)
        } else {
            GccishToolchain::cross_gcc(target)
        }
    }

    fn cross_gcc(triple : &str) -> GccishToolchain {
        GccishToolchain {
            target_triple : triple.into(),
            cc_cmd : format!("{}-gcc", triple),
            cxx_cmd : format!("{}-g++", triple),
            ar_cmd : "ar".into()
        }
    }

    fn native_gcc(triple : &str) -> GccishToolchain {
        GccishToolchain {
            target_triple : triple.into(),
            cc_cmd : "gcc".into(),
            cxx_cmd : "g++".into(),
            ar_cmd : "ar".into()
        }
    }

    fn clang(triple : &str) -> GccishToolchain {
        GccishToolchain {
            target_triple : triple.into(),
            cc_cmd : "clang".into(),
            cxx_cmd : "clang++".into(),
            ar_cmd : "ar".into()
        }
    }

    pub fn cflags(&self) -> Vec<&'static str> {
        let target = self.target_triple();
        let mut cflags = vec!["-ffunction-sections", "-fdata-sections"];

        if target.contains("aarch64") {
            cflags.push("-D__aarch64__");
        }

        if target.contains("android") {
            cflags.push("-DANDROID");
            cflags.push("-D__ANDROID__");
        }

        if target.contains("windows") {
            cflags.push("-mwin32");
        }

        if target.contains("i686") {
            cflags.push("-m32");
        } else if target.contains("x86_64") {
            cflags.push("-m64");
        }

        if !target.contains("i686") {
            cflags.push("-fPIC");
        }

        cflags
    }

    fn add_args(&self, cmd : &mut Command, src : &Path, obj : &Path,
                inc_dirs : &[PathBuf]) {
        cmd.arg("-c").args(&self.cflags());

        let target = self.target_triple();

        if target.contains("darwin") {
            // for some reason clang on darwin doesn't seem to define this
            cmd.arg("-DCHAR_BIT=8");
        }

        for directory in inc_dirs {
            println!("{:?}", directory);
            cmd.arg("-I").arg(directory);
        }

        cmd.arg("-o").arg(obj).arg(src);
    }
}

impl Toolchain for GccishToolchain {
    fn target_triple(&self) -> &str {
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
    target_triple : String
}

impl MsvcToolchain {
    fn new(triple : &str) -> MsvcToolchain {
        MsvcToolchain {
            target_triple : triple.into()
        }
    }
}

impl Toolchain for MsvcToolchain {
    fn target_triple(&self) -> &str {
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

pub fn build_static_lib(cfg : &Config) -> StaticLib {
    let target = cfg.target();
    let toolchain : Box<Toolchain> = if target.contains("msvc") {
        Box::new(MsvcToolchain::new(target))
    } else {
        Box::new(GccishToolchain::new(target))
    };
    let mut slib = StaticLib::new(toolchain, LLVMTools::new(cfg));
    slib.set_src_dir(&cfg.src_dir());
    slib.set_build_dir(cfg.out_dir());
    slib
}
