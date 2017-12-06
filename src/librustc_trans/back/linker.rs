// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::prelude::*;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

use back::archive;
use back::command::Command;
use back::symbol_export;
use rustc::hir::def_id::{LOCAL_CRATE, CrateNum};
use rustc::middle::dependency_format::Linkage;
use rustc::session::Session;
use rustc::session::config::{self, CrateType, OptLevel, DebugInfoLevel};
use rustc::ty::TyCtxt;
use rustc_back::LinkerFlavor;
use serialize::{json, Encoder};

/// For all the linkers we support, and information they might
/// need out of the shared crate context before we get rid of it.
pub struct LinkerInfo {
    exports: HashMap<CrateType, Vec<String>>,
}

impl LinkerInfo {
    pub fn new(tcx: TyCtxt) -> LinkerInfo {
        LinkerInfo {
            exports: tcx.sess.crate_types.borrow().iter().map(|&c| {
                (c, exported_symbols(tcx, c))
            }).collect(),
        }
    }

    pub fn to_linker<'a>(&'a self,
                         cmd: Command,
                         sess: &'a Session) -> Box<Linker+'a> {
        match sess.linker_flavor() {
            LinkerFlavor::Msvc => {
                Box::new(MsvcLinker {
                    cmd,
                    sess,
                    info: self
                }) as Box<Linker>
            }
            LinkerFlavor::Em =>  {
                Box::new(EmLinker {
                    cmd,
                    sess,
                    info: self
                }) as Box<Linker>
            }
            LinkerFlavor::Gcc =>  {
                Box::new(GccLinker {
                    cmd,
                    sess,
                    info: self,
                    hinted_static: false,
                    is_ld: false,
                }) as Box<Linker>
            }
            LinkerFlavor::Ld => {
                Box::new(GccLinker {
                    cmd,
                    sess,
                    info: self,
                    hinted_static: false,
                    is_ld: true,
                }) as Box<Linker>
            }
            LinkerFlavor::Binaryen => {
                panic!("can't instantiate binaryen linker")
            }
        }
    }
}

/// Linker abstraction used by back::link to build up the command to invoke a
/// linker.
///
/// This trait is the total list of requirements needed by `back::link` and
/// represents the meaning of each option being passed down. This trait is then
/// used to dispatch on whether a GNU-like linker (generally `ld.exe`) or an
/// MSVC linker (e.g. `link.exe`) is being used.
pub trait Linker {
    fn link_dylib(&mut self, lib: &str);
    fn link_rust_dylib(&mut self, lib: &str, path: &Path);
    fn link_framework(&mut self, framework: &str);
    fn link_staticlib(&mut self, lib: &str);
    fn link_rlib(&mut self, lib: &Path);
    fn link_whole_rlib(&mut self, lib: &Path);
    fn link_whole_staticlib(&mut self, lib: &str, search_path: &[PathBuf]);
    fn include_path(&mut self, path: &Path);
    fn framework_path(&mut self, path: &Path);
    fn output_filename(&mut self, path: &Path);
    fn add_object(&mut self, path: &Path);
    fn gc_sections(&mut self, keep_metadata: bool);
    fn position_independent_executable(&mut self);
    fn partial_relro(&mut self);
    fn full_relro(&mut self);
    fn optimize(&mut self);
    fn debuginfo(&mut self);
    fn no_default_libraries(&mut self);
    fn build_dylib(&mut self, out_filename: &Path);
    fn build_static_executable(&mut self);
    fn args(&mut self, args: &[String]);
    fn export_symbols(&mut self, tmpdir: &Path, crate_type: CrateType);
    fn subsystem(&mut self, subsystem: &str);
    // Should have been finalize(self), but we don't support self-by-value on trait objects (yet?).
    fn finalize(&mut self) -> Command;
}

pub struct GccLinker<'a> {
    cmd: Command,
    sess: &'a Session,
    info: &'a LinkerInfo,
    hinted_static: bool, // Keeps track of the current hinting mode.
    // Link as ld
    is_ld: bool,
}

impl<'a> GccLinker<'a> {
    /// Argument that must be passed *directly* to the linker
    ///
    /// These arguments need to be prepended with '-Wl,' when a gcc-style linker is used
    fn linker_arg<S>(&mut self, arg: S) -> &mut Self
        where S: AsRef<OsStr>
    {
        if !self.is_ld {
            let mut os = OsString::from("-Wl,");
            os.push(arg.as_ref());
            self.cmd.arg(os);
        } else {
            self.cmd.arg(arg);
        }
        self
    }

    fn takes_hints(&self) -> bool {
        !self.sess.target.target.options.is_like_osx
    }

    // Some platforms take hints about whether a library is static or dynamic.
    // For those that support this, we ensure we pass the option if the library
    // was flagged "static" (most defaults are dynamic) to ensure that if
    // libfoo.a and libfoo.so both exist that the right one is chosen.
    fn hint_static(&mut self) {
        if !self.takes_hints() { return }
        if !self.hinted_static {
            self.linker_arg("-Bstatic");
            self.hinted_static = true;
        }
    }

    fn hint_dynamic(&mut self) {
        if !self.takes_hints() { return }
        if self.hinted_static {
            self.linker_arg("-Bdynamic");
            self.hinted_static = false;
        }
    }
}

impl<'a> Linker for GccLinker<'a> {
    fn link_dylib(&mut self, lib: &str) { self.hint_dynamic(); self.cmd.arg("-l").arg(lib); }
    fn link_staticlib(&mut self, lib: &str) { self.hint_static(); self.cmd.arg("-l").arg(lib); }
    fn link_rlib(&mut self, lib: &Path) { self.hint_static(); self.cmd.arg(lib); }
    fn include_path(&mut self, path: &Path) { self.cmd.arg("-L").arg(path); }
    fn framework_path(&mut self, path: &Path) { self.cmd.arg("-F").arg(path); }
    fn output_filename(&mut self, path: &Path) { self.cmd.arg("-o").arg(path); }
    fn add_object(&mut self, path: &Path) { self.cmd.arg(path); }
    fn position_independent_executable(&mut self) { self.cmd.arg("-pie"); }
    fn partial_relro(&mut self) { self.linker_arg("-z,relro"); }
    fn full_relro(&mut self) { self.linker_arg("-z,relro,-z,now"); }
    fn build_static_executable(&mut self) { self.cmd.arg("-static"); }
    fn args(&mut self, args: &[String]) { self.cmd.args(args); }

    fn link_rust_dylib(&mut self, lib: &str, _path: &Path) {
        self.hint_dynamic();
        self.cmd.arg("-l").arg(lib);
    }

    fn link_framework(&mut self, framework: &str) {
        self.hint_dynamic();
        self.cmd.arg("-framework").arg(framework);
    }

    // Here we explicitly ask that the entire archive is included into the
    // result artifact. For more details see #15460, but the gist is that
    // the linker will strip away any unused objects in the archive if we
    // don't otherwise explicitly reference them. This can occur for
    // libraries which are just providing bindings, libraries with generic
    // functions, etc.
    fn link_whole_staticlib(&mut self, lib: &str, search_path: &[PathBuf]) {
        self.hint_static();
        let target = &self.sess.target.target;
        if !target.options.is_like_osx {
            self.linker_arg("--whole-archive").cmd.arg("-l").arg(lib);
            self.linker_arg("--no-whole-archive");
        } else {
            // -force_load is the macOS equivalent of --whole-archive, but it
            // involves passing the full path to the library to link.
            let mut v = OsString::from("-force_load,");
            v.push(&archive::find_library(lib, search_path, &self.sess));
            self.linker_arg(&v);
        }
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        self.hint_static();
        if self.sess.target.target.options.is_like_osx {
            let mut v = OsString::from("-force_load,");
            v.push(lib);
            self.linker_arg(&v);
        } else {
            self.linker_arg("--whole-archive").cmd.arg(lib);
            self.linker_arg("--no-whole-archive");
        }
    }

    fn gc_sections(&mut self, keep_metadata: bool) {
        // The dead_strip option to the linker specifies that functions and data
        // unreachable by the entry point will be removed. This is quite useful
        // with Rust's compilation model of compiling libraries at a time into
        // one object file. For example, this brings hello world from 1.7MB to
        // 458K.
        //
        // Note that this is done for both executables and dynamic libraries. We
        // won't get much benefit from dylibs because LLVM will have already
        // stripped away as much as it could. This has not been seen to impact
        // link times negatively.
        //
        // -dead_strip can't be part of the pre_link_args because it's also used
        // for partial linking when using multiple codegen units (-r).  So we
        // insert it here.
        if self.sess.target.target.options.is_like_osx {
            self.linker_arg("-dead_strip");
        } else if self.sess.target.target.options.is_like_solaris {
            self.linker_arg("-z");
            self.linker_arg("ignore");

        // If we're building a dylib, we don't use --gc-sections because LLVM
        // has already done the best it can do, and we also don't want to
        // eliminate the metadata. If we're building an executable, however,
        // --gc-sections drops the size of hello world from 1.8MB to 597K, a 67%
        // reduction.
        } else if !keep_metadata {
            self.linker_arg("--gc-sections");
        }
    }

    fn optimize(&mut self) {
        if !self.sess.target.target.options.linker_is_gnu { return }

        // GNU-style linkers support optimization with -O. GNU ld doesn't
        // need a numeric argument, but other linkers do.
        if self.sess.opts.optimize == config::OptLevel::Default ||
           self.sess.opts.optimize == config::OptLevel::Aggressive {
            self.linker_arg("-O1");
        }
    }

    fn debuginfo(&mut self) {
        // Don't do anything special here for GNU-style linkers.
    }

    fn no_default_libraries(&mut self) {
        if !self.is_ld {
            self.cmd.arg("-nodefaultlibs");
        }
    }

    fn build_dylib(&mut self, out_filename: &Path) {
        // On mac we need to tell the linker to let this library be rpathed
        if self.sess.target.target.options.is_like_osx {
            self.cmd.arg("-dynamiclib");
            self.linker_arg("-dylib");

            // Note that the `osx_rpath_install_name` option here is a hack
            // purely to support rustbuild right now, we should get a more
            // principled solution at some point to force the compiler to pass
            // the right `-Wl,-install_name` with an `@rpath` in it.
            if self.sess.opts.cg.rpath ||
               self.sess.opts.debugging_opts.osx_rpath_install_name {
                let mut v = OsString::from("-install_name,@rpath/");
                v.push(out_filename.file_name().unwrap());
                self.linker_arg(&v);
            }
        } else {
            self.cmd.arg("-shared");
        }
    }

    fn export_symbols(&mut self, tmpdir: &Path, crate_type: CrateType) {
        // If we're compiling a dylib, then we let symbol visibility in object
        // files to take care of whether they're exported or not.
        //
        // If we're compiling a cdylib, however, we manually create a list of
        // exported symbols to ensure we don't expose any more. The object files
        // have far more public symbols than we actually want to export, so we
        // hide them all here.
        if crate_type == CrateType::CrateTypeDylib ||
           crate_type == CrateType::CrateTypeProcMacro {
            return
        }

        let mut arg = OsString::new();
        let path = tmpdir.join("list");

        debug!("EXPORTED SYMBOLS:");

        if self.sess.target.target.options.is_like_osx {
            // Write a plain, newline-separated list of symbols
            let res = (|| -> io::Result<()> {
                let mut f = BufWriter::new(File::create(&path)?);
                for sym in self.info.exports[&crate_type].iter() {
                    debug!("  _{}", sym);
                    writeln!(f, "_{}", sym)?;
                }
                Ok(())
            })();
            if let Err(e) = res {
                self.sess.fatal(&format!("failed to write lib.def file: {}", e));
            }
        } else {
            // Write an LD version script
            let res = (|| -> io::Result<()> {
                let mut f = BufWriter::new(File::create(&path)?);
                writeln!(f, "{{\n  global:")?;
                for sym in self.info.exports[&crate_type].iter() {
                    debug!("    {};", sym);
                    writeln!(f, "    {};", sym)?;
                }
                writeln!(f, "\n  local:\n    *;\n}};")?;
                Ok(())
            })();
            if let Err(e) = res {
                self.sess.fatal(&format!("failed to write version script: {}", e));
            }
        }

        if self.sess.target.target.options.is_like_osx {
            if !self.is_ld {
                arg.push("-Wl,")
            }
            arg.push("-exported_symbols_list,");
        } else if self.sess.target.target.options.is_like_solaris {
            if !self.is_ld {
                arg.push("-Wl,")
            }
            arg.push("-M,");
        } else {
            if !self.is_ld {
                arg.push("-Wl,")
            }
            arg.push("--version-script=");
        }

        arg.push(&path);
        self.cmd.arg(arg);
    }

    fn subsystem(&mut self, subsystem: &str) {
        self.linker_arg(&format!("--subsystem,{}", subsystem));
    }

    fn finalize(&mut self) -> Command {
        self.hint_dynamic(); // Reset to default before returning the composed command line.
        let mut cmd = Command::new("");
        ::std::mem::swap(&mut cmd, &mut self.cmd);
        cmd
    }
}

pub struct MsvcLinker<'a> {
    cmd: Command,
    sess: &'a Session,
    info: &'a LinkerInfo
}

impl<'a> Linker for MsvcLinker<'a> {
    fn link_rlib(&mut self, lib: &Path) { self.cmd.arg(lib); }
    fn add_object(&mut self, path: &Path) { self.cmd.arg(path); }
    fn args(&mut self, args: &[String]) { self.cmd.args(args); }

    fn build_dylib(&mut self, out_filename: &Path) {
        self.cmd.arg("/DLL");
        let mut arg: OsString = "/IMPLIB:".into();
        arg.push(out_filename.with_extension("dll.lib"));
        self.cmd.arg(arg);
    }

    fn build_static_executable(&mut self) {
        // noop
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        // MSVC's ICF (Identical COMDAT Folding) link optimization is
        // slow for Rust and thus we disable it by default when not in
        // optimization build.
        if self.sess.opts.optimize != config::OptLevel::No {
            self.cmd.arg("/OPT:REF,ICF");
        } else {
            // It is necessary to specify NOICF here, because /OPT:REF
            // implies ICF by default.
            self.cmd.arg("/OPT:REF,NOICF");
        }
    }

    fn link_dylib(&mut self, lib: &str) {
        self.cmd.arg(&format!("{}.lib", lib));
    }

    fn link_rust_dylib(&mut self, lib: &str, path: &Path) {
        // When producing a dll, the MSVC linker may not actually emit a
        // `foo.lib` file if the dll doesn't actually export any symbols, so we
        // check to see if the file is there and just omit linking to it if it's
        // not present.
        let name = format!("{}.dll.lib", lib);
        if fs::metadata(&path.join(&name)).is_ok() {
            self.cmd.arg(name);
        }
    }

    fn link_staticlib(&mut self, lib: &str) {
        self.cmd.arg(&format!("{}.lib", lib));
    }

    fn position_independent_executable(&mut self) {
        // noop
    }

    fn partial_relro(&mut self) {
        // noop
    }

    fn full_relro(&mut self) {
        // noop
    }

    fn no_default_libraries(&mut self) {
        // Currently we don't pass the /NODEFAULTLIB flag to the linker on MSVC
        // as there's been trouble in the past of linking the C++ standard
        // library required by LLVM. This likely needs to happen one day, but
        // in general Windows is also a more controlled environment than
        // Unix, so it's not necessarily as critical that this be implemented.
        //
        // Note that there are also some licensing worries about statically
        // linking some libraries which require a specific agreement, so it may
        // not ever be possible for us to pass this flag.
    }

    fn include_path(&mut self, path: &Path) {
        let mut arg = OsString::from("/LIBPATH:");
        arg.push(path);
        self.cmd.arg(&arg);
    }

    fn output_filename(&mut self, path: &Path) {
        let mut arg = OsString::from("/OUT:");
        arg.push(path);
        self.cmd.arg(&arg);
    }

    fn framework_path(&mut self, _path: &Path) {
        bug!("frameworks are not supported on windows")
    }
    fn link_framework(&mut self, _framework: &str) {
        bug!("frameworks are not supported on windows")
    }

    fn link_whole_staticlib(&mut self, lib: &str, _search_path: &[PathBuf]) {
        // not supported?
        self.link_staticlib(lib);
    }
    fn link_whole_rlib(&mut self, path: &Path) {
        // not supported?
        self.link_rlib(path);
    }
    fn optimize(&mut self) {
        // Needs more investigation of `/OPT` arguments
    }

    fn debuginfo(&mut self) {
        // This will cause the Microsoft linker to generate a PDB file
        // from the CodeView line tables in the object files.
        self.cmd.arg("/DEBUG");

        // This will cause the Microsoft linker to embed .natvis info into the the PDB file
        let sysroot = self.sess.sysroot();
        let natvis_dir_path = sysroot.join("lib\\rustlib\\etc");
        if let Ok(natvis_dir) = fs::read_dir(&natvis_dir_path) {
            // LLVM 5.0.0's lld-link frontend doesn't yet recognize, and chokes
            // on, the /NATVIS:... flags.  LLVM 6 (or earlier) should at worst ignore
            // them, eventually mooting this workaround, per this landed patch:
            // https://github.com/llvm-mirror/lld/commit/27b9c4285364d8d76bb43839daa100
            if let Some(ref linker_path) = self.sess.opts.cg.linker {
                if let Some(linker_name) = Path::new(&linker_path).file_stem() {
                    if linker_name.to_str().unwrap().to_lowercase() == "lld-link" {
                        self.sess.warn("not embedding natvis: lld-link may not support the flag");
                        return;
                    }
                }
            }
            for entry in natvis_dir {
                match entry {
                    Ok(entry) => {
                        let path = entry.path();
                        if path.extension() == Some("natvis".as_ref()) {
                            let mut arg = OsString::from("/NATVIS:");
                            arg.push(path);
                            self.cmd.arg(arg);
                        }
                    },
                    Err(err) => {
                        self.sess.warn(&format!("error enumerating natvis directory: {}", err));
                    },
                }
            }
        }
    }

    // Currently the compiler doesn't use `dllexport` (an LLVM attribute) to
    // export symbols from a dynamic library. When building a dynamic library,
    // however, we're going to want some symbols exported, so this function
    // generates a DEF file which lists all the symbols.
    //
    // The linker will read this `*.def` file and export all the symbols from
    // the dynamic library. Note that this is not as simple as just exporting
    // all the symbols in the current crate (as specified by `trans.reachable`)
    // but rather we also need to possibly export the symbols of upstream
    // crates. Upstream rlibs may be linked statically to this dynamic library,
    // in which case they may continue to transitively be used and hence need
    // their symbols exported.
    fn export_symbols(&mut self,
                      tmpdir: &Path,
                      crate_type: CrateType) {
        let path = tmpdir.join("lib.def");
        let res = (|| -> io::Result<()> {
            let mut f = BufWriter::new(File::create(&path)?);

            // Start off with the standard module name header and then go
            // straight to exports.
            writeln!(f, "LIBRARY")?;
            writeln!(f, "EXPORTS")?;
            for symbol in self.info.exports[&crate_type].iter() {
                debug!("  _{}", symbol);
                writeln!(f, "  {}", symbol)?;
            }
            Ok(())
        })();
        if let Err(e) = res {
            self.sess.fatal(&format!("failed to write lib.def file: {}", e));
        }
        let mut arg = OsString::from("/DEF:");
        arg.push(path);
        self.cmd.arg(&arg);
    }

    fn subsystem(&mut self, subsystem: &str) {
        // Note that previous passes of the compiler validated this subsystem,
        // so we just blindly pass it to the linker.
        self.cmd.arg(&format!("/SUBSYSTEM:{}", subsystem));

        // Windows has two subsystems we're interested in right now, the console
        // and windows subsystems. These both implicitly have different entry
        // points (starting symbols). The console entry point starts with
        // `mainCRTStartup` and the windows entry point starts with
        // `WinMainCRTStartup`. These entry points, defined in system libraries,
        // will then later probe for either `main` or `WinMain`, respectively to
        // start the application.
        //
        // In Rust we just always generate a `main` function so we want control
        // to always start there, so we force the entry point on the windows
        // subsystem to be `mainCRTStartup` to get everything booted up
        // correctly.
        //
        // For more information see RFC #1665
        if subsystem == "windows" {
            self.cmd.arg("/ENTRY:mainCRTStartup");
        }
    }

    fn finalize(&mut self) -> Command {
        let mut cmd = Command::new("");
        ::std::mem::swap(&mut cmd, &mut self.cmd);
        cmd
    }
}

pub struct EmLinker<'a> {
    cmd: Command,
    sess: &'a Session,
    info: &'a LinkerInfo
}

impl<'a> Linker for EmLinker<'a> {
    fn include_path(&mut self, path: &Path) {
        self.cmd.arg("-L").arg(path);
    }

    fn link_staticlib(&mut self, lib: &str) {
        self.cmd.arg("-l").arg(lib);
    }

    fn output_filename(&mut self, path: &Path) {
        self.cmd.arg("-o").arg(path);
    }

    fn add_object(&mut self, path: &Path) {
        self.cmd.arg(path);
    }

    fn link_dylib(&mut self, lib: &str) {
        // Emscripten always links statically
        self.link_staticlib(lib);
    }

    fn link_whole_staticlib(&mut self, lib: &str, _search_path: &[PathBuf]) {
        // not supported?
        self.link_staticlib(lib);
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        // not supported?
        self.link_rlib(lib);
    }

    fn link_rust_dylib(&mut self, lib: &str, _path: &Path) {
        self.link_dylib(lib);
    }

    fn link_rlib(&mut self, lib: &Path) {
        self.add_object(lib);
    }

    fn position_independent_executable(&mut self) {
        // noop
    }

    fn partial_relro(&mut self) {
        // noop
    }

    fn full_relro(&mut self) {
        // noop
    }

    fn args(&mut self, args: &[String]) {
        self.cmd.args(args);
    }

    fn framework_path(&mut self, _path: &Path) {
        bug!("frameworks are not supported on Emscripten")
    }

    fn link_framework(&mut self, _framework: &str) {
        bug!("frameworks are not supported on Emscripten")
    }

    fn gc_sections(&mut self, _keep_metadata: bool) {
        // noop
    }

    fn optimize(&mut self) {
        // Emscripten performs own optimizations
        self.cmd.arg(match self.sess.opts.optimize {
            OptLevel::No => "-O0",
            OptLevel::Less => "-O1",
            OptLevel::Default => "-O2",
            OptLevel::Aggressive => "-O3",
            OptLevel::Size => "-Os",
            OptLevel::SizeMin => "-Oz"
        });
        // Unusable until https://github.com/rust-lang/rust/issues/38454 is resolved
        self.cmd.args(&["--memory-init-file", "0"]);
    }

    fn debuginfo(&mut self) {
        // Preserve names or generate source maps depending on debug info
        self.cmd.arg(match self.sess.opts.debuginfo {
            DebugInfoLevel::NoDebugInfo => "-g0",
            DebugInfoLevel::LimitedDebugInfo => "-g3",
            DebugInfoLevel::FullDebugInfo => "-g4"
        });
    }

    fn no_default_libraries(&mut self) {
        self.cmd.args(&["-s", "DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]"]);
    }

    fn build_dylib(&mut self, _out_filename: &Path) {
        bug!("building dynamic library is unsupported on Emscripten")
    }

    fn build_static_executable(&mut self) {
        // noop
    }

    fn export_symbols(&mut self, _tmpdir: &Path, crate_type: CrateType) {
        let symbols = &self.info.exports[&crate_type];

        debug!("EXPORTED SYMBOLS:");

        self.cmd.arg("-s");

        let mut arg = OsString::from("EXPORTED_FUNCTIONS=");
        let mut encoded = String::new();

        {
            let mut encoder = json::Encoder::new(&mut encoded);
            let res = encoder.emit_seq(symbols.len(), |encoder| {
                for (i, sym) in symbols.iter().enumerate() {
                    encoder.emit_seq_elt(i, |encoder| {
                        encoder.emit_str(&("_".to_string() + sym))
                    })?;
                }
                Ok(())
            });
            if let Err(e) = res {
                self.sess.fatal(&format!("failed to encode exported symbols: {}", e));
            }
        }
        debug!("{}", encoded);
        arg.push(encoded);

        self.cmd.arg(arg);
    }

    fn subsystem(&mut self, _subsystem: &str) {
        // noop
    }

    fn finalize(&mut self) -> Command {
        let mut cmd = Command::new("");
        ::std::mem::swap(&mut cmd, &mut self.cmd);
        cmd
    }
}

fn exported_symbols(tcx: TyCtxt, crate_type: CrateType) -> Vec<String> {
    let mut symbols = Vec::new();

    let export_threshold = symbol_export::crates_export_threshold(&[crate_type]);
    for &(ref name, _, level) in tcx.exported_symbols(LOCAL_CRATE).iter() {
        if level.is_below_threshold(export_threshold) {
            symbols.push(name.clone());
        }
    }

    let formats = tcx.sess.dependency_formats.borrow();
    let deps = formats[&crate_type].iter();

    for (index, dep_format) in deps.enumerate() {
        let cnum = CrateNum::new(index + 1);
        // For each dependency that we are linking to statically ...
        if *dep_format == Linkage::Static {
            // ... we add its symbol list to our export list.
            for &(ref name, _, level) in tcx.exported_symbols(cnum).iter() {
                if level.is_below_threshold(export_threshold) {
                    symbols.push(name.clone());
                }
            }
        }
    }

    symbols
}
