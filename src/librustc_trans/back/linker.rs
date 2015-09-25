// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter};
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

use back::archive;
use metadata::csearch;
use middle::dependency_format::Linkage;
use session::Session;
use session::config::CrateTypeDylib;
use session::config;
use syntax::ast;
use trans::CrateTranslation;

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
    fn gc_sections(&mut self, is_dylib: bool);
    fn position_independent_executable(&mut self);
    fn optimize(&mut self);
    fn debuginfo(&mut self);
    fn no_default_libraries(&mut self);
    fn build_dylib(&mut self, out_filename: &Path);
    fn args(&mut self, args: &[String]);
    fn hint_static(&mut self);
    fn hint_dynamic(&mut self);
    fn whole_archives(&mut self);
    fn no_whole_archives(&mut self);
    fn export_symbols(&mut self, sess: &Session, trans: &CrateTranslation,
                      tmpdir: &Path);
}

pub struct GnuLinker<'a> {
    pub cmd: &'a mut Command,
    pub sess: &'a Session,
}

impl<'a> GnuLinker<'a> {
    fn takes_hints(&self) -> bool {
        !self.sess.target.target.options.is_like_osx
    }
}

impl<'a> Linker for GnuLinker<'a> {
    fn link_dylib(&mut self, lib: &str) { self.cmd.arg("-l").arg(lib); }
    fn link_staticlib(&mut self, lib: &str) { self.cmd.arg("-l").arg(lib); }
    fn link_rlib(&mut self, lib: &Path) { self.cmd.arg(lib); }
    fn include_path(&mut self, path: &Path) { self.cmd.arg("-L").arg(path); }
    fn framework_path(&mut self, path: &Path) { self.cmd.arg("-F").arg(path); }
    fn output_filename(&mut self, path: &Path) { self.cmd.arg("-o").arg(path); }
    fn add_object(&mut self, path: &Path) { self.cmd.arg(path); }
    fn position_independent_executable(&mut self) { self.cmd.arg("-pie"); }
    fn args(&mut self, args: &[String]) { self.cmd.args(args); }

    fn link_rust_dylib(&mut self, lib: &str, _path: &Path) {
        self.cmd.arg("-l").arg(lib);
    }

    fn link_framework(&mut self, framework: &str) {
        self.cmd.arg("-framework").arg(framework);
    }

    fn link_whole_staticlib(&mut self, lib: &str, search_path: &[PathBuf]) {
        let target = &self.sess.target.target;
        if !target.options.is_like_osx {
            self.cmd.arg("-Wl,--whole-archive")
                    .arg("-l").arg(lib)
                    .arg("-Wl,--no-whole-archive");
        } else {
            // -force_load is the OSX equivalent of --whole-archive, but it
            // involves passing the full path to the library to link.
            let mut v = OsString::from("-Wl,-force_load,");
            v.push(&archive::find_library(lib, search_path, &self.sess));
            self.cmd.arg(&v);
        }
    }

    fn link_whole_rlib(&mut self, lib: &Path) {
        if self.sess.target.target.options.is_like_osx {
            let mut v = OsString::from("-Wl,-force_load,");
            v.push(lib);
            self.cmd.arg(&v);
        } else {
            self.cmd.arg("-Wl,--whole-archive").arg(lib)
                    .arg("-Wl,--no-whole-archive");
        }
    }

    fn gc_sections(&mut self, is_dylib: bool) {
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
            self.cmd.arg("-Wl,-dead_strip");

        // If we're building a dylib, we don't use --gc-sections because LLVM
        // has already done the best it can do, and we also don't want to
        // eliminate the metadata. If we're building an executable, however,
        // --gc-sections drops the size of hello world from 1.8MB to 597K, a 67%
        // reduction.
        } else if !is_dylib {
            self.cmd.arg("-Wl,--gc-sections");
        }
    }

    fn optimize(&mut self) {
        if !self.sess.target.target.options.linker_is_gnu { return }

        // GNU-style linkers support optimization with -O. GNU ld doesn't
        // need a numeric argument, but other linkers do.
        if self.sess.opts.optimize == config::Default ||
           self.sess.opts.optimize == config::Aggressive {
            self.cmd.arg("-Wl,-O1");
        }
    }

    fn debuginfo(&mut self) {
        // Don't do anything special here for GNU-style linkers.
    }

    fn no_default_libraries(&mut self) {
        self.cmd.arg("-nodefaultlibs");
    }

    fn build_dylib(&mut self, out_filename: &Path) {
        // On mac we need to tell the linker to let this library be rpathed
        if self.sess.target.target.options.is_like_osx {
            self.cmd.args(&["-dynamiclib", "-Wl,-dylib"]);

            if self.sess.opts.cg.rpath {
                let mut v = OsString::from("-Wl,-install_name,@rpath/");
                v.push(out_filename.file_name().unwrap());
                self.cmd.arg(&v);
            }
        } else {
            self.cmd.arg("-shared");
        }
    }

    fn whole_archives(&mut self) {
        if !self.takes_hints() { return }
        self.cmd.arg("-Wl,--whole-archive");
    }

    fn no_whole_archives(&mut self) {
        if !self.takes_hints() { return }
        self.cmd.arg("-Wl,--no-whole-archive");
    }

    fn hint_static(&mut self) {
        if !self.takes_hints() { return }
        self.cmd.arg("-Wl,-Bstatic");
    }

    fn hint_dynamic(&mut self) {
        if !self.takes_hints() { return }
        self.cmd.arg("-Wl,-Bdynamic");
    }

    fn export_symbols(&mut self, _: &Session, _: &CrateTranslation, _: &Path) {
        // noop, visibility in object files takes care of this
    }
}

pub struct MsvcLinker<'a> {
    pub cmd: &'a mut Command,
    pub sess: &'a Session,
}

impl<'a> Linker for MsvcLinker<'a> {
    fn link_rlib(&mut self, lib: &Path) { self.cmd.arg(lib); }
    fn add_object(&mut self, path: &Path) { self.cmd.arg(path); }
    fn args(&mut self, args: &[String]) { self.cmd.args(args); }
    fn build_dylib(&mut self, _out_filename: &Path) { self.cmd.arg("/DLL"); }
    fn gc_sections(&mut self, _is_dylib: bool) { self.cmd.arg("/OPT:REF,ICF"); }

    fn link_dylib(&mut self, lib: &str) {
        self.cmd.arg(&format!("{}.lib", lib));
    }

    fn link_rust_dylib(&mut self, lib: &str, path: &Path) {
        // When producing a dll, the MSVC linker may not actually emit a
        // `foo.lib` file if the dll doesn't actually export any symbols, so we
        // check to see if the file is there and just omit linking to it if it's
        // not present.
        let name = format!("{}.lib", lib);
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
        panic!("frameworks are not supported on windows")
    }
    fn link_framework(&mut self, _framework: &str) {
        panic!("frameworks are not supported on windows")
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
    }

    fn whole_archives(&mut self) {
        // hints not supported?
    }
    fn no_whole_archives(&mut self) {
        // hints not supported?
    }

    // On windows static libraries are of the form `foo.lib` and dynamic
    // libraries are not linked against directly, but rather through their
    // import libraries also called `foo.lib`. As a result there's no
    // possibility for a native library to appear both dynamically and
    // statically in the same folder so we don't have to worry about hints like
    // we do on Unix platforms.
    fn hint_static(&mut self) {}
    fn hint_dynamic(&mut self) {}

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
    fn export_symbols(&mut self, sess: &Session, trans: &CrateTranslation,
                      tmpdir: &Path) {
        let path = tmpdir.join("lib.def");
        let res = (|| -> io::Result<()> {
            let mut f = BufWriter::new(try!(File::create(&path)));

            // Start off with the standard module name header and then go
            // straight to exports.
            try!(writeln!(f, "LIBRARY"));
            try!(writeln!(f, "EXPORTS"));

            // Write out all our local symbols
            for sym in trans.reachable.iter() {
                try!(writeln!(f, "  {}", sym));
            }

            // Take a look at how all upstream crates are linked into this
            // dynamic library. For all statically linked libraries we take all
            // their reachable symbols and emit them as well.
            let cstore = &sess.cstore;
            let formats = sess.dependency_formats.borrow();
            let symbols = formats[&CrateTypeDylib].iter();
            let symbols = symbols.enumerate().filter_map(|(i, f)| {
                if *f == Linkage::Static {
                    Some((i + 1) as ast::CrateNum)
                } else {
                    None
                }
            }).flat_map(|cnum| {
                csearch::get_reachable_ids(cstore, cnum)
            }).map(|did| {
                csearch::get_symbol(cstore, did)
            });
            for symbol in symbols {
                try!(writeln!(f, "  {}", symbol));
            }
            Ok(())
        })();
        if let Err(e) = res {
            sess.fatal(&format!("failed to write lib.def file: {}", e));
        }
        let mut arg = OsString::from("/DEF:");
        arg.push(path);
        self.cmd.arg(&arg);
    }
}
