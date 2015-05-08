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
use std::path::{Path, PathBuf};
use std::process::Command;

use rustc_back::archive;
use session::Session;
use session::config;

/// Linker abstraction used by back::link to build up the command to invoke a
/// linker.
///
/// This trait is the total list of requirements needed by `back::link` and
/// represents the meaning of each option being passed down. This trait is then
/// used to dispatch on whether a GNU-like linker (generally `ld.exe`) or an
/// MSVC linker (e.g. `link.exe`) is being used.
pub trait Linker {
    fn link_dylib(&mut self, lib: &str);
    fn link_framework(&mut self, framework: &str);
    fn link_staticlib(&mut self, lib: &str);
    fn link_rlib(&mut self, lib: &Path);
    fn link_whole_staticlib(&mut self, lib: &str, search_path: &[PathBuf]);
    fn include_path(&mut self, path: &Path);
    fn framework_path(&mut self, path: &Path);
    fn output_filename(&mut self, path: &Path);
    fn add_object(&mut self, path: &Path);
    fn gc_sections(&mut self, is_dylib: bool);
    fn position_independent_executable(&mut self);
    fn optimize(&mut self);
    fn no_default_libraries(&mut self);
    fn build_dylib(&mut self, out_filename: &Path);
    fn args(&mut self, args: &[String]);
    fn hint_static(&mut self);
    fn hint_dynamic(&mut self);
    fn whole_archives(&mut self);
    fn no_whole_archives(&mut self);
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
            v.push(&archive::find_library(lib,
                                          &target.options.staticlib_prefix,
                                          &target.options.staticlib_suffix,
                                          search_path,
                                          &self.sess.diagnostic().handler));
            self.cmd.arg(&v);
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

    fn no_default_libraries(&mut self) {
        // Unfortunately right now passing -nodefaultlibs to gcc on windows
        // doesn't work so hot (in terms of native dependencies). This if
        // statement should hopefully be removed one day though!
        if !self.sess.target.target.options.is_like_windows {
            self.cmd.arg("-nodefaultlibs");
        }
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
}
