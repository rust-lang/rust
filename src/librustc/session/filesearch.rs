// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

pub use self::FileMatch::*;

use std::borrow::Cow;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use session::search_paths::{SearchPaths, PathKind};
use util::fs as rustcfs;

#[derive(Copy, Clone)]
pub enum FileMatch {
    FileMatches,
    FileDoesntMatch,
}

// A module for searching for libraries
// FIXME (#2658): I'm not happy how this module turned out. Should
// probably just be folded into cstore.

pub struct FileSearch<'a> {
    pub sysroot: &'a Path,
    pub search_paths: &'a SearchPaths,
    pub triple: &'a str,
    pub kind: PathKind,
}

impl<'a> FileSearch<'a> {
    pub fn for_each_lib_search_path<F>(&self, mut f: F) where
        F: FnMut(&Path, PathKind)
    {
        let mut visited_dirs = HashSet::new();

        for (path, kind) in self.search_paths.iter(self.kind) {
            f(path, kind);
            visited_dirs.insert(path.to_path_buf());
        }

        debug!("filesearch: searching lib path");
        let tlib_path = make_target_lib_path(self.sysroot,
                                             self.triple);
        if !visited_dirs.contains(&tlib_path) {
            f(&tlib_path, PathKind::All);
        }

        visited_dirs.insert(tlib_path);
    }

    pub fn get_lib_path(&self) -> PathBuf {
        make_target_lib_path(self.sysroot, self.triple)
    }

    pub fn search<F>(&self, mut pick: F)
        where F: FnMut(&Path, PathKind) -> FileMatch
    {
        self.for_each_lib_search_path(|lib_search_path, kind| {
            debug!("searching {}", lib_search_path.display());
            let files = match fs::read_dir(lib_search_path) {
                Ok(files) => files,
                Err(..) => return,
            };
            let files = files.filter_map(|p| p.ok().map(|s| s.path()))
                             .collect::<Vec<_>>();
            fn is_rlib(p: &Path) -> bool {
                p.extension() == Some("rlib".as_ref())
            }
            // Reading metadata out of rlibs is faster, and if we find both
            // an rlib and a dylib we only read one of the files of
            // metadata, so in the name of speed, bring all rlib files to
            // the front of the search list.
            let files1 = files.iter().filter(|p| is_rlib(p));
            let files2 = files.iter().filter(|p| !is_rlib(p));
            for path in files1.chain(files2) {
                debug!("testing {}", path.display());
                let maybe_picked = pick(path, kind);
                match maybe_picked {
                    FileMatches => {
                        debug!("picked {}", path.display());
                    }
                    FileDoesntMatch => {
                        debug!("rejected {}", path.display());
                    }
                }
            }
        });
    }

    pub fn new(sysroot: &'a Path,
               triple: &'a str,
               search_paths: &'a SearchPaths,
               kind: PathKind) -> FileSearch<'a> {
        debug!("using sysroot = {}, triple = {}", sysroot.display(), triple);
        FileSearch {
            sysroot: sysroot,
            search_paths: search_paths,
            triple: triple,
            kind: kind,
        }
    }

    // Returns a list of directories where target-specific dylibs might be located.
    pub fn get_dylib_search_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        self.for_each_lib_search_path(|lib_search_path, _| {
            paths.push(lib_search_path.to_path_buf());
        });
        paths
    }

    // Returns a list of directories where target-specific tool binaries are located.
    pub fn get_tools_search_paths(&self) -> Vec<PathBuf> {
        let mut p = PathBuf::from(self.sysroot);
        p.push(find_libdir(self.sysroot).as_ref());
        p.push(RUST_LIB_DIR);
        p.push(&self.triple);
        p.push("bin");
        vec![p]
    }
}

pub fn relative_target_lib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let mut p = PathBuf::from(find_libdir(sysroot).as_ref());
    assert!(p.is_relative());
    p.push(RUST_LIB_DIR);
    p.push(target_triple);
    p.push("lib");
    p
}

fn make_target_lib_path(sysroot: &Path,
                        target_triple: &str) -> PathBuf {
    sysroot.join(&relative_target_lib_path(sysroot, target_triple))
}

pub fn get_or_default_sysroot() -> PathBuf {
    // Follow symlinks.  If the resolved path is relative, make it absolute.
    fn canonicalize(path: Option<PathBuf>) -> Option<PathBuf> {
        path.and_then(|path| {
            match fs::canonicalize(&path) {
                // See comments on this target function, but the gist is that
                // gcc chokes on verbatim paths which fs::canonicalize generates
                // so we try to avoid those kinds of paths.
                Ok(canon) => Some(rustcfs::fix_windows_verbatim_for_gcc(&canon)),
                Err(e) => bug!("failed to get realpath: {}", e),
            }
        })
    }

    match canonicalize(env::current_exe().ok()) {
        Some(mut p) => { p.pop(); p.pop(); p }
        None => bug!("can't determine value for sysroot")
    }
}

// The name of the directory rustc expects libraries to be located.
fn find_libdir(sysroot: &Path) -> Cow<'static, str> {
    // FIXME: This is a quick hack to make the rustc binary able to locate
    // Rust libraries in Linux environments where libraries might be installed
    // to lib64/lib32. This would be more foolproof by basing the sysroot off
    // of the directory where librustc is located, rather than where the rustc
    // binary is.
    //If --libdir is set during configuration to the value other than
    // "lib" (i.e. non-default), this value is used (see issue #16552).

    match option_env!("CFG_LIBDIR_RELATIVE") {
        Some(libdir) if libdir != "lib" => return libdir.into(),
        _ => if sysroot.join(PRIMARY_LIB_DIR).join(RUST_LIB_DIR).exists() {
            return PRIMARY_LIB_DIR.into();
        } else {
            return SECONDARY_LIB_DIR.into();
        }
    }

    #[cfg(target_pointer_width = "64")]
    const PRIMARY_LIB_DIR: &'static str = "lib64";

    #[cfg(target_pointer_width = "32")]
    const PRIMARY_LIB_DIR: &'static str = "lib32";

    const SECONDARY_LIB_DIR: &'static str = "lib";
}

// The name of rustc's own place to organize libraries.
// Used to be "rustc", now the default is "rustlib"
const RUST_LIB_DIR: &'static str = "rustlib";
