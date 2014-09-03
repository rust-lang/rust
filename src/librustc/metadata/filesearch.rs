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

use std::cell::RefCell;
use std::os;
use std::io::fs;
use std::collections::HashSet;

use util::fs as myfs;

pub enum FileMatch { FileMatches, FileDoesntMatch }

// A module for searching for libraries
// FIXME (#2658): I'm not happy how this module turned out. Should
// probably just be folded into cstore.

/// Functions with type `pick` take a parent directory as well as
/// a file found in that directory.
pub type pick<'a> = |path: &Path|: 'a -> FileMatch;

pub struct FileSearch<'a> {
    pub sysroot: &'a Path,
    pub addl_lib_search_paths: &'a RefCell<Vec<Path>>,
    pub triple: &'a str,
}

impl<'a> FileSearch<'a> {
    pub fn for_each_lib_search_path(&self, f: |&Path| -> FileMatch) {
        let mut visited_dirs = HashSet::new();
        let mut found = false;

        debug!("filesearch: searching additional lib search paths [{:?}]",
               self.addl_lib_search_paths.borrow().len());
        for path in self.addl_lib_search_paths.borrow().iter() {
            match f(path) {
                FileMatches => found = true,
                FileDoesntMatch => ()
            }
            visited_dirs.insert(path.as_vec().to_vec());
        }

        debug!("filesearch: searching lib path");
        let tlib_path = make_target_lib_path(self.sysroot,
                                    self.triple);
        if !visited_dirs.contains_equiv(&tlib_path.as_vec()) {
            match f(&tlib_path) {
                FileMatches => found = true,
                FileDoesntMatch => ()
            }
        }

        visited_dirs.insert(tlib_path.as_vec().to_vec());
        // Try RUST_PATH
        if !found {
            let rustpath = rust_path();
            for path in rustpath.iter() {
                let tlib_path = make_rustpkg_lib_path(
                    self.sysroot, path, self.triple);
                debug!("is {} in visited_dirs? {:?}", tlib_path.display(),
                        visited_dirs.contains_equiv(&tlib_path.as_vec().to_vec()));

                if !visited_dirs.contains_equiv(&tlib_path.as_vec()) {
                    visited_dirs.insert(tlib_path.as_vec().to_vec());
                    // Don't keep searching the RUST_PATH if one match turns up --
                    // if we did, we'd get a "multiple matching crates" error
                    match f(&tlib_path) {
                       FileMatches => {
                           break;
                       }
                       FileDoesntMatch => ()
                    }
                }
            }
        }
    }

    pub fn get_lib_path(&self) -> Path {
        make_target_lib_path(self.sysroot, self.triple)
    }

    pub fn search(&self, pick: pick) {
        self.for_each_lib_search_path(|lib_search_path| {
            debug!("searching {}", lib_search_path.display());
            match fs::readdir(lib_search_path) {
                Ok(files) => {
                    let mut rslt = FileDoesntMatch;
                    fn is_rlib(p: & &Path) -> bool {
                        p.extension_str() == Some("rlib")
                    }
                    // Reading metadata out of rlibs is faster, and if we find both
                    // an rlib and a dylib we only read one of the files of
                    // metadata, so in the name of speed, bring all rlib files to
                    // the front of the search list.
                    let files1 = files.iter().filter(|p| is_rlib(p));
                    let files2 = files.iter().filter(|p| !is_rlib(p));
                    for path in files1.chain(files2) {
                        debug!("testing {}", path.display());
                        let maybe_picked = pick(path);
                        match maybe_picked {
                            FileMatches => {
                                debug!("picked {}", path.display());
                                rslt = FileMatches;
                            }
                            FileDoesntMatch => {
                                debug!("rejected {}", path.display());
                            }
                        }
                    }
                    rslt
                }
                Err(..) => FileDoesntMatch,
            }
        });
    }

    pub fn new(sysroot: &'a Path,
               triple: &'a str,
               addl_lib_search_paths: &'a RefCell<Vec<Path>>) -> FileSearch<'a> {
        debug!("using sysroot = {}, triple = {}", sysroot.display(), triple);
        FileSearch {
            sysroot: sysroot,
            addl_lib_search_paths: addl_lib_search_paths,
            triple: triple,
        }
    }

    // Returns a list of directories where target-specific dylibs might be located.
    pub fn get_dylib_search_paths(&self) -> Vec<Path> {
        let mut paths = Vec::new();
        self.for_each_lib_search_path(|lib_search_path| {
            paths.push(lib_search_path.clone());
            FileDoesntMatch
        });
        paths
    }

    // Returns a list of directories where target-specific tool binaries are located.
    pub fn get_tools_search_paths(&self) -> Vec<Path> {
        let mut p = Path::new(self.sysroot);
        p.push(find_libdir(self.sysroot));
        p.push(rustlibdir());
        p.push(self.triple);
        p.push("bin");
        vec![p]
    }
}

pub fn relative_target_lib_path(sysroot: &Path, target_triple: &str) -> Path {
    let mut p = Path::new(find_libdir(sysroot));
    assert!(p.is_relative());
    p.push(rustlibdir());
    p.push(target_triple);
    p.push("lib");
    p
}

fn make_target_lib_path(sysroot: &Path,
                        target_triple: &str) -> Path {
    sysroot.join(&relative_target_lib_path(sysroot, target_triple))
}

fn make_rustpkg_lib_path(sysroot: &Path,
                         dir: &Path,
                         triple: &str) -> Path {
    let mut p = dir.join(find_libdir(sysroot));
    p.push(triple);
    p
}

pub fn get_or_default_sysroot() -> Path {
    // Follow symlinks.  If the resolved path is relative, make it absolute.
    fn canonicalize(path: Option<Path>) -> Option<Path> {
        path.and_then(|path|
            match myfs::realpath(&path) {
                Ok(canon) => Some(canon),
                Err(e) => fail!("failed to get realpath: {}", e),
            })
    }

    match canonicalize(os::self_exe_name()) {
        Some(mut p) => { p.pop(); p.pop(); p }
        None => fail!("can't determine value for sysroot")
    }
}

#[cfg(windows)]
static PATH_ENTRY_SEPARATOR: &'static str = ";";
#[cfg(not(windows))]
static PATH_ENTRY_SEPARATOR: &'static str = ":";

/// Returns RUST_PATH as a string, without default paths added
pub fn get_rust_path() -> Option<String> {
    os::getenv("RUST_PATH").map(|x| x.to_string())
}

/// Returns the value of RUST_PATH, as a list
/// of Paths. Includes default entries for, if they exist:
/// $HOME/.rust
/// DIR/.rust for any DIR that's the current working directory
/// or an ancestor of it
pub fn rust_path() -> Vec<Path> {
    let mut env_rust_path: Vec<Path> = match get_rust_path() {
        Some(env_path) => {
            let env_path_components =
                env_path.as_slice().split_str(PATH_ENTRY_SEPARATOR);
            env_path_components.map(|s| Path::new(s)).collect()
        }
        None => Vec::new()
    };
    let mut cwd = os::getcwd();
    // now add in default entries
    let cwd_dot_rust = cwd.join(".rust");
    if !env_rust_path.contains(&cwd_dot_rust) {
        env_rust_path.push(cwd_dot_rust);
    }
    if !env_rust_path.contains(&cwd) {
        env_rust_path.push(cwd.clone());
    }
    loop {
        if { let f = cwd.filename(); f.is_none() || f.unwrap() == b".." } {
            break
        }
        cwd.set_filename(".rust");
        if !env_rust_path.contains(&cwd) && cwd.exists() {
            env_rust_path.push(cwd.clone());
        }
        cwd.pop();
    }
    let h = os::homedir();
    for h in h.iter() {
        let p = h.join(".rust");
        if !env_rust_path.contains(&p) && p.exists() {
            env_rust_path.push(p);
        }
    }
    env_rust_path
}

// The name of the directory rustc expects libraries to be located.
// On Unix should be "lib", on windows "bin"
#[cfg(unix)]
fn find_libdir(sysroot: &Path) -> String {
    // FIXME: This is a quick hack to make the rustc binary able to locate
    // Rust libraries in Linux environments where libraries might be installed
    // to lib64/lib32. This would be more foolproof by basing the sysroot off
    // of the directory where librustc is located, rather than where the rustc
    // binary is.

    if sysroot.join(primary_libdir_name()).join(rustlibdir()).exists() {
        return primary_libdir_name();
    } else {
        return secondary_libdir_name();
    }

    #[cfg(target_word_size = "64")]
    fn primary_libdir_name() -> String {
        "lib64".to_string()
    }

    #[cfg(target_word_size = "32")]
    fn primary_libdir_name() -> String {
        "lib32".to_string()
    }

    fn secondary_libdir_name() -> String {
        "lib".to_string()
    }
}

#[cfg(windows)]
fn find_libdir(_sysroot: &Path) -> String {
    "bin".to_string()
}

// The name of rustc's own place to organize libraries.
// Used to be "rustc", now the default is "rustlib"
pub fn rustlibdir() -> String {
    "rustlib".to_string()
}
