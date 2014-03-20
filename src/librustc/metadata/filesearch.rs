// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(non_camel_case_types)];

use std::cell::RefCell;
use std::os;
use std::io::fs;
use collections::HashSet;

pub enum FileMatch { FileMatches, FileDoesntMatch }

// A module for searching for libraries
// FIXME (#2658): I'm not happy how this module turned out. Should
// probably just be folded into cstore.

/// Functions with type `pick` take a parent directory as well as
/// a file found in that directory.
pub type pick<'a> = 'a |path: &Path| -> FileMatch;

pub struct FileSearch<'a> {
    sysroot: &'a Path,
    addl_lib_search_paths: &'a RefCell<HashSet<Path>>,
    target_triple: &'a str
}

impl<'a> FileSearch<'a> {
    pub fn for_each_lib_search_path(&self, f: |&Path| -> FileMatch) {
        let mut visited_dirs = HashSet::new();
        let mut found = false;

        let addl_lib_search_paths = self.addl_lib_search_paths.borrow();
        debug!("filesearch: searching additional lib search paths [{:?}]",
               addl_lib_search_paths.get().len());
        for path in addl_lib_search_paths.get().iter() {
            match f(path) {
                FileMatches => found = true,
                FileDoesntMatch => ()
            }
            visited_dirs.insert(path.as_vec().to_owned());
        }

        debug!("filesearch: searching target lib path");
        let tlib_path = make_target_lib_path(self.sysroot,
                                    self.target_triple);
        if !visited_dirs.contains_equiv(&tlib_path.as_vec()) {
            match f(&tlib_path) {
                FileMatches => found = true,
                FileDoesntMatch => ()
            }
        }
        visited_dirs.insert(tlib_path.as_vec().to_owned());
        // Try RUST_PATH
        if !found {
            let rustpath = rust_path();
            for path in rustpath.iter() {
                let tlib_path = make_rustpkg_target_lib_path(path, self.target_triple);
                debug!("is {} in visited_dirs? {:?}", tlib_path.display(),
                        visited_dirs.contains_equiv(&tlib_path.as_vec().to_owned()));

                if !visited_dirs.contains_equiv(&tlib_path.as_vec()) {
                    visited_dirs.insert(tlib_path.as_vec().to_owned());
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

    pub fn get_target_lib_path(&self) -> Path {
        make_target_lib_path(self.sysroot, self.target_triple)
    }

    pub fn get_target_lib_file_path(&self, file: &Path) -> Path {
        let mut p = self.get_target_lib_path();
        p.push(file);
        p
    }

    pub fn search(&self, pick: pick) {
        self.for_each_lib_search_path(|lib_search_path| {
            debug!("searching {}", lib_search_path.display());
            match fs::readdir(lib_search_path) {
                Ok(files) => {
                    let mut rslt = FileDoesntMatch;
                    let is_rlib = |p: & &Path| {
                        p.extension_str() == Some("rlib")
                    };
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
               target_triple: &'a str,
               addl_lib_search_paths: &'a RefCell<HashSet<Path>>) -> FileSearch<'a> {
        debug!("using sysroot = {}", sysroot.display());
        FileSearch {
            sysroot: sysroot,
            addl_lib_search_paths: addl_lib_search_paths,
            target_triple: target_triple
        }
    }
}

pub fn relative_target_lib_path(target_triple: &str) -> Path {
    let mut p = Path::new(libdir());
    assert!(p.is_relative());
    p.push(rustlibdir());
    p.push(target_triple);
    p.push("lib");
    p
}

fn make_target_lib_path(sysroot: &Path,
                        target_triple: &str) -> Path {
    sysroot.join(&relative_target_lib_path(target_triple))
}

fn make_rustpkg_target_lib_path(dir: &Path,
                        target_triple: &str) -> Path {
    let mut p = dir.join(libdir());
    p.push(target_triple);
    p
}

pub fn get_or_default_sysroot() -> Path {
    // Follow symlinks.  If the resolved path is relative, make it absolute.
    fn canonicalize(path: Option<Path>) -> Option<Path> {
        path.and_then(|mut path|
            match fs::readlink(&path) {
                Ok(canon) => {
                    if canon.is_absolute() {
                        Some(canon)
                    } else {
                        path.pop();
                        Some(path.join(canon))
                    }
                },
                Err(..) => Some(path),
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
pub fn get_rust_path() -> Option<~str> {
    os::getenv("RUST_PATH")
}

/// Returns the value of RUST_PATH, as a list
/// of Paths. Includes default entries for, if they exist:
/// $HOME/.rust
/// DIR/.rust for any DIR that's the current working directory
/// or an ancestor of it
pub fn rust_path() -> Vec<Path> {
    let mut env_rust_path: Vec<Path> = match get_rust_path() {
        Some(env_path) => {
            let env_path_components: Vec<&str> =
                env_path.split_str(PATH_ENTRY_SEPARATOR).collect();
            env_path_components.map(|&s| Path::new(s))
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
        if { let f = cwd.filename(); f.is_none() || f.unwrap() == bytes!("..") } {
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
pub fn libdir() -> ~str {
    (env!("CFG_LIBDIR_RELATIVE")).to_owned()
}

// The name of rustc's own place to organize libraries.
// Used to be "rustc", now the default is "rustlib"
pub fn rustlibdir() -> ~str {
    (env!("CFG_RUSTLIBDIR")).to_owned()
}
