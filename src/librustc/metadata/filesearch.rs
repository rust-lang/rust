// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::option;
use std::os;
use std::hashmap::HashSet;

// A module for searching for libraries
// FIXME (#2658): I'm not happy how this module turned out. Should
// probably just be folded into cstore.

/// Functions with type `pick` take a parent directory as well as
/// a file found in that directory.
pub type pick<'self, T> = &'self fn(path: &Path) -> Option<T>;

pub fn pick_file(file: Path, path: &Path) -> Option<Path> {
    if path.file_path() == file {
        option::Some((*path).clone())
    } else {
        option::None
    }
}

pub trait FileSearch {
    fn sysroot(&self) -> @Path;
    fn for_each_lib_search_path(&self, f: &fn(&Path) -> bool) -> bool;
    fn get_target_lib_path(&self) -> Path;
    fn get_target_lib_file_path(&self, file: &Path) -> Path;
}

pub fn mk_filesearch(maybe_sysroot: &Option<@Path>,
                     target_triple: &str,
                     addl_lib_search_paths: @mut ~[Path])
                  -> @FileSearch {
    struct FileSearchImpl {
        sysroot: @Path,
        addl_lib_search_paths: @mut ~[Path],
        target_triple: ~str
    }
    impl FileSearch for FileSearchImpl {
        fn sysroot(&self) -> @Path { self.sysroot }
        fn for_each_lib_search_path(&self, f: &fn(&Path) -> bool) -> bool {
            let mut visited_dirs = HashSet::new();

            debug!("filesearch: searching additional lib search paths [%?]",
                   self.addl_lib_search_paths.len());
            for path in self.addl_lib_search_paths.iter() {
                f(path);
                visited_dirs.insert(path.to_str());
            }

            debug!("filesearch: searching target lib path");
            let tlib_path = make_target_lib_path(self.sysroot,
                                        self.target_triple);
            if !visited_dirs.contains(&tlib_path.to_str()) {
                if !f(&tlib_path) {
                    return false;
                }
            }
            visited_dirs.insert(tlib_path.to_str());
            // Try RUST_PATH
            let rustpath = rust_path();
            for path in rustpath.iter() {
                    if !visited_dirs.contains(&path.push("lib").to_str()) {
                        f(&path.push("lib"));
                        visited_dirs.insert(path.push("lib").to_str());
                    }
            }
            true
        }
        fn get_target_lib_path(&self) -> Path {
            make_target_lib_path(self.sysroot, self.target_triple)
        }
        fn get_target_lib_file_path(&self, file: &Path) -> Path {
            self.get_target_lib_path().push_rel(file)
        }
    }

    let sysroot = get_sysroot(maybe_sysroot);
    debug!("using sysroot = %s", sysroot.to_str());
    @FileSearchImpl {
        sysroot: sysroot,
        addl_lib_search_paths: addl_lib_search_paths,
        target_triple: target_triple.to_owned()
    } as @FileSearch
}

pub fn search<T>(filesearch: @FileSearch, pick: pick<T>) -> Option<T> {
    let mut rslt = None;
    do filesearch.for_each_lib_search_path() |lib_search_path| {
        debug!("searching %s", lib_search_path.to_str());
        let r = os::list_dir_path(lib_search_path);
        for path in r.iter() {
            debug!("testing %s", path.to_str());
            let maybe_picked = pick(path);
            match maybe_picked {
                Some(_) => {
                    debug!("picked %s", path.to_str());
                    rslt = maybe_picked;
                    break;
                }
                None => {
                    debug!("rejected %s", path.to_str());
                }
            }
        }
        rslt.is_none()
    };
    return rslt;
}

pub fn relative_target_lib_path(target_triple: &str) -> Path {
    Path(libdir()).push_many([~"rustc",
                              target_triple.to_owned(),
                              libdir()])
}

fn make_target_lib_path(sysroot: &Path,
                        target_triple: &str) -> Path {
    sysroot.push_rel(&relative_target_lib_path(target_triple))
}

fn get_or_default_sysroot() -> Path {
    match os::self_exe_path() {
      option::Some(ref p) => (*p).pop(),
      option::None => fail!("can't determine value for sysroot")
    }
}

fn get_sysroot(maybe_sysroot: &Option<@Path>) -> @Path {
    match *maybe_sysroot {
      option::Some(sr) => sr,
      option::None => @get_or_default_sysroot()
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
pub fn rust_path() -> ~[Path] {
    let mut env_rust_path: ~[Path] = match get_rust_path() {
        Some(env_path) => {
            let env_path_components: ~[&str] =
                env_path.split_str_iter(PATH_ENTRY_SEPARATOR).collect();
            env_path_components.map(|&s| Path(s))
        }
        None => ~[]
    };
    let cwd = os::getcwd();
    // now add in default entries
    let cwd_dot_rust = cwd.push(".rust");
    if !env_rust_path.contains(&cwd_dot_rust) {
        env_rust_path.push(cwd_dot_rust);
    }
    if !env_rust_path.contains(&cwd) {
        env_rust_path.push(cwd.clone());
    }
    do cwd.each_parent() |p| {
        if !env_rust_path.contains(&p.push(".rust")) {
            push_if_exists(&mut env_rust_path, p);
        }
    }
    let h = os::homedir();
    for h in h.iter() {
        if !env_rust_path.contains(&h.push(".rust")) {
            push_if_exists(&mut env_rust_path, h);
        }
    }
    env_rust_path
}


/// Adds p/.rust into vec, only if it exists
fn push_if_exists(vec: &mut ~[Path], p: &Path) {
    let maybe_dir = p.push(".rust");
    if os::path_exists(&maybe_dir) {
        vec.push(maybe_dir);
    }
}

// The name of the directory rustc expects libraries to be located.
// On Unix should be "lib", on windows "bin"
#[cfg(stage0)]
pub fn libdir() -> ~str {
   let libdir = env!("CFG_LIBDIR");
   if libdir.is_empty() {
      fail!("rustc compiled without CFG_LIBDIR environment variable");
   }
   libdir.to_owned()
}

#[cfg(not(stage0))]
pub fn libdir() -> ~str {
    (env!("CFG_LIBDIR")).to_owned()
}
