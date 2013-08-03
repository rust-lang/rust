// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use std::result;

// A module for searching for libraries
// FIXME (#2658): I'm not happy how this module turned out. Should
// probably just be folded into cstore.

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
            debug!("filesearch: searching additional lib search paths [%?]",
                   self.addl_lib_search_paths.len());
            // a little weird
            self.addl_lib_search_paths.iter().advance(|path| f(path));

            debug!("filesearch: searching target lib path");
            if !f(&make_target_lib_path(self.sysroot,
                                        self.target_triple)) {
                return false;
            }
            debug!("filesearch: searching rustpkg lib path nearest");
            if match get_rustpkg_lib_path_nearest() {
                    result::Ok(ref p) => f(p),
                    result::Err(_) => true
                } {
                    return true;
                }
           debug!("filesearch: searching rustpkg lib path");
           match get_rustpkg_lib_path() {
              result::Ok(ref p) => f(p),
              result::Err(_) => true
           }
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
        foreach path in r.iter() {
            debug!("testing %s", path.to_str());
            let maybe_picked = pick(path);
            if maybe_picked.is_some() {
                debug!("picked %s", path.to_str());
                rslt = maybe_picked;
                break;
            } else {
                debug!("rejected %s", path.to_str());
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

pub fn get_rustpkg_sysroot() -> Result<Path, ~str> {
    result::Ok(get_or_default_sysroot().push_many([libdir(), ~"rustpkg"]))
}

pub fn get_rustpkg_root() -> Result<Path, ~str> {
    match os::getenv("RUSTPKG_ROOT") {
        Some(ref _p) => result::Ok(Path((*_p))),
        None => match os::homedir() {
          Some(ref _q) => result::Ok((*_q).push(".rustpkg")),
          None => result::Err(~"no RUSTPKG_ROOT or home directory")
        }
    }
}

pub fn get_rustpkg_root_nearest() -> Result<Path, ~str> {
    do get_rustpkg_root().chain |p| {
        let cwd = os::getcwd();
        let cwd_rustpkg = cwd.push(".rustpkg");
        let rustpkg_is_non_root_file =
            !os::path_is_dir(&cwd_rustpkg) && cwd_rustpkg != p;
        let mut par_rustpkg = cwd.pop().push(".rustpkg");
        let mut rslt = result::Ok(cwd_rustpkg);

        if rustpkg_is_non_root_file {
            while par_rustpkg != p {
                if os::path_is_dir(&par_rustpkg) {
                    rslt = result::Ok(par_rustpkg);
                    break;
                }
                if par_rustpkg.components.len() == 1 {
                    // We just checked /.rustpkg, stop now.
                    break;
                }
                par_rustpkg = par_rustpkg.pop().pop().push(".rustpkg");
            }
        }
        rslt
    }
}

fn get_rustpkg_lib_path() -> Result<Path, ~str> {
    do get_rustpkg_root().chain |p| {
        result::Ok(p.push(libdir()))
    }
}

fn get_rustpkg_lib_path_nearest() -> Result<Path, ~str> {
    do get_rustpkg_root_nearest().chain |p| {
        result::Ok(p.push(libdir()))
    }
}

// The name of the directory rustc expects libraries to be located.
// On Unix should be "lib", on windows "bin"
pub fn libdir() -> ~str {
   let libdir = env!("CFG_LIBDIR");
   if libdir.is_empty() {
      fail!("rustc compiled without CFG_LIBDIR environment variable");
   }
   libdir.to_owned()
}
