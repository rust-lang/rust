#![allow(non_camel_case_types)]

pub use self::FileMatch::*;

use std::borrow::Cow;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::session::search_paths::{SearchPath, PathKind};
use rustc_fs_util::fix_windows_verbatim_for_gcc;

#[derive(Copy, Clone)]
pub enum FileMatch {
    FileMatches,
    FileDoesntMatch,
}

// A module for searching for libraries

#[derive(Clone)]
pub struct FileSearch<'a> {
    sysroot: &'a Path,
    triple: &'a str,
    search_paths: &'a [SearchPath],
    tlib_path: &'a SearchPath,
    kind: PathKind,
}

impl<'a> FileSearch<'a> {
    pub fn search_paths(&self) -> impl Iterator<Item = &'a SearchPath> {
        let kind = self.kind;
        self.search_paths.iter()
            .filter(move |sp| sp.kind.matches(kind))
            .chain(std::iter::once(self.tlib_path))
    }

    pub fn get_lib_path(&self) -> PathBuf {
        make_target_lib_path(self.sysroot, self.triple)
    }

    pub fn search<F>(&self, mut pick: F)
        where F: FnMut(&Path, PathKind) -> FileMatch
    {
        for search_path in self.search_paths() {
            debug!("searching {}", search_path.dir.display());
            fn is_rlib(p: &Path) -> bool {
                p.extension() == Some("rlib".as_ref())
            }
            // Reading metadata out of rlibs is faster, and if we find both
            // an rlib and a dylib we only read one of the files of
            // metadata, so in the name of speed, bring all rlib files to
            // the front of the search list.
            let files1 = search_path.files.iter().filter(|p| is_rlib(p));
            let files2 = search_path.files.iter().filter(|p| !is_rlib(p));
            for path in files1.chain(files2) {
                debug!("testing {}", path.display());
                let maybe_picked = pick(path, search_path.kind);
                match maybe_picked {
                    FileMatches => {
                        debug!("picked {}", path.display());
                    }
                    FileDoesntMatch => {
                        debug!("rejected {}", path.display());
                    }
                }
            }
        }
    }

    pub fn new(sysroot: &'a Path,
               triple: &'a str,
               search_paths: &'a Vec<SearchPath>,
               tlib_path: &'a SearchPath,
               kind: PathKind)
               -> FileSearch<'a> {
        debug!("using sysroot = {}, triple = {}", sysroot.display(), triple);
        FileSearch {
            sysroot,
            triple,
            search_paths,
            tlib_path,
            kind,
        }
    }

    // Returns just the directories within the search paths.
    pub fn search_path_dirs(&self) -> Vec<PathBuf> {
        self.search_paths()
            .map(|sp| sp.dir.to_path_buf())
            .collect()
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

pub fn make_target_lib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
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
                Ok(canon) => Some(fix_windows_verbatim_for_gcc(&canon)),
                Err(e) => bug!("failed to get realpath: {}", e),
            }
        })
    }

    match env::current_exe() {
        Ok(exe) => {
            match canonicalize(Some(exe)) {
                Some(mut p) => { p.pop(); p.pop(); p },
                None => bug!("can't determine value for sysroot")
            }
        }
        Err(ref e) => panic!(format!("failed to get current_exe: {}", e))
    }
}

// The name of the directory rustc expects libraries to be located.
fn find_libdir(sysroot: &Path) -> Cow<'static, str> {
    // FIXME: This is a quick hack to make the rustc binary able to locate
    // Rust libraries in Linux environments where libraries might be installed
    // to lib64/lib32. This would be more foolproof by basing the sysroot off
    // of the directory where librustc is located, rather than where the rustc
    // binary is.
    // If --libdir is set during configuration to the value other than
    // "lib" (i.e., non-default), this value is used (see issue #16552).

    #[cfg(target_pointer_width = "64")]
    const PRIMARY_LIB_DIR: &str = "lib64";

    #[cfg(target_pointer_width = "32")]
    const PRIMARY_LIB_DIR: &str = "lib32";

    const SECONDARY_LIB_DIR: &str = "lib";

    match option_env!("CFG_LIBDIR_RELATIVE") {
        Some(libdir) if libdir != "lib" => libdir.into(),
        _ => if sysroot.join(PRIMARY_LIB_DIR).join(RUST_LIB_DIR).exists() {
            PRIMARY_LIB_DIR.into()
        } else {
            SECONDARY_LIB_DIR.into()
        }
    }
}

// The name of rustc's own place to organize libraries.
// Used to be "rustc", now the default is "rustlib"
const RUST_LIB_DIR: &str = "rustlib";
