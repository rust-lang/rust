//! A module for searching for libraries

pub use self::FileMatch::*;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::search_paths::{PathKind, SearchPath, SearchPathFile};
use rustc_fs_util::fix_windows_verbatim_for_gcc;
use tracing::debug;

#[derive(Copy, Clone)]
pub enum FileMatch {
    FileMatches,
    FileDoesntMatch,
}

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
        self.search_paths
            .iter()
            .filter(move |sp| sp.kind.matches(kind))
            .chain(std::iter::once(self.tlib_path))
    }

    pub fn get_lib_path(&self) -> PathBuf {
        make_target_lib_path(self.sysroot, self.triple)
    }

    pub fn get_self_contained_lib_path(&self) -> PathBuf {
        self.get_lib_path().join("self-contained")
    }

    pub fn search<F>(&self, mut pick: F)
    where
        F: FnMut(&SearchPathFile, PathKind) -> FileMatch,
    {
        for search_path in self.search_paths() {
            debug!("searching {}", search_path.dir.display());
            fn is_rlib(spf: &SearchPathFile) -> bool {
                if let Some(f) = &spf.file_name_str { f.ends_with(".rlib") } else { false }
            }
            // Reading metadata out of rlibs is faster, and if we find both
            // an rlib and a dylib we only read one of the files of
            // metadata, so in the name of speed, bring all rlib files to
            // the front of the search list.
            let files1 = search_path.files.iter().filter(|spf| is_rlib(&spf));
            let files2 = search_path.files.iter().filter(|spf| !is_rlib(&spf));
            for spf in files1.chain(files2) {
                debug!("testing {}", spf.path.display());
                let maybe_picked = pick(spf, search_path.kind);
                match maybe_picked {
                    FileMatches => {
                        debug!("picked {}", spf.path.display());
                    }
                    FileDoesntMatch => {
                        debug!("rejected {}", spf.path.display());
                    }
                }
            }
        }
    }

    pub fn new(
        sysroot: &'a Path,
        triple: &'a str,
        search_paths: &'a [SearchPath],
        tlib_path: &'a SearchPath,
        kind: PathKind,
    ) -> FileSearch<'a> {
        debug!("using sysroot = {}, triple = {}", sysroot.display(), triple);
        FileSearch { sysroot, triple, search_paths, tlib_path, kind }
    }

    /// Returns just the directories within the search paths.
    pub fn search_path_dirs(&self) -> Vec<PathBuf> {
        self.search_paths().map(|sp| sp.dir.to_path_buf()).collect()
    }
}

pub fn make_target_lib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let rustlib_path = rustc_target::target_rustlib_path(sysroot, target_triple);
    std::array::IntoIter::new([sysroot, Path::new(&rustlib_path), Path::new("lib")])
        .collect::<PathBuf>()
}

/// This function checks if sysroot is found using env::args().next(), and if it
/// is not found, uses env::current_exe() to imply sysroot.
pub fn get_or_default_sysroot() -> PathBuf {
    // Follow symlinks.  If the resolved path is relative, make it absolute.
    fn canonicalize(path: PathBuf) -> PathBuf {
        let path = fs::canonicalize(&path).unwrap_or(path);
        // See comments on this target function, but the gist is that
        // gcc chokes on verbatim paths which fs::canonicalize generates
        // so we try to avoid those kinds of paths.
        fix_windows_verbatim_for_gcc(&path)
    }

    // Use env::current_exe() to get the path of the executable following
    // symlinks/canonicalizing components.
    fn from_current_exe() -> PathBuf {
        match env::current_exe() {
            Ok(exe) => {
                let mut p = canonicalize(exe);
                p.pop();
                p.pop();
                p
            }
            Err(e) => panic!("failed to get current_exe: {}", e),
        }
    }

    // Use env::args().next() to get the path of the executable without
    // following symlinks/canonicalizing any component. This makes the rustc
    // binary able to locate Rust libraries in systems using content-addressable
    // storage (CAS).
    fn from_env_args_next() -> Option<PathBuf> {
        match env::args_os().next() {
            Some(first_arg) => {
                let mut p = PathBuf::from(first_arg);

                // Check if sysroot is found using env::args().next() only if the rustc in argv[0]
                // is a symlink (see #79253). We might want to change/remove it to conform with
                // https://www.gnu.org/prep/standards/standards.html#Finding-Program-Files in the
                // future.
                if fs::read_link(&p).is_err() {
                    // Path is not a symbolic link or does not exist.
                    return None;
                }

                // Pop off `bin/rustc`, obtaining the suspected sysroot.
                p.pop();
                p.pop();
                // Look for the target rustlib directory in the suspected sysroot.
                let mut rustlib_path = rustc_target::target_rustlib_path(&p, "dummy");
                rustlib_path.pop(); // pop off the dummy target.
                if rustlib_path.exists() { Some(p) } else { None }
            }
            None => None,
        }
    }

    // Check if sysroot is found using env::args().next(), and if is not found,
    // use env::current_exe() to imply sysroot.
    from_env_args_next().unwrap_or_else(from_current_exe)
}
