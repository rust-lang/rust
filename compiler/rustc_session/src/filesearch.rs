//! A module for searching for libraries

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{env, fs, iter};

use rustc_fs_util::try_canonicalize;
use rustc_target::spec::Target;

use crate::search_paths::{PathKind, SearchPath};

pub struct FileSearch {
    cli_search_paths: Vec<SearchPath>,
    tlib_path: SearchPath,
    use_implicit_sysroot_deps: bool,
    files: Vec<FileSearchCandidate>,
}

impl FileSearch {
    pub fn cli_search_paths<'b>(&'b self, kind: PathKind) -> impl Iterator<Item = &'b SearchPath> {
        self.cli_search_paths.iter().filter(move |sp| sp.kind.matches(kind))
    }

    pub fn search_paths<'b>(&'b self, kind: PathKind) -> impl Iterator<Item = &'b SearchPath> {
        // If the crate is `PathKind::Crate` (a top level dependency)
        // and `-Z implicit-sysroot-deps=false`, then don't include the sysroot in the search paths.
        let exclude_sysroot = kind.matches(PathKind::Crate) && !self.use_implicit_sysroot_deps;
        let maybe_tlib = (!exclude_sysroot).then_some(&self.tlib_path);

        self.cli_search_paths
            .iter()
            .filter(move |sp| sp.kind.matches(kind))
            .chain(maybe_tlib.into_iter())
    }

    /// Return files from the search dirs of this filesearch that match the given `prefix` and
    /// `suffix` and have the given `kind`.
    ///
    /// Note that this function only searches files that match lib/staticlib/dlllib prefixes, not
    /// all files from the search paths!
    /// Access `search_paths` directly if you want to scan all files within them.
    pub fn get_library_candidates<'b>(
        &'b self,
        prefix: &'b str,
        suffix: &'b str,
        kind: PathKind,
    ) -> impl Iterator<Item = (&'b str, PathBuf)> {
        let exclude_sysroot = kind.matches(PathKind::Crate) && !self.use_implicit_sysroot_deps;

        // The indices are clipped to have only a single iterator returned from this function, to
        // avoid allocating it.
        let start = self.files.partition_point(|v| *v.filename < *prefix).min(self.files.len());
        let end = self.files[start..].partition_point(|v| v.filename.starts_with(prefix));
        let prefixed_items = &self.files[start..][..end];

        prefixed_items
            .into_iter()
            .filter(move |c| {
                c.kind.matches(kind)
                    && !(exclude_sysroot && c.from_sysroot)
                    && c.filename.ends_with(suffix)
            })
            .map(|c| (&c.filename[prefix.len()..c.filename.len() - suffix.len()], c.path()))
    }

    pub fn new(
        cli_search_paths: &[SearchPath],
        tlib_path: &SearchPath,
        target: &Target,
        use_implicit_sysroot_deps: bool,
    ) -> Self {
        // We keep a list of all found paths that look like libraries in `FileSearch`, to optimize
        // lookup in `get_library_candidates`.
        // These prefixes should be kept in sync with `CrateLocator::find_library_crate`.
        let prefixes = ["lib", &target.staticlib_prefix, &target.dll_prefix];

        // Load all files from all search paths, filter them by supported prefixes, and sort them,
        // so that we can efficiently look them up in `get_file_candidates` via binary search.
        let mut files: Vec<FileSearchCandidate> = Vec::with_capacity(cli_search_paths.len());
        for (search_path, is_sysroot) in
            cli_search_paths.iter().map(|path| (path, false)).chain(iter::once((tlib_path, true)))
        {
            let Ok(dir) = fs::read_dir(&search_path.dir) else {
                continue;
            };
            files.extend(dir.filter_map(|entry| {
                let entry = entry.ok()?;

                let filename = entry.file_name();
                let filename = filename.to_str()?;

                if !prefixes.iter().any(|prefix| filename.starts_with(prefix)) {
                    return None;
                }
                Some(FileSearchCandidate {
                    dir: Arc::clone(&search_path.dir),
                    filename: filename.into(),
                    kind: search_path.kind,
                    from_sysroot: is_sysroot,
                })
            }));
        }
        files.sort_unstable_by(|lhs, rhs| lhs.filename.cmp(&rhs.filename));

        FileSearch {
            cli_search_paths: cli_search_paths.to_owned(),
            tlib_path: tlib_path.clone(),
            use_implicit_sysroot_deps,
            files,
        }
    }
}

/// This type stores `Box<str>` instead of `PathBuf` for the filename, because getting the
/// `file_name` of a `PathBuf` allocates, which is unnecessary. We have to go through the files
/// a lot of times, so storing file name and the directory separately saves time and memory.
///
/// The filename must be valid UTF-8. If it's not, the entry should be skipped, because all Rust
/// output files are valid UTF-8, and so a non-UTF-8 filename couldn't be one we're looking for.
#[derive(Debug)]
struct FileSearchCandidate {
    dir: Arc<Path>,
    filename: Box<str>,
    kind: PathKind,
    /// Was this file added through the target sysroot?
    from_sysroot: bool,
}

impl FileSearchCandidate {
    /// Constructs the full path to the file.
    fn path(&self) -> PathBuf {
        self.dir.join(&*self.filename)
    }
}

pub fn make_target_lib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let rustlib_path = rustc_target::relative_target_rustlib_path(sysroot, target_triple);
    sysroot.join(rustlib_path).join("lib")
}

/// Returns a path to the target's `bin` folder within its `rustlib` path in the sysroot. This is
/// where binaries are usually installed, e.g. the self-contained linkers, lld-wrappers, LLVM tools,
/// etc.
pub fn make_target_bin_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let rustlib_path = rustc_target::relative_target_rustlib_path(sysroot, target_triple);
    sysroot.join(rustlib_path).join("bin")
}

/// Attempts to find the path to the dynamic library containing a function.
///
/// SAFETY: `function` must be a valid pointer to some function.
#[cfg(unix)]
pub unsafe fn dll_path(function: *mut std::ffi::c_void) -> Result<PathBuf, String> {
    use std::ffi::{CStr, OsStr};
    use std::os::unix::prelude::*;

    #[cfg(not(target_os = "aix"))]
    unsafe {
        let mut info = std::mem::zeroed();
        if libc::dladdr(function, &mut info) == 0 {
            return Err("dladdr failed".into());
        }
        #[cfg(target_os = "cygwin")]
        let fname_ptr = info.dli_fname.as_ptr();
        #[cfg(not(target_os = "cygwin"))]
        let fname_ptr = {
            assert!(!info.dli_fname.is_null(), "dli_fname cannot be null");
            info.dli_fname
        };
        let bytes = CStr::from_ptr(fname_ptr).to_bytes();
        let os = OsStr::from_bytes(bytes);
        try_canonicalize(Path::new(os)).map_err(|e| e.to_string())
    }

    #[cfg(target_os = "aix")]
    unsafe {
        // On AIX, the symbol references a function descriptor.
        // A function descriptor is consisted of (See https://reviews.llvm.org/D62532)
        // * The address of the entry point of the function.
        // * The TOC base address for the function.
        // * The environment pointer.
        // The function descriptor is in the data section.
        let addr = function as u64;
        let mut buffer = vec![std::mem::zeroed::<libc::ld_info>(); 64];
        loop {
            if libc::loadquery(
                libc::L_GETINFO,
                buffer.as_mut_ptr() as *mut libc::c_void,
                (size_of::<libc::ld_info>() * buffer.len()) as u32,
            ) >= 0
            {
                break;
            } else {
                if std::io::Error::last_os_error().raw_os_error().unwrap() != libc::ENOMEM {
                    return Err("loadquery failed".into());
                }
                buffer.resize(buffer.len() * 2, std::mem::zeroed::<libc::ld_info>());
            }
        }
        let mut current = buffer.as_mut_ptr() as *mut libc::ld_info;
        loop {
            let data_base = (*current).ldinfo_dataorg as u64;
            let data_end = data_base + (*current).ldinfo_datasize;
            if (data_base..data_end).contains(&addr) {
                let bytes = CStr::from_ptr(&(*current).ldinfo_filename[0]).to_bytes();
                let os = OsStr::from_bytes(bytes);
                return try_canonicalize(Path::new(os)).map_err(|e| e.to_string());
            }
            if (*current).ldinfo_next == 0 {
                break;
            }
            current =
                (current as *mut i8).offset((*current).ldinfo_next as isize) as *mut libc::ld_info;
        }
        return Err(format!("current dll's address {} is not in the load map", addr));
    }
}

#[cfg(windows)]
pub unsafe fn dll_path(function: *mut std::ffi::c_void) -> Result<PathBuf, String> {
    use std::ffi::OsString;
    use std::io;
    use std::os::windows::prelude::*;

    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GetModuleFileNameW, GetModuleHandleExW,
    };
    use windows::core::PCWSTR;

    let mut module = HMODULE::default();
    unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            PCWSTR(function as *mut u16),
            &mut module,
        )
    }
    .map_err(|e| e.to_string())?;

    let mut filename = vec![0; 1024];
    let n = unsafe { GetModuleFileNameW(Some(module), &mut filename) } as usize;
    if n == 0 {
        return Err(format!("GetModuleFileNameW failed: {}", io::Error::last_os_error()));
    }
    if n >= filename.capacity() {
        return Err(format!("our buffer was too small? {}", io::Error::last_os_error()));
    }

    filename.truncate(n);

    let path = try_canonicalize(OsString::from_wide(&filename)).map_err(|e| e.to_string())?;

    // See comments on this target function, but the gist is that
    // gcc chokes on verbatim paths which fs::canonicalize generates
    // so we try to avoid those kinds of paths.
    Ok(rustc_fs_util::fix_windows_verbatim_for_gcc(&path))
}

#[cfg(target_os = "wasi")]
pub unsafe fn dll_path(function: *mut std::ffi::c_void) -> Result<PathBuf, String> {
    Err("dll_path is not supported on WASI".to_string())
}

fn current_dll_path() -> Result<PathBuf, String> {
    use std::sync::OnceLock;

    // This is somewhat expensive relative to other work when compiling `fn main() {}` as `dladdr`
    // needs to iterate over the symbol table of librustc_driver.so until it finds a match.
    // As such cache this to avoid recomputing if we try to get the sysroot in multiple places.
    static CURRENT_DLL_PATH: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    CURRENT_DLL_PATH
        .get_or_init(|| unsafe { dll_path(current_dll_path as fn() -> _ as *mut _) })
        .clone()
}

/// This function checks if sysroot is found using env::args().next(), and if it
/// is not found, finds sysroot from current rustc_driver dll.
pub(crate) fn default_sysroot() -> PathBuf {
    fn default_from_rustc_driver_dll() -> Result<PathBuf, String> {
        let dll = current_dll_path()?;

        // `dll` will be in one of the following two:
        // - compiler's libdir: $sysroot/lib/*.dll
        // - target's libdir: $sysroot/lib/rustlib/$target/lib/*.dll
        //
        // use `parent` twice to chop off the file name and then also the
        // directory containing the dll
        let dir = dll.parent().and_then(|p| p.parent()).ok_or_else(|| {
            format!("Could not move 2 levels upper using `parent()` on {}", dll.display())
        })?;

        // if `dir` points to target's dir, move up to the sysroot
        let mut sysroot_dir = if dir.ends_with(crate::config::host_tuple()) {
            dir.parent() // chop off `$target`
                .and_then(|p| p.parent()) // chop off `rustlib`
                .and_then(|p| p.parent()) // chop off `lib`
                .map(|s| s.to_owned())
                .ok_or_else(|| {
                    format!("Could not move 3 levels upper using `parent()` on {}", dir.display())
                })?
        } else {
            dir.to_owned()
        };

        // On multiarch linux systems, there will be multiarch directory named
        // with the architecture(e.g `x86_64-linux-gnu`) under the `lib` directory.
        // Which cause us to mistakenly end up in the lib directory instead of the sysroot directory.
        if sysroot_dir.ends_with("lib") {
            sysroot_dir =
                sysroot_dir.parent().map(|real_sysroot| real_sysroot.to_owned()).ok_or_else(
                    || format!("Could not move to parent path of {}", sysroot_dir.display()),
                )?
        }

        Ok(sysroot_dir)
    }

    // Use env::args().next() to get the path of the executable without
    // following symlinks/canonicalizing any component. This makes the rustc
    // binary able to locate Rust libraries in systems using content-addressable
    // storage (CAS).
    fn from_env_args_next() -> Option<PathBuf> {
        let mut p = PathBuf::from(env::args_os().next()?);

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
        let mut rustlib_path = rustc_target::relative_target_rustlib_path(&p, "dummy");
        rustlib_path.pop(); // pop off the dummy target.
        rustlib_path.exists().then_some(p)
    }

    from_env_args_next()
        .unwrap_or_else(|| default_from_rustc_driver_dll().expect("Failed finding sysroot"))
}
