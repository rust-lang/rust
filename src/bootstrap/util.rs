//! Various utility functions used throughout rustbuild.
//!
//! Simple things like testing the various filesystem operations here and there,
//! not a lot of interesting happenings here unfortunately.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str;
use std::time::Instant;

use build_helper::t;

use crate::builder::Builder;
use crate::config::{Config, TargetSelection};

/// Returns the `name` as the filename of a static library for `target`.
pub fn staticlib(name: &str, target: TargetSelection) -> String {
    if target.contains("windows") { format!("{}.lib", name) } else { format!("lib{}.a", name) }
}

/// Given an executable called `name`, return the filename for the
/// executable for a particular target.
pub fn exe(name: &str, target: TargetSelection) -> String {
    if target.contains("windows") { format!("{}.exe", name) } else { name.to_string() }
}

/// Returns `true` if the file name given looks like a dynamic library.
pub fn is_dylib(name: &str) -> bool {
    name.ends_with(".dylib") || name.ends_with(".so") || name.ends_with(".dll")
}

/// Returns `true` if the file name given looks like a debug info file
pub fn is_debug_info(name: &str) -> bool {
    // FIXME: consider split debug info on other platforms (e.g., Linux, macOS)
    name.ends_with(".pdb")
}

/// Returns the corresponding relative library directory that the compiler's
/// dylibs will be found in.
pub fn libdir(target: TargetSelection) -> &'static str {
    if target.contains("windows") { "bin" } else { "lib" }
}

/// Adds a list of lookup paths to `cmd`'s dynamic library lookup path.
/// If The dylib_path_par is already set for this cmd, the old value will be overwritten!
pub fn add_dylib_path(path: Vec<PathBuf>, cmd: &mut Command) {
    let mut list = dylib_path();
    for path in path {
        list.insert(0, path);
    }
    cmd.env(dylib_path_var(), t!(env::join_paths(list)));
}

/// Returns the environment variable which the dynamic library lookup path
/// resides in for this platform.
pub fn dylib_path_var() -> &'static str {
    if cfg!(target_os = "windows") {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "haiku") {
        "LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Parses the `dylib_path_var()` environment variable, returning a list of
/// paths that are members of this lookup path.
pub fn dylib_path() -> Vec<PathBuf> {
    let var = match env::var_os(dylib_path_var()) {
        Some(v) => v,
        None => return vec![],
    };
    env::split_paths(&var).collect()
}

/// Adds a list of lookup paths to `cmd`'s link library lookup path.
pub fn add_link_lib_path(path: Vec<PathBuf>, cmd: &mut Command) {
    let mut list = link_lib_path();
    for path in path {
        list.insert(0, path);
    }
    cmd.env(link_lib_path_var(), t!(env::join_paths(list)));
}

/// Returns the environment variable which the link library lookup path
/// resides in for this platform.
fn link_lib_path_var() -> &'static str {
    if cfg!(target_env = "msvc") { "LIB" } else { "LIBRARY_PATH" }
}

/// Parses the `link_lib_path_var()` environment variable, returning a list of
/// paths that are members of this lookup path.
fn link_lib_path() -> Vec<PathBuf> {
    let var = match env::var_os(link_lib_path_var()) {
        Some(v) => v,
        None => return vec![],
    };
    env::split_paths(&var).collect()
}

/// `push` all components to `buf`. On windows, append `.exe` to the last component.
pub fn push_exe_path(mut buf: PathBuf, components: &[&str]) -> PathBuf {
    let (&file, components) = components.split_last().expect("at least one component required");
    let mut file = file.to_owned();

    if cfg!(windows) {
        file.push_str(".exe");
    }

    buf.extend(components);
    buf.push(file);

    buf
}

pub struct TimeIt(bool, Instant);

/// Returns an RAII structure that prints out how long it took to drop.
pub fn timeit(builder: &Builder<'_>) -> TimeIt {
    TimeIt(builder.config.dry_run, Instant::now())
}

impl Drop for TimeIt {
    fn drop(&mut self) {
        let time = self.1.elapsed();
        if !self.0 {
            println!("\tfinished in {}.{:03} seconds", time.as_secs(), time.subsec_millis());
        }
    }
}

/// Symlinks two directories, using junctions on Windows and normal symlinks on
/// Unix.
pub fn symlink_dir(config: &Config, src: &Path, dest: &Path) -> io::Result<()> {
    if config.dry_run {
        return Ok(());
    }
    let _ = fs::remove_dir(dest);
    return symlink_dir_inner(src, dest);

    #[cfg(not(windows))]
    fn symlink_dir_inner(src: &Path, dest: &Path) -> io::Result<()> {
        use std::os::unix::fs;
        fs::symlink(src, dest)
    }

    // Creating a directory junction on windows involves dealing with reparse
    // points and the DeviceIoControl function, and this code is a skeleton of
    // what can be found here:
    //
    // http://www.flexhex.com/docs/articles/hard-links.phtml
    #[cfg(windows)]
    fn symlink_dir_inner(target: &Path, junction: &Path) -> io::Result<()> {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use std::ptr;

        use winapi::shared::minwindef::{DWORD, WORD};
        use winapi::um::fileapi::{CreateFileW, OPEN_EXISTING};
        use winapi::um::handleapi::CloseHandle;
        use winapi::um::ioapiset::DeviceIoControl;
        use winapi::um::winbase::{FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT};
        use winapi::um::winioctl::FSCTL_SET_REPARSE_POINT;
        use winapi::um::winnt::{
            FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, GENERIC_WRITE,
            IO_REPARSE_TAG_MOUNT_POINT, MAXIMUM_REPARSE_DATA_BUFFER_SIZE, WCHAR,
        };

        #[allow(non_snake_case)]
        #[repr(C)]
        struct REPARSE_MOUNTPOINT_DATA_BUFFER {
            ReparseTag: DWORD,
            ReparseDataLength: DWORD,
            Reserved: WORD,
            ReparseTargetLength: WORD,
            ReparseTargetMaximumLength: WORD,
            Reserved1: WORD,
            ReparseTarget: WCHAR,
        }

        fn to_u16s<S: AsRef<OsStr>>(s: S) -> io::Result<Vec<u16>> {
            Ok(s.as_ref().encode_wide().chain(Some(0)).collect())
        }

        // We're using low-level APIs to create the junction, and these are more
        // picky about paths. For example, forward slashes cannot be used as a
        // path separator, so we should try to canonicalize the path first.
        let target = fs::canonicalize(target)?;

        fs::create_dir(junction)?;

        let path = to_u16s(junction)?;

        unsafe {
            let h = CreateFileW(
                path.as_ptr(),
                GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                ptr::null_mut(),
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                ptr::null_mut(),
            );

            let mut data = [0u8; MAXIMUM_REPARSE_DATA_BUFFER_SIZE as usize];
            let db = data.as_mut_ptr() as *mut REPARSE_MOUNTPOINT_DATA_BUFFER;
            let buf = &mut (*db).ReparseTarget as *mut u16;
            let mut i = 0;
            // FIXME: this conversion is very hacky
            let v = br"\??\";
            let v = v.iter().map(|x| *x as u16);
            for c in v.chain(target.as_os_str().encode_wide().skip(4)) {
                *buf.offset(i) = c;
                i += 1;
            }
            *buf.offset(i) = 0;
            i += 1;
            (*db).ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
            (*db).ReparseTargetMaximumLength = (i * 2) as WORD;
            (*db).ReparseTargetLength = ((i - 1) * 2) as WORD;
            (*db).ReparseDataLength = (*db).ReparseTargetLength as DWORD + 12;

            let mut ret = 0;
            let res = DeviceIoControl(
                h as *mut _,
                FSCTL_SET_REPARSE_POINT,
                data.as_ptr() as *mut _,
                (*db).ReparseDataLength + 8,
                ptr::null_mut(),
                0,
                &mut ret,
                ptr::null_mut(),
            );

            let out = if res == 0 { Err(io::Error::last_os_error()) } else { Ok(()) };
            CloseHandle(h);
            out
        }
    }
}

/// The CI environment rustbuild is running in. This mainly affects how the logs
/// are printed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CiEnv {
    /// Not a CI environment.
    None,
    /// The Azure Pipelines environment, for Linux (including Docker), Windows, and macOS builds.
    AzurePipelines,
    /// The GitHub Actions environment, for Linux (including Docker), Windows and macOS builds.
    GitHubActions,
}

impl CiEnv {
    /// Obtains the current CI environment.
    pub fn current() -> CiEnv {
        if env::var("TF_BUILD").map_or(false, |e| e == "True") {
            CiEnv::AzurePipelines
        } else if env::var("GITHUB_ACTIONS").map_or(false, |e| e == "true") {
            CiEnv::GitHubActions
        } else {
            CiEnv::None
        }
    }

    /// If in a CI environment, forces the command to run with colors.
    pub fn force_coloring_in_ci(self, cmd: &mut Command) {
        if self != CiEnv::None {
            // Due to use of stamp/docker, the output stream of rustbuild is not
            // a TTY in CI, so coloring is by-default turned off.
            // The explicit `TERM=xterm` environment is needed for
            // `--color always` to actually work. This env var was lost when
            // compiling through the Makefile. Very strange.
            cmd.env("TERM", "xterm").args(&["--color", "always"]);
        }
    }
}

pub fn forcing_clang_based_tests() -> bool {
    if let Some(var) = env::var_os("RUSTBUILD_FORCE_CLANG_BASED_TESTS") {
        match &var.to_string_lossy().to_lowercase()[..] {
            "1" | "yes" | "on" => true,
            "0" | "no" | "off" => false,
            other => {
                // Let's make sure typos don't go unnoticed
                panic!(
                    "Unrecognized option '{}' set in \
                        RUSTBUILD_FORCE_CLANG_BASED_TESTS",
                    other
                )
            }
        }
    } else {
        false
    }
}

pub fn use_host_linker(target: TargetSelection) -> bool {
    // FIXME: this information should be gotten by checking the linker flavor
    // of the rustc target
    !(target.contains("emscripten")
        || target.contains("wasm32")
        || target.contains("nvptx")
        || target.contains("fortanix")
        || target.contains("fuchsia")
        || target.contains("bpf"))
}

pub fn is_valid_test_suite_arg<'a, P: AsRef<Path>>(
    path: &'a Path,
    suite_path: P,
    builder: &Builder<'_>,
) -> Option<&'a str> {
    let suite_path = suite_path.as_ref();
    let path = match path.strip_prefix(".") {
        Ok(p) => p,
        Err(_) => path,
    };
    if !path.starts_with(suite_path) {
        return None;
    }
    let exists = path.is_dir() || path.is_file();
    if !exists {
        if let Some(p) = path.to_str() {
            builder.info(&format!("Warning: Skipping \"{}\": not a regular file or directory", p));
        }
        return None;
    }
    // Since test suite paths are themselves directories, if we don't
    // specify a directory or file, we'll get an empty string here
    // (the result of the test suite directory without its suite prefix).
    // Therefore, we need to filter these out, as only the first --test-args
    // flag is respected, so providing an empty --test-args conflicts with
    // any following it.
    match path.strip_prefix(suite_path).ok().and_then(|p| p.to_str()) {
        Some(s) if !s.is_empty() => Some(s),
        _ => None,
    }
}
