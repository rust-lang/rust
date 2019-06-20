//! Various utility functions used throughout rustbuild.
//!
//! Simple things like testing the various filesystem operations here and there,
//! not a lot of interesting happenings here unfortunately.

use std::env;
use std::str;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, Instant};

use build_helper::t;

use crate::config::Config;
use crate::builder::Builder;

/// Returns the `name` as the filename of a static library for `target`.
pub fn staticlib(name: &str, target: &str) -> String {
    if target.contains("windows") {
        format!("{}.lib", name)
    } else {
        format!("lib{}.a", name)
    }
}

/// Given an executable called `name`, return the filename for the
/// executable for a particular target.
pub fn exe(name: &str, target: &str) -> String {
    if target.contains("windows") {
        format!("{}.exe", name)
    } else {
        name.to_string()
    }
}

/// Returns `true` if the file name given looks like a dynamic library.
pub fn is_dylib(name: &str) -> bool {
    name.ends_with(".dylib") || name.ends_with(".so") || name.ends_with(".dll")
}

/// Returns the corresponding relative library directory that the compiler's
/// dylibs will be found in.
pub fn libdir(target: &str) -> &'static str {
    if target.contains("windows") {"bin"} else {"lib"}
}

/// Adds a list of lookup paths to `cmd`'s dynamic library lookup path.
pub fn add_lib_path(path: Vec<PathBuf>, cmd: &mut Command) {
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
            println!("\tfinished in {}.{:03}",
                    time.as_secs(),
                    time.subsec_nanos() / 1_000_000);
        }
    }
}

/// Symlinks two directories, using junctions on Windows and normal symlinks on
/// Unix.
pub fn symlink_dir(config: &Config, src: &Path, dest: &Path) -> io::Result<()> {
    if config.dry_run { return Ok(()); }
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
    //
    // Copied from std
    #[cfg(windows)]
    #[allow(nonstandard_style)]
    fn symlink_dir_inner(target: &Path, junction: &Path) -> io::Result<()> {
        use std::ptr;
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        const MAXIMUM_REPARSE_DATA_BUFFER_SIZE: usize = 16 * 1024;
        const GENERIC_WRITE: DWORD = 0x40000000;
        const OPEN_EXISTING: DWORD = 3;
        const FILE_FLAG_OPEN_REPARSE_POINT: DWORD = 0x00200000;
        const FILE_FLAG_BACKUP_SEMANTICS: DWORD = 0x02000000;
        const FSCTL_SET_REPARSE_POINT: DWORD = 0x900a4;
        const IO_REPARSE_TAG_MOUNT_POINT: DWORD = 0xa0000003;
        const FILE_SHARE_DELETE: DWORD = 0x4;
        const FILE_SHARE_READ: DWORD = 0x1;
        const FILE_SHARE_WRITE: DWORD = 0x2;

        type BOOL = i32;
        type DWORD = u32;
        type HANDLE = *mut u8;
        type LPCWSTR = *const u16;
        type LPDWORD = *mut DWORD;
        type LPOVERLAPPED = *mut u8;
        type LPSECURITY_ATTRIBUTES = *mut u8;
        type LPVOID = *mut u8;
        type WCHAR = u16;
        type WORD = u16;

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

        extern "system" {
            fn CreateFileW(lpFileName: LPCWSTR,
                           dwDesiredAccess: DWORD,
                           dwShareMode: DWORD,
                           lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
                           dwCreationDisposition: DWORD,
                           dwFlagsAndAttributes: DWORD,
                           hTemplateFile: HANDLE)
                           -> HANDLE;
            fn DeviceIoControl(hDevice: HANDLE,
                               dwIoControlCode: DWORD,
                               lpInBuffer: LPVOID,
                               nInBufferSize: DWORD,
                               lpOutBuffer: LPVOID,
                               nOutBufferSize: DWORD,
                               lpBytesReturned: LPDWORD,
                               lpOverlapped: LPOVERLAPPED) -> BOOL;
            fn CloseHandle(hObject: HANDLE) -> BOOL;
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
            let h = CreateFileW(path.as_ptr(),
                                GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                ptr::null_mut(),
                                OPEN_EXISTING,
                                FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                                ptr::null_mut());

            let mut data = [0u8; MAXIMUM_REPARSE_DATA_BUFFER_SIZE];
            let db = data.as_mut_ptr()
                            as *mut REPARSE_MOUNTPOINT_DATA_BUFFER;
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
            (*db).ReparseDataLength =
                    (*db).ReparseTargetLength as DWORD + 12;

            let mut ret = 0;
            let res = DeviceIoControl(h as *mut _,
                                      FSCTL_SET_REPARSE_POINT,
                                      data.as_ptr() as *mut _,
                                      (*db).ReparseDataLength + 8,
                                      ptr::null_mut(), 0,
                                      &mut ret,
                                      ptr::null_mut());

            let out = if res == 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            };
            CloseHandle(h);
            out
        }
    }
}

/// An RAII structure that indicates all output until this instance is dropped
/// is part of the same group.
///
/// On Travis CI, these output will be folded by default, together with the
/// elapsed time in this block. This reduces noise from unnecessary logs,
/// allowing developers to quickly identify the error.
///
/// Travis CI supports folding by printing `travis_fold:start:<name>` and
/// `travis_fold:end:<name>` around the block. Time elapsed is recognized
/// similarly with `travis_time:[start|end]:<name>`. These are undocumented, but
/// can easily be deduced from source code of the [Travis build commands].
///
/// [Travis build commands]:
/// https://github.com/travis-ci/travis-build/blob/f603c0089/lib/travis/build/templates/header.sh
pub struct OutputFolder {
    name: String,
    start_time: SystemTime, // we need SystemTime to get the UNIX timestamp.
}

impl OutputFolder {
    /// Creates a new output folder with the given group name.
    pub fn new(name: String) -> OutputFolder {
        // "\r" moves the cursor to the beginning of the line, and "\x1b[0K" is
        // the ANSI escape code to clear from the cursor to end of line.
        // Travis seems to have trouble when _not_ using "\r\x1b[0K", that will
        // randomly put lines to the top of the webpage.
        print!("travis_fold:start:{0}\r\x1b[0Ktravis_time:start:{0}\r\x1b[0K", name);
        OutputFolder {
            name,
            start_time: SystemTime::now(),
        }
    }
}

impl Drop for OutputFolder {
    fn drop(&mut self) {
        use std::time::*;
        use std::u64;

        fn to_nanos(duration: Result<Duration, SystemTimeError>) -> u64 {
            match duration {
                Ok(d) => d.as_secs() * 1_000_000_000 + d.subsec_nanos() as u64,
                Err(_) => u64::MAX,
            }
        }

        let end_time = SystemTime::now();
        let duration = end_time.duration_since(self.start_time);
        let start = self.start_time.duration_since(UNIX_EPOCH);
        let finish = end_time.duration_since(UNIX_EPOCH);
        println!(
            "travis_fold:end:{0}\r\x1b[0K\n\
                travis_time:end:{0}:start={1},finish={2},duration={3}\r\x1b[0K",
            self.name,
            to_nanos(start),
            to_nanos(finish),
            to_nanos(duration)
        );
        io::stdout().flush().unwrap();
    }
}

/// The CI environment rustbuild is running in. This mainly affects how the logs
/// are printed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CiEnv {
    /// Not a CI environment.
    None,
    /// The Travis CI environment, for Linux (including Docker) and macOS builds.
    Travis,
    /// The AppVeyor environment, for Windows builds.
    AppVeyor,
    /// The Azure Pipelines environment, for Linux (including Docker), Windows, and macOS builds.
    AzurePipelines,
}

impl CiEnv {
    /// Obtains the current CI environment.
    pub fn current() -> CiEnv {
        if env::var("TRAVIS").ok().map_or(false, |e| &*e == "true") {
            CiEnv::Travis
        } else if env::var("APPVEYOR").ok().map_or(false, |e| &*e == "True") {
            CiEnv::AppVeyor
        } else if env::var("TF_BUILD").ok().map_or(false, |e| &*e == "True") {
            CiEnv::AzurePipelines
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
                panic!("Unrecognized option '{}' set in \
                        RUSTBUILD_FORCE_CLANG_BASED_TESTS", other)
            }
        }
    } else {
        false
    }
}
