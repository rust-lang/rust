//! Various utility functions used throughout rustbuild.
//!
//! Simple things like testing the various filesystem operations here and there,
//! not a lot of interesting happenings here unfortunately.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::builder::Builder;
use crate::config::{Config, TargetSelection};
use crate::OnceCell;

/// A helper macro to `unwrap` a result except also print out details like:
///
/// * The file/line of the panic
/// * The expression that failed
/// * The error itself
///
/// This is currently used judiciously throughout the build system rather than
/// using a `Result` with `try!`, but this may change one day...
#[macro_export]
macro_rules! t {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    };
    // it can show extra info in the second parameter
    ($e:expr, $extra:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {} ({:?})", stringify!($e), e, $extra),
        }
    };
}
pub use t;

/// Given an executable called `name`, return the filename for the
/// executable for a particular target.
pub fn exe(name: &str, target: TargetSelection) -> String {
    if target.contains("windows") {
        format!("{}.exe", name)
    } else if target.contains("uefi") {
        format!("{}.efi", name)
    } else {
        name.to_string()
    }
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
/// If the dylib_path_var is already set for this cmd, the old value will be overwritten!
pub fn add_dylib_path(path: Vec<PathBuf>, cmd: &mut Command) {
    let mut list = dylib_path();
    for path in path {
        list.insert(0, path);
    }
    cmd.env(dylib_path_var(), t!(env::join_paths(list)));
}

include!("dylib_util.rs");

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

pub struct TimeIt(bool, Instant);

/// Returns an RAII structure that prints out how long it took to drop.
pub fn timeit(builder: &Builder<'_>) -> TimeIt {
    TimeIt(builder.config.dry_run(), Instant::now())
}

impl Drop for TimeIt {
    fn drop(&mut self) {
        let time = self.1.elapsed();
        if !self.0 {
            println!("\tfinished in {}.{:03} seconds", time.as_secs(), time.subsec_millis());
        }
    }
}

/// Used for download caching
pub(crate) fn program_out_of_date(stamp: &Path, key: &str) -> bool {
    if !stamp.exists() {
        return true;
    }
    t!(fs::read_to_string(stamp)) != key
}

/// Symlinks two directories, using junctions on Windows and normal symlinks on
/// Unix.
pub fn symlink_dir(config: &Config, src: &Path, dest: &Path) -> io::Result<()> {
    if config.dry_run() {
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

        use windows::{
            core::PCWSTR,
            Win32::Foundation::{CloseHandle, HANDLE},
            Win32::Storage::FileSystem::{
                CreateFileW, FILE_ACCESS_FLAGS, FILE_FLAG_BACKUP_SEMANTICS,
                FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
                MAXIMUM_REPARSE_DATA_BUFFER_SIZE, OPEN_EXISTING,
            },
            Win32::System::Ioctl::FSCTL_SET_REPARSE_POINT,
            Win32::System::SystemServices::{GENERIC_WRITE, IO_REPARSE_TAG_MOUNT_POINT},
            Win32::System::IO::DeviceIoControl,
        };

        #[allow(non_snake_case)]
        #[repr(C)]
        struct REPARSE_MOUNTPOINT_DATA_BUFFER {
            ReparseTag: u32,
            ReparseDataLength: u32,
            Reserved: u16,
            ReparseTargetLength: u16,
            ReparseTargetMaximumLength: u16,
            Reserved1: u16,
            ReparseTarget: u16,
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

        let h = unsafe {
            CreateFileW(
                PCWSTR(path.as_ptr()),
                FILE_ACCESS_FLAGS(GENERIC_WRITE),
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                None,
                OPEN_EXISTING,
                FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS,
                HANDLE::default(),
            )
        }
        .map_err(|_| io::Error::last_os_error())?;

        unsafe {
            #[repr(C, align(8))]
            struct Align8<T>(T);
            let mut data = Align8([0u8; MAXIMUM_REPARSE_DATA_BUFFER_SIZE as usize]);
            let db = data.0.as_mut_ptr() as *mut REPARSE_MOUNTPOINT_DATA_BUFFER;
            let buf = core::ptr::addr_of_mut!((*db).ReparseTarget) as *mut u16;
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
            (*db).ReparseTargetMaximumLength = (i * 2) as u16;
            (*db).ReparseTargetLength = ((i - 1) * 2) as u16;
            (*db).ReparseDataLength = ((*db).ReparseTargetLength + 12) as u32;

            let mut ret = 0u32;
            DeviceIoControl(
                h,
                FSCTL_SET_REPARSE_POINT,
                Some(db.cast()),
                (*db).ReparseDataLength + 8,
                None,
                0,
                Some(&mut ret),
                None,
            )
            .ok()
            .map_err(|_| io::Error::last_os_error())?;
        }

        unsafe { CloseHandle(h) };
        Ok(())
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
        || target.contains("bpf")
        || target.contains("switch"))
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
    let abs_path = builder.src.join(path);
    let exists = abs_path.is_dir() || abs_path.is_file();
    if !exists {
        panic!(
            "Invalid test suite filter \"{}\": file or directory does not exist",
            abs_path.display()
        );
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

pub fn run(cmd: &mut Command, print_cmd_on_fail: bool) {
    if !try_run(cmd, print_cmd_on_fail) {
        crate::detail_exit(1);
    }
}

pub fn try_run(cmd: &mut Command, print_cmd_on_fail: bool) -> bool {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !status.success() && print_cmd_on_fail {
        println!(
            "\n\ncommand did not execute successfully: {:?}\n\
             expected success, got: {}\n\n",
            cmd, status
        );
    }
    status.success()
}

pub fn check_run(cmd: &mut Command, print_cmd_on_fail: bool) -> bool {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => {
            println!("failed to execute command: {:?}\nerror: {}", cmd, e);
            return false;
        }
    };
    if !status.success() && print_cmd_on_fail {
        println!(
            "\n\ncommand did not execute successfully: {:?}\n\
             expected success, got: {}\n\n",
            cmd, status
        );
    }
    status.success()
}

pub fn run_suppressed(cmd: &mut Command) {
    if !try_run_suppressed(cmd) {
        crate::detail_exit(1);
    }
}

pub fn try_run_suppressed(cmd: &mut Command) -> bool {
    let output = match cmd.output() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !output.status.success() {
        println!(
            "\n\ncommand did not execute successfully: {:?}\n\
             expected success, got: {}\n\n\
             stdout ----\n{}\n\
             stderr ----\n{}\n\n",
            cmd,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.status.success()
}

pub fn make(host: &str) -> PathBuf {
    if host.contains("dragonfly")
        || host.contains("freebsd")
        || host.contains("netbsd")
        || host.contains("openbsd")
    {
        PathBuf::from("gmake")
    } else {
        PathBuf::from("make")
    }
}

#[track_caller]
pub fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !output.status.success() {
        panic!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}",
            cmd, output.status
        );
    }
    String::from_utf8(output.stdout).unwrap()
}

pub fn output_result(cmd: &mut Command) -> Result<String, String> {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => return Err(format!("failed to run command: {:?}: {}", cmd, e)),
    };
    if !output.status.success() {
        return Err(format!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}\n{}",
            cmd,
            output.status,
            String::from_utf8(output.stderr).map_err(|err| format!("{err:?}"))?
        ));
    }
    Ok(String::from_utf8(output.stdout).map_err(|err| format!("{err:?}"))?)
}

/// Returns the last-modified time for `path`, or zero if it doesn't exist.
pub fn mtime(path: &Path) -> SystemTime {
    fs::metadata(path).and_then(|f| f.modified()).unwrap_or(UNIX_EPOCH)
}

/// Returns `true` if `dst` is up to date given that the file or files in `src`
/// are used to generate it.
///
/// Uses last-modified time checks to verify this.
pub fn up_to_date(src: &Path, dst: &Path) -> bool {
    if !dst.exists() {
        return false;
    }
    let threshold = mtime(dst);
    let meta = match fs::metadata(src) {
        Ok(meta) => meta,
        Err(e) => panic!("source {:?} failed to get metadata: {}", src, e),
    };
    if meta.is_dir() {
        dir_up_to_date(src, threshold)
    } else {
        meta.modified().unwrap_or(UNIX_EPOCH) <= threshold
    }
}

fn dir_up_to_date(src: &Path, threshold: SystemTime) -> bool {
    t!(fs::read_dir(src)).map(|e| t!(e)).all(|e| {
        let meta = t!(e.metadata());
        if meta.is_dir() {
            dir_up_to_date(&e.path(), threshold)
        } else {
            meta.modified().unwrap_or(UNIX_EPOCH) < threshold
        }
    })
}

fn fail(s: &str) -> ! {
    eprintln!("\n\n{}\n\n", s);
    crate::detail_exit(1);
}

/// Copied from `std::path::absolute` until it stabilizes.
///
/// FIXME: this shouldn't exist.
pub(crate) fn absolute(path: &Path) -> PathBuf {
    if path.as_os_str().is_empty() {
        panic!("can't make empty path absolute");
    }
    #[cfg(unix)]
    {
        t!(absolute_unix(path), format!("could not make path absolute: {}", path.display()))
    }
    #[cfg(windows)]
    {
        t!(absolute_windows(path), format!("could not make path absolute: {}", path.display()))
    }
    #[cfg(not(any(unix, windows)))]
    {
        println!("warning: bootstrap is not supported on non-unix platforms");
        t!(std::fs::canonicalize(t!(std::env::current_dir()))).join(path)
    }
}

#[cfg(unix)]
/// Make a POSIX path absolute without changing its semantics.
fn absolute_unix(path: &Path) -> io::Result<PathBuf> {
    // This is mostly a wrapper around collecting `Path::components`, with
    // exceptions made where this conflicts with the POSIX specification.
    // See 4.13 Pathname Resolution, IEEE Std 1003.1-2017
    // https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13

    use std::os::unix::prelude::OsStrExt;
    let mut components = path.components();
    let path_os = path.as_os_str().as_bytes();

    let mut normalized = if path.is_absolute() {
        // "If a pathname begins with two successive <slash> characters, the
        // first component following the leading <slash> characters may be
        // interpreted in an implementation-defined manner, although more than
        // two leading <slash> characters shall be treated as a single <slash>
        // character."
        if path_os.starts_with(b"//") && !path_os.starts_with(b"///") {
            components.next();
            PathBuf::from("//")
        } else {
            PathBuf::new()
        }
    } else {
        env::current_dir()?
    };
    normalized.extend(components);

    // "Interfaces using pathname resolution may specify additional constraints
    // when a pathname that does not name an existing directory contains at
    // least one non- <slash> character and contains one or more trailing
    // <slash> characters".
    // A trailing <slash> is also meaningful if "a symbolic link is
    // encountered during pathname resolution".

    if path_os.ends_with(b"/") {
        normalized.push("");
    }

    Ok(normalized)
}

#[cfg(windows)]
fn absolute_windows(path: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    use std::ffi::OsString;
    use std::io::Error;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::ptr::null_mut;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetFullPathNameW(
            lpFileName: *const u16,
            nBufferLength: u32,
            lpBuffer: *mut u16,
            lpFilePart: *mut *const u16,
        ) -> u32;
    }

    unsafe {
        // encode the path as UTF-16
        let path: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let mut buffer = Vec::new();
        // Loop until either success or failure.
        loop {
            // Try to get the absolute path
            let len = GetFullPathNameW(
                path.as_ptr(),
                buffer.len().try_into().unwrap(),
                buffer.as_mut_ptr(),
                null_mut(),
            );
            match len as usize {
                // Failure
                0 => return Err(Error::last_os_error()),
                // Buffer is too small, resize.
                len if len > buffer.len() => buffer.resize(len, 0),
                // Success!
                len => {
                    buffer.truncate(len);
                    return Ok(OsString::from_wide(&buffer).into());
                }
            }
        }
    }
}

/// Adapted from https://github.com/llvm/llvm-project/blob/782e91224601e461c019e0a4573bbccc6094fbcd/llvm/cmake/modules/HandleLLVMOptions.cmake#L1058-L1079
///
/// When `clang-cl` is used with instrumentation, we need to add clang's runtime library resource
/// directory to the linker flags, otherwise there will be linker errors about the profiler runtime
/// missing. This function returns the path to that directory.
pub fn get_clang_cl_resource_dir(clang_cl_path: &str) -> PathBuf {
    // Similar to how LLVM does it, to find clang's library runtime directory:
    // - we ask `clang-cl` to locate the `clang_rt.builtins` lib.
    let mut builtins_locator = Command::new(clang_cl_path);
    builtins_locator.args(&["/clang:-print-libgcc-file-name", "/clang:--rtlib=compiler-rt"]);

    let clang_rt_builtins = output(&mut builtins_locator);
    let clang_rt_builtins = Path::new(clang_rt_builtins.trim());
    assert!(
        clang_rt_builtins.exists(),
        "`clang-cl` must correctly locate the library runtime directory"
    );

    // - the profiler runtime will be located in the same directory as the builtins lib, like
    // `$LLVM_DISTRO_ROOT/lib/clang/$LLVM_VERSION/lib/windows`.
    let clang_rt_dir = clang_rt_builtins.parent().expect("The clang lib folder should exist");
    clang_rt_dir.to_path_buf()
}

pub fn lld_flag_no_threads(is_windows: bool) -> &'static str {
    static LLD_NO_THREADS: OnceCell<(&'static str, &'static str)> = OnceCell::new();
    let (windows, other) = LLD_NO_THREADS.get_or_init(|| {
        let out = output(Command::new("lld").arg("-flavor").arg("ld").arg("--version"));
        let newer = match (out.find(char::is_numeric), out.find('.')) {
            (Some(b), Some(e)) => out.as_str()[b..e].parse::<i32>().ok().unwrap_or(14) > 10,
            _ => true,
        };
        if newer { ("/threads:1", "--threads=1") } else { ("/no-threads", "--no-threads") }
    });
    if is_windows { windows } else { other }
}
