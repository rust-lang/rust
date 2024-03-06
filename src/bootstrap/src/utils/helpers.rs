//! Various utility functions used throughout rustbuild.
//!
//! Simple things like testing the various filesystem operations here and there,
//! not a lot of interesting happenings here unfortunately.

use build_helper::util::fail;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str;
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::core::builder::Builder;
use crate::core::config::{Config, TargetSelection};
use crate::LldMode;

pub use crate::utils::dylib::{dylib_path, dylib_path_var};

#[cfg(test)]
mod tests;

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

pub fn exe(name: &str, target: TargetSelection) -> String {
    crate::utils::dylib::exe(name, &target.triple)
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
    if target.is_windows() { "bin" } else { "lib" }
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
pub fn symlink_dir(config: &Config, original: &Path, link: &Path) -> io::Result<()> {
    if config.dry_run() {
        return Ok(());
    }
    let _ = fs::remove_dir(link);
    return symlink_dir_inner(original, link);

    #[cfg(not(windows))]
    fn symlink_dir_inner(original: &Path, link: &Path) -> io::Result<()> {
        use std::os::unix::fs;
        fs::symlink(original, link)
    }

    #[cfg(windows)]
    fn symlink_dir_inner(target: &Path, junction: &Path) -> io::Result<()> {
        junction::create(&target, &junction)
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
                    "Unrecognized option '{other}' set in \
                        RUSTBUILD_FORCE_CLANG_BASED_TESTS"
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

pub fn target_supports_cranelift_backend(target: TargetSelection) -> bool {
    if target.contains("linux") {
        target.contains("x86_64")
            || target.contains("aarch64")
            || target.contains("s390x")
            || target.contains("riscv64gc")
    } else if target.contains("darwin") || target.is_windows() {
        target.contains("x86_64")
    } else {
        false
    }
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

pub fn check_run(cmd: &mut Command, print_cmd_on_fail: bool) -> bool {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => {
            println!("failed to execute command: {cmd:?}\nERROR: {e}");
            return false;
        }
    };
    if !status.success() && print_cmd_on_fail {
        println!(
            "\n\ncommand did not execute successfully: {cmd:?}\n\
             expected success, got: {status}\n\n"
        );
    }
    status.success()
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
        Err(e) => fail(&format!("failed to execute command: {cmd:?}\nERROR: {e}")),
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
        Err(e) => panic!("source {src:?} failed to get metadata: {e}"),
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
        println!("WARNING: bootstrap is not supported on non-unix platforms");
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

/// Adapted from <https://github.com/llvm/llvm-project/blob/782e91224601e461c019e0a4573bbccc6094fbcd/llvm/cmake/modules/HandleLLVMOptions.cmake#L1058-L1079>
///
/// When `clang-cl` is used with instrumentation, we need to add clang's runtime library resource
/// directory to the linker flags, otherwise there will be linker errors about the profiler runtime
/// missing. This function returns the path to that directory.
pub fn get_clang_cl_resource_dir(clang_cl_path: &str) -> PathBuf {
    // Similar to how LLVM does it, to find clang's library runtime directory:
    // - we ask `clang-cl` to locate the `clang_rt.builtins` lib.
    let mut builtins_locator = Command::new(clang_cl_path);
    builtins_locator.args(["/clang:-print-libgcc-file-name", "/clang:--rtlib=compiler-rt"]);

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

/// Returns a flag that configures LLD to use only a single thread.
/// If we use an external LLD, we need to find out which version is it to know which flag should we
/// pass to it (LLD older than version 10 had a different flag).
fn lld_flag_no_threads(lld_mode: LldMode, is_windows: bool) -> &'static str {
    static LLD_NO_THREADS: OnceLock<(&'static str, &'static str)> = OnceLock::new();

    let new_flags = ("/threads:1", "--threads=1");
    let old_flags = ("/no-threads", "--no-threads");

    let (windows_flag, other_flag) = LLD_NO_THREADS.get_or_init(|| {
        let newer_version = match lld_mode {
            LldMode::External => {
                let out = output(Command::new("lld").arg("-flavor").arg("ld").arg("--version"));
                match (out.find(char::is_numeric), out.find('.')) {
                    (Some(b), Some(e)) => out.as_str()[b..e].parse::<i32>().ok().unwrap_or(14) > 10,
                    _ => true,
                }
            }
            _ => true,
        };
        if newer_version { new_flags } else { old_flags }
    });
    if is_windows { windows_flag } else { other_flag }
}

pub fn dir_is_empty(dir: &Path) -> bool {
    t!(std::fs::read_dir(dir)).next().is_none()
}

/// Extract the beta revision from the full version string.
///
/// The full version string looks like "a.b.c-beta.y". And we need to extract
/// the "y" part from the string.
pub fn extract_beta_rev(version: &str) -> Option<String> {
    let parts = version.splitn(2, "-beta.").collect::<Vec<_>>();
    let count = parts.get(1).and_then(|s| s.find(' ').map(|p| s[..p].to_string()));

    count
}

pub enum LldThreads {
    Yes,
    No,
}

/// Returns the linker arguments for rustc/rustdoc for the given builder and target.
pub fn linker_args(
    builder: &Builder<'_>,
    target: TargetSelection,
    lld_threads: LldThreads,
) -> Vec<String> {
    let mut args = linker_flags(builder, target, lld_threads);

    if let Some(linker) = builder.linker(target) {
        args.push(format!("-Clinker={}", linker.display()));
    }

    args
}

/// Returns the linker arguments for rustc/rustdoc for the given builder and target, without the
/// -Clinker flag.
pub fn linker_flags(
    builder: &Builder<'_>,
    target: TargetSelection,
    lld_threads: LldThreads,
) -> Vec<String> {
    let mut args = vec![];
    if !builder.is_lld_direct_linker(target) && builder.config.lld_mode.is_used() {
        args.push(String::from("-Clink-arg=-fuse-ld=lld"));

        if matches!(lld_threads, LldThreads::No) {
            args.push(format!(
                "-Clink-arg=-Wl,{}",
                lld_flag_no_threads(builder.config.lld_mode, target.is_windows())
            ));
        }
    }
    args
}

pub fn add_rustdoc_cargo_linker_args(
    cmd: &mut Command,
    builder: &Builder<'_>,
    target: TargetSelection,
    lld_threads: LldThreads,
) {
    let args = linker_args(builder, target, lld_threads);
    let mut flags = cmd
        .get_envs()
        .find_map(|(k, v)| if k == OsStr::new("RUSTDOCFLAGS") { v } else { None })
        .unwrap_or_default()
        .to_os_string();
    for arg in args {
        if !flags.is_empty() {
            flags.push(" ");
        }
        flags.push(arg);
    }
    if !flags.is_empty() {
        cmd.env("RUSTDOCFLAGS", flags);
    }
}

/// Converts `T` into a hexadecimal `String`.
pub fn hex_encode<T>(input: T) -> String
where
    T: AsRef<[u8]>,
{
    input.as_ref().iter().map(|x| format!("{:02x}", x)).collect()
}

/// Create a `--check-cfg` argument invocation for a given name
/// and it's values.
pub fn check_cfg_arg(name: &str, values: Option<&[&str]>) -> String {
    // Creating a string of the values by concatenating each value:
    // ',values("tvos","watchos")' or '' (nothing) when there are no values.
    let next = match values {
        Some(values) => {
            let mut tmp = values.iter().flat_map(|val| [",", "\"", val, "\""]).collect::<String>();

            tmp.insert_str(1, "values(");
            tmp.push(')');
            tmp
        }
        None => "".to_string(),
    };
    format!("--check-cfg=cfg({name}{next})")
}
