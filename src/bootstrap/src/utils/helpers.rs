//! Various utility functions used throughout bootstrap.
//!
//! Simple things like testing the various filesystem operations here and there,
//! not a lot of interesting happenings here unfortunately.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::thread::panicking;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{env, fs, io, panic, str};

use build_helper::util::fail;
use object::read::archive::ArchiveFile;

use crate::LldMode;
use crate::core::builder::Builder;
use crate::core::config::{Config, TargetSelection};
use crate::utils::exec::{BootstrapCommand, command};
pub use crate::utils::shared_helpers::{dylib_path, dylib_path_var};

#[cfg(test)]
mod tests;

/// A wrapper around `std::panic::Location` used to track the location of panics
/// triggered by `t` macro usage.
pub struct PanicTracker<'a>(pub &'a panic::Location<'a>);

impl Drop for PanicTracker<'_> {
    fn drop(&mut self) {
        if panicking() {
            eprintln!(
                "Panic was initiated from {}:{}:{}",
                self.0.file(),
                self.0.line(),
                self.0.column()
            );
        }
    }
}

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
    ($e:expr) => {{
        let _panic_guard = $crate::PanicTracker(std::panic::Location::caller());
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    }};
    // it can show extra info in the second parameter
    ($e:expr, $extra:expr) => {{
        let _panic_guard = $crate::PanicTracker(std::panic::Location::caller());
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {} ({:?})", stringify!($e), e, $extra),
        }
    }};
}

pub use t;
pub fn exe(name: &str, target: TargetSelection) -> String {
    crate::utils::shared_helpers::exe(name, &target.triple)
}

/// Returns the path to the split debug info for the specified file if it exists.
pub fn split_debuginfo(name: impl Into<PathBuf>) -> Option<PathBuf> {
    // FIXME: only msvc is currently supported

    let path = name.into();
    let pdb = path.with_extension("pdb");
    if pdb.exists() {
        return Some(pdb);
    }

    // pdbs get named with '-' replaced by '_'
    let file_name = pdb.file_name()?.to_str()?.replace("-", "_");

    let pdb: PathBuf = [path.parent()?, Path::new(&file_name)].into_iter().collect();
    pdb.exists().then_some(pdb)
}

/// Returns `true` if the file name given looks like a dynamic library.
pub fn is_dylib(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| {
        ext == "dylib" || ext == "so" || ext == "dll" || (ext == "a" && is_aix_shared_archive(path))
    })
}

/// Return the path to the containing submodule if available.
pub fn submodule_path_of(builder: &Builder<'_>, path: &str) -> Option<String> {
    let submodule_paths = build_helper::util::parse_gitmodules(&builder.src);
    submodule_paths.iter().find_map(|submodule_path| {
        if path.starts_with(submodule_path) { Some(submodule_path.to_string()) } else { None }
    })
}

fn is_aix_shared_archive(path: &Path) -> bool {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return false,
    };
    let reader = object::ReadCache::new(file);
    let archive = match ArchiveFile::parse(&reader) {
        Ok(result) => result,
        Err(_) => return false,
    };

    archive
        .members()
        .filter_map(Result::ok)
        .any(|entry| String::from_utf8_lossy(entry.name()).contains(".so"))
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
pub fn add_dylib_path(path: Vec<PathBuf>, cmd: &mut BootstrapCommand) {
    let mut list = dylib_path();
    for path in path {
        list.insert(0, path);
    }
    cmd.env(dylib_path_var(), t!(env::join_paths(list)));
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

/// Symlinks two directories, using junctions on Windows and normal symlinks on
/// Unix.
pub fn symlink_dir(config: &Config, original: &Path, link: &Path) -> io::Result<()> {
    if config.dry_run() {
        return Ok(());
    }
    let _ = fs::remove_dir_all(link);
    return symlink_dir_inner(original, link);

    #[cfg(not(windows))]
    fn symlink_dir_inner(original: &Path, link: &Path) -> io::Result<()> {
        use std::os::unix::fs;
        fs::symlink(original, link)
    }

    #[cfg(windows)]
    fn symlink_dir_inner(target: &Path, junction: &Path) -> io::Result<()> {
        junction::create(target, junction)
    }
}

/// Rename a file if from and to are in the same filesystem or
/// copy and remove the file otherwise
pub fn move_file<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()> {
    match fs::rename(&from, &to) {
        Err(e) if e.kind() == io::ErrorKind::CrossesDevices => {
            std::fs::copy(&from, &to)?;
            std::fs::remove_file(&from)
        }
        r => r,
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
    } else if target.contains("darwin") {
        target.contains("x86_64") || target.contains("aarch64")
    } else if target.is_windows() {
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

// FIXME: get rid of this function
pub fn check_run(cmd: &mut BootstrapCommand, print_cmd_on_fail: bool) -> bool {
    let status = match cmd.as_command_mut().status() {
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
    #[cfg(feature = "tracing")]
    let _run_span = crate::trace_cmd!(cmd);

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

/// Spawn a process and return a closure that will wait for the process
/// to finish and then return its output. This allows the spawned process
/// to do work without immediately blocking bootstrap.
#[track_caller]
pub fn start_process(cmd: &mut Command) -> impl FnOnce() -> String + use<> {
    let child = match cmd.stderr(Stdio::inherit()).stdout(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(e) => fail(&format!("failed to execute command: {cmd:?}\nERROR: {e}")),
    };

    let command = format!("{cmd:?}");

    move || {
        let output = child.wait_with_output().unwrap();

        if !output.status.success() {
            panic!(
                "command did not execute successfully: {}\n\
                 expected success, got: {}",
                command, output.status
            );
        }

        String::from_utf8(output.stdout).unwrap()
    }
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

/// Returns the filename without the hash prefix added by the cc crate.
///
/// Since v1.0.78 of the cc crate, object files are prefixed with a 16-character hash
/// to avoid filename collisions.
pub fn unhashed_basename(obj: &Path) -> &str {
    let basename = obj.file_stem().unwrap().to_str().expect("UTF-8 file name");
    basename.split_once('-').unwrap().1
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

/// Adapted from <https://github.com/llvm/llvm-project/blob/782e91224601e461c019e0a4573bbccc6094fbcd/llvm/cmake/modules/HandleLLVMOptions.cmake#L1058-L1079>
///
/// When `clang-cl` is used with instrumentation, we need to add clang's runtime library resource
/// directory to the linker flags, otherwise there will be linker errors about the profiler runtime
/// missing. This function returns the path to that directory.
pub fn get_clang_cl_resource_dir(builder: &Builder<'_>, clang_cl_path: &str) -> PathBuf {
    // Similar to how LLVM does it, to find clang's library runtime directory:
    // - we ask `clang-cl` to locate the `clang_rt.builtins` lib.
    let mut builtins_locator = command(clang_cl_path);
    builtins_locator.args(["/clang:-print-libgcc-file-name", "/clang:--rtlib=compiler-rt"]);

    let clang_rt_builtins = builtins_locator.run_capture_stdout(builder).stdout();
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
fn lld_flag_no_threads(builder: &Builder<'_>, lld_mode: LldMode, is_windows: bool) -> &'static str {
    static LLD_NO_THREADS: OnceLock<(&'static str, &'static str)> = OnceLock::new();

    let new_flags = ("/threads:1", "--threads=1");
    let old_flags = ("/no-threads", "--no-threads");

    let (windows_flag, other_flag) = LLD_NO_THREADS.get_or_init(|| {
        let newer_version = match lld_mode {
            LldMode::External => {
                let mut cmd = command("lld");
                cmd.arg("-flavor").arg("ld").arg("--version");
                let out = cmd.run_capture_stdout(builder).stdout();
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
    t!(std::fs::read_dir(dir), dir).next().is_none()
}

/// Extract the beta revision from the full version string.
///
/// The full version string looks like "a.b.c-beta.y". And we need to extract
/// the "y" part from the string.
pub fn extract_beta_rev(version: &str) -> Option<String> {
    let parts = version.splitn(2, "-beta.").collect::<Vec<_>>();
    parts.get(1).and_then(|s| s.find(' ').map(|p| s[..p].to_string()))
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
        match builder.config.lld_mode {
            LldMode::External => {
                args.push("-Zlinker-features=+lld".to_string());
                // FIXME(kobzol): remove this flag once MCP510 gets stabilized
                args.push("-Zunstable-options".to_string());
            }
            LldMode::SelfContained => {
                args.push("-Zlinker-features=+lld".to_string());
                args.push("-Clink-self-contained=+linker".to_string());
                // FIXME(kobzol): remove this flag once MCP510 gets stabilized
                args.push("-Zunstable-options".to_string());
            }
            LldMode::Unused => unreachable!(),
        };

        if matches!(lld_threads, LldThreads::No) {
            args.push(format!(
                "-Clink-arg=-Wl,{}",
                lld_flag_no_threads(builder, builder.config.lld_mode, target.is_windows())
            ));
        }
    }
    args
}

pub fn add_rustdoc_cargo_linker_args(
    cmd: &mut BootstrapCommand,
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
    use std::fmt::Write;

    input.as_ref().iter().fold(String::with_capacity(input.as_ref().len() * 2), |mut acc, &byte| {
        write!(&mut acc, "{byte:02x}").expect("Failed to write byte to the hex String.");
        acc
    })
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

/// Prepares `BootstrapCommand` that runs git inside the source directory if given.
///
/// Whenever a git invocation is needed, this function should be preferred over
/// manually building a git `BootstrapCommand`. This approach allows us to manage
/// bootstrap-specific needs/hacks from a single source, rather than applying them on next to every
/// git command creation, which is painful to ensure that the required change is applied
/// on each one of them correctly.
#[track_caller]
pub fn git(source_dir: Option<&Path>) -> BootstrapCommand {
    let mut git = command("git");

    if let Some(source_dir) = source_dir {
        git.current_dir(source_dir);
        // If we are running inside git (e.g. via a hook), `GIT_DIR` is set and takes precedence
        // over the current dir. Un-set it to make the current dir matter.
        git.env_remove("GIT_DIR");
        // Also un-set some other variables, to be on the safe side (based on cargo's
        // `fetch_with_cli`). In particular un-setting `GIT_INDEX_FILE` is required to fix some odd
        // misbehavior.
        git.env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_OBJECT_DIRECTORY")
            .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES");
    }

    git
}

/// Sets the file times for a given file at `path`.
pub fn set_file_times<P: AsRef<Path>>(path: P, times: fs::FileTimes) -> io::Result<()> {
    // Windows requires file to be writable to modify file times. But on Linux CI the file does not
    // need to be writable to modify file times and might be read-only.
    let f = if cfg!(windows) {
        fs::File::options().write(true).open(path)?
    } else {
        fs::File::open(path)?
    };
    f.set_times(times)
}
