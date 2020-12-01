use std::ffi::{OsStr, OsString};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

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

/// Reads an environment variable and adds it to dependencies.
/// Supposed to be used for all variables except those set for build scripts by cargo
/// <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts>
pub fn tracked_env_var_os<K: AsRef<OsStr> + Display>(key: K) -> Option<OsString> {
    println!("cargo:rerun-if-env-changed={}", key);
    env::var_os(key)
}

// Because Cargo adds the compiler's dylib path to our library search path, llvm-config may
// break: the dylib path for the compiler, as of this writing, contains a copy of the LLVM
// shared library, which means that when our freshly built llvm-config goes to load it's
// associated LLVM, it actually loads the compiler's LLVM. In particular when building the first
// compiler (i.e., in stage 0) that's a problem, as the compiler's LLVM is likely different from
// the one we want to use. As such, we restore the environment to what bootstrap saw. This isn't
// perfect -- we might actually want to see something from Cargo's added library paths -- but
// for now it works.
pub fn restore_library_path() {
    let key = tracked_env_var_os("REAL_LIBRARY_PATH_VAR").expect("REAL_LIBRARY_PATH_VAR");
    if let Some(env) = tracked_env_var_os("REAL_LIBRARY_PATH") {
        env::set_var(&key, &env);
    } else {
        env::remove_var(&key);
    }
}

/// Run the command, printing what we are running.
pub fn run_verbose(cmd: &mut Command) {
    println!("running: {:?}", cmd);
    run(cmd);
}

pub fn run(cmd: &mut Command) {
    if !try_run(cmd) {
        std::process::exit(1);
    }
}

pub fn try_run(cmd: &mut Command) -> bool {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !status.success() {
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
        std::process::exit(1);
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

pub fn gnu_target(target: &str) -> &str {
    match target {
        "i686-pc-windows-msvc" => "i686-pc-win32",
        "x86_64-pc-windows-msvc" => "x86_64-pc-win32",
        "i686-pc-windows-gnu" => "i686-w64-mingw32",
        "x86_64-pc-windows-gnu" => "x86_64-w64-mingw32",
        s => s,
    }
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

pub fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| &*e.file_name() != ".git")
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
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
    println!("\n\n{}\n\n", s);
    std::process::exit(1);
}
