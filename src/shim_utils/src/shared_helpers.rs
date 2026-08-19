use std::env;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{self, BufRead};
use std::path::Path;
use std::process::Command;

/// Returns the environment variable name for the platform's dynamic library lookup path.
pub const fn dylib_path_var() -> &'static str {
    if cfg!(any(target_os = "windows", target_os = "cygwin")) {
        "PATH"
    } else if cfg!(target_vendor = "apple") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "haiku") {
        "LIBRARY_PATH"
    } else if cfg!(target_os = "aix") {
        "LIBPATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Returns the parsed dynamic library lookup paths for this platform.
pub fn dylib_path() -> Vec<std::path::PathBuf> {
    env::var_os(dylib_path_var())
        .map(|var| env::split_paths(&var).collect())
        .unwrap_or_default()
}

/// Returns the executable filename for the given target platform.
pub fn exe(name: &str, target: &str) -> String {
    // On Cygwin, the decision to append .exe or not is not as straightforward.
    // Executable files do actually have .exe extensions so on hosts other than
    // Cygwin it is necessary. But on a Cygwin host there is magic happening
    // that redirects requests for file X to file X.exe if it exists, and
    // furthermore /proc/self/exe (and thus std::env::current_exe) always
    // returns the name *without* the .exe extension. For comparisons against
    // that to match, we therefore do not append .exe for Cygwin targets on
    // a Cygwin host.
    let ext = if target.contains("windows")
        || (cfg!(not(target_os = "cygwin")) && target.contains("cygwin"))
    {
        ".exe"
    } else if target.contains("uefi") {
        ".efi"
    } else if target.contains("wasm") {
        ".wasm"
    } else {
        return name.to_string();
    };
    format!("{name}{ext}")
}

/// Parses the `RUSTC_VERBOSE` environment variable as a verbosity level.
/// Defaults to 0 if not set.
pub fn parse_rustc_verbose() -> usize {
    env::var("RUSTC_VERBOSE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Parses the `RUSTC_STAGE` environment variable.
/// Exits with code 101 if not set.
pub fn parse_rustc_stage() -> u32 {
    env::var("RUSTC_STAGE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            eprintln!("rustc shim: FATAL: RUSTC_STAGE was not set");
            eprintln!("rustc shim: NOTE: use `x.py build -vvv` to see all environment variables");
            std::process::exit(101);
        })
}

/// Writes the command invocation to a file if `DUMP_BOOTSTRAP_SHIMS` is set.
/// Replaces environment-specific paths with placeholders for portability.
pub fn maybe_dump(dump_name: &str, cmd: &Command) {
    let Ok(dump_dir) = env::var("DUMP_BOOTSTRAP_SHIMS") else {
        return;
    };

    let dump_file = format!("{dump_dir}/{dump_name}");
    let mut file = match OpenOptions::new().create(true).append(true).open(dump_file) {
        Ok(f) => f,
        Err(_) => return,
    };

    let mut cmd_dump = format!("{cmd:?}\n");
    if let Ok(val) = env::var("BUILD_OUT") {
        cmd_dump = cmd_dump.replace(&val, "${BUILD_OUT}");
    }
    if let Ok(val) = env::var("CARGO_HOME") {
        cmd_dump = cmd_dump.replace(&val, "${CARGO_HOME}");
    }

    let _ = file.write_all(cmd_dump.as_bytes());
}

/// Finds a key in the argument list and returns its value.
/// Supports both `key=value` and `key value` formats.
pub fn parse_value_from_args(args: &[OsString], key: &str) -> Option<&str> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let arg_str = arg.to_str()?;

        if let Some(value) = arg_str.strip_prefix(&format!("{key}=")) {
            return Some(value);
        } else if arg_str == key {
            return iter.next().and_then(|v| v.to_str());
        }
    }
    None
}

/// Collects command-line arguments, expanding `@argfile` references.
pub fn collect_args() -> Vec<OsString> {
    let mut args = Vec::new();
    for arg in env::args_os().skip(1) {
        if let Some(path) = arg.to_str().and_then(|s| s.strip_prefix('@')) {
            args.extend(args_from_argfile(Path::new(path)));
        } else {
            args.push(arg);
        }
    }
    args
}

/// Reads arguments from a file, one per line.
fn args_from_argfile(path: &Path) -> Vec<OsString> {
    let file = std::fs::File::open(path).expect("read args from argfile {path:?}");
    io::BufReader::new(file)
        .lines()
        .filter_map(|line| line.ok().map(OsString::from))
        .collect()
}
