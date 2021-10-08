//! Script to invoke the bundled rust-lld with the correct flavor. The flavor is selected by
//! feature.
//!
//! lld supports multiple command line interfaces. If `-flavor <flavor>` are passed as the first
//! two arguments the `<flavor>` command line interface is used to process the remaining arguments.
//! If no `-flavor` argument is present the flavor is determined by the executable name.
//!
//! In Rust with `-Z gcc-ld=lld` we have gcc or clang invoke rust-lld. Since there is no way to
//! make gcc/clang pass `-flavor <flavor>` as the first two arguments in the linker invocation
//! and since Windows does not support symbolic links for files this wrapper is used in place of a
//! symblic link. It execs `../rust-lld -flavor ld` if the feature `ld` is enabled and
//! `../rust-lld -flavor ld64` if `ld64` is enabled. On Windows it spawns a `..\rust-lld.exe`
//! child process.

#[cfg(not(any(feature = "ld", feature = "ld64")))]
compile_error!("One of the features ld and ld64 must be enabled.");

#[cfg(all(feature = "ld", feature = "ld64"))]
compile_error!("Only one of the feature ld or ld64 can be enabled.");

#[cfg(feature = "ld")]
const FLAVOR: &str = "ld";

#[cfg(feature = "ld64")]
const FLAVOR: &str = "ld64";

use std::env;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::process;

trait ResultExt<T, E> {
    fn unwrap_or_exit_with(self, context: &str) -> T;
}

impl<T, E> ResultExt<T, E> for Result<T, E>
where
    E: Display,
{
    fn unwrap_or_exit_with(self, context: &str) -> T {
        match self {
            Ok(t) => t,
            Err(e) => {
                eprintln!("lld-wrapper: {}: {}", context, e);
                process::exit(1);
            }
        }
    }
}

trait OptionExt<T> {
    fn unwrap_or_exit_with(self, context: &str) -> T;
}

impl<T> OptionExt<T> for Option<T> {
    fn unwrap_or_exit_with(self, context: &str) -> T {
        match self {
            Some(t) => t,
            None => {
                eprintln!("lld-wrapper: {}", context);
                process::exit(1);
            }
        }
    }
}

/// Returns the path to rust-lld in the parent directory.
///
/// Exits if the parent directory cannot be determined.
fn get_rust_lld_path(current_exe_path: &Path) -> PathBuf {
    let mut rust_lld_exe_name = "rust-lld".to_owned();
    rust_lld_exe_name.push_str(env::consts::EXE_SUFFIX);
    let mut rust_lld_path = current_exe_path
        .parent()
        .unwrap_or_exit_with("directory containing current executable could not be determined")
        .parent()
        .unwrap_or_exit_with("parent directory could not be determined")
        .to_owned();
    rust_lld_path.push(rust_lld_exe_name);
    rust_lld_path
}

/// Returns the command for invoking rust-lld with the correct flavor.
///
/// Exits on error.
fn get_rust_lld_command(current_exe_path: &Path) -> process::Command {
    let rust_lld_path = get_rust_lld_path(current_exe_path);
    let mut command = process::Command::new(rust_lld_path);
    command.arg("-flavor");
    command.arg(FLAVOR);
    command.args(env::args_os().skip(1));
    command
}

#[cfg(unix)]
fn exec_lld(mut command: process::Command) {
    use std::os::unix::prelude::CommandExt;
    Result::<(), _>::Err(command.exec()).unwrap_or_exit_with("could not exec rust-lld");
    unreachable!("lld-wrapper: after exec without error");
}

#[cfg(not(unix))]
fn exec_lld(mut command: process::Command) {
    // Windows has no exec(), spawn a child process and wait for it
    let exit_status = command.status().unwrap_or_exit_with("error running rust-lld child process");
    if !exit_status.success() {
        match exit_status.code() {
            Some(code) => {
                // return the original lld exit code
                process::exit(code)
            }
            None => {
                eprintln!("lld-wrapper: rust-lld child process exited with error: {}", exit_status,);
                process::exit(1);
            }
        }
    }
}

fn main() {
    let current_exe_path =
        env::current_exe().unwrap_or_exit_with("could not get the path of the current executable");

    exec_lld(get_rust_lld_command(current_exe_path.as_ref()));
}
