//! Script to invoke the bundled rust-lld with the correct flavor.
//!
//! lld supports multiple command line interfaces. If `-flavor <flavor>` are passed as the first
//! two arguments the `<flavor>` command line interface is used to process the remaining arguments.
//! If no `-flavor` argument is present the flavor is determined by the executable name.
//!
//! In Rust with `-Z gcc-ld=lld` we have gcc or clang invoke rust-lld. Since there is no way to
//! make gcc/clang pass `-flavor <flavor>` as the first two arguments in the linker invocation
//! and since Windows does not support symbolic links for files this wrapper is used in place of a
//! symbolic link. It execs `../rust-lld -flavor <flavor>` by propagating the flavor argument
//! obtained from the wrapper's name as the first two arguments.
//! On Windows it spawns a `..\rust-lld.exe` child process.

use std::env::{self, consts::EXE_SUFFIX};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::process;

trait UnwrapOrExitWith<T> {
    fn unwrap_or_exit_with(self, context: &str) -> T;
}

impl<T> UnwrapOrExitWith<T> for Option<T> {
    fn unwrap_or_exit_with(self, context: &str) -> T {
        self.unwrap_or_else(|| {
            eprintln!("lld-wrapper: {}", context);
            process::exit(1);
        })
    }
}

impl<T, E: Display> UnwrapOrExitWith<T> for Result<T, E> {
    fn unwrap_or_exit_with(self, context: &str) -> T {
        self.unwrap_or_else(|err| {
            eprintln!("lld-wrapper: {}: {}", context, err);
            process::exit(1);
        })
    }
}

/// Returns the path to rust-lld in the parent directory.
///
/// Exits if the parent directory cannot be determined.
fn get_rust_lld_path(current_exe_path: &Path) -> PathBuf {
    let mut rust_lld_exe_name = "rust-lld".to_owned();
    rust_lld_exe_name.push_str(EXE_SUFFIX);
    let mut rust_lld_path = current_exe_path
        .parent()
        .unwrap_or_exit_with("directory containing current executable could not be determined")
        .parent()
        .unwrap_or_exit_with("parent directory could not be determined")
        .to_owned();
    rust_lld_path.push(rust_lld_exe_name);
    rust_lld_path
}

/// Extract LLD flavor name from the lld-wrapper executable name.
fn get_lld_flavor(current_exe_path: &Path) -> Result<&'static str, String> {
    let file = current_exe_path.file_name();
    let stem = file.and_then(|s| s.to_str()).map(|s| s.trim_end_matches(EXE_SUFFIX));
    Ok(match stem {
        Some("ld.lld") => "gnu",
        Some("ld64.lld") => "darwin",
        Some("lld-link") => "link",
        Some("wasm-ld") => "wasm",
        _ => return Err(format!("{:?}", file)),
    })
}

/// Returns the command for invoking rust-lld with the correct flavor.
/// LLD only accepts the flavor argument at the first two arguments, so pass it there.
///
/// Exits on error.
fn get_rust_lld_command(current_exe_path: &Path) -> process::Command {
    let rust_lld_path = get_rust_lld_path(current_exe_path);
    let mut command = process::Command::new(rust_lld_path);

    let flavor =
        get_lld_flavor(current_exe_path).unwrap_or_exit_with("executable has unexpected name");

    command.arg("-flavor");
    command.arg(flavor);
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
    // Windows has no exec(), spawn a child process and wait for it.
    let exit_status = command.status().unwrap_or_exit_with("error running rust-lld child process");
    let code = exit_status
        .code()
        .ok_or(exit_status)
        .unwrap_or_exit_with("rust-lld child process exited with error");
    // Return the original lld exit code.
    process::exit(code);
}

fn main() {
    let current_exe_path =
        env::current_exe().unwrap_or_exit_with("could not get the path of the current executable");

    exec_lld(get_rust_lld_command(current_exe_path.as_ref()));
}
