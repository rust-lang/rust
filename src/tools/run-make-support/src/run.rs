use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use crate::is_windows;

use super::handle_failed_output;

fn run_common(name: &str) -> (Command, Output) {
    let mut bin_path = PathBuf::new();
    bin_path.push(env::var("TMPDIR").unwrap());
    bin_path.push(name);
    let ld_lib_path_envvar = env::var("LD_LIB_PATH_ENVVAR").unwrap();
    let mut cmd = Command::new(bin_path);
    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(PathBuf::from(env::var("TMPDIR").unwrap()));
        for p in env::split_paths(&env::var("TARGET_RPATH_ENV").unwrap()) {
            paths.push(p.to_path_buf());
        }
        for p in env::split_paths(&env::var(&ld_lib_path_envvar).unwrap()) {
            paths.push(p.to_path_buf());
        }
        env::join_paths(paths.iter()).unwrap()
    });

    if is_windows() {
        let mut paths = vec![];
        for p in env::split_paths(&std::env::var("PATH").unwrap_or(String::new())) {
            paths.push(p.to_path_buf());
        }
        paths.push(Path::new(&std::env::var("TARGET_RPATH_DIR").unwrap()).to_path_buf());
        cmd.env("PATH", env::join_paths(paths.iter()).unwrap());
    }

    let output = cmd.output().unwrap();
    (cmd, output)
}

/// Run a built binary and make sure it succeeds.
#[track_caller]
pub fn run(name: &str) -> Output {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let (cmd, output) = run_common(name);
    if !output.status.success() {
        handle_failed_output(&cmd, output, caller_line_number);
    }
    output
}

/// Run a built binary and make sure it fails.
#[track_caller]
pub fn run_fail(name: &str) -> Output {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let (cmd, output) = run_common(name);
    if output.status.success() {
        handle_failed_output(&cmd, output, caller_line_number);
    }
    output
}
