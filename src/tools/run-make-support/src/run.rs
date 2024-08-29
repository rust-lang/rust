use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::{env, panic};

use crate::command::{Command, CompletedProcess};
use crate::util::{handle_failed_output, set_host_rpath};
use crate::{cwd, env_var, is_windows};

#[track_caller]
fn run_common(name: &str, args: Option<&[&str]>) -> Command {
    let mut bin_path = PathBuf::new();
    bin_path.push(cwd());
    bin_path.push(name);
    let ld_lib_path_envvar = env_var("LD_LIB_PATH_ENVVAR");
    let mut cmd = Command::new(bin_path);
    if let Some(args) = args {
        for arg in args {
            cmd.arg(arg);
        }
    }
    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(cwd());
        for p in env::split_paths(&env_var("TARGET_RPATH_ENV")) {
            paths.push(p.to_path_buf());
        }
        for p in env::split_paths(&env_var(&ld_lib_path_envvar)) {
            paths.push(p.to_path_buf());
        }
        env::join_paths(paths.iter()).unwrap()
    });
    cmd.env("LC_ALL", "C"); // force english locale

    if is_windows() {
        let mut paths = vec![];
        for p in env::split_paths(&std::env::var("PATH").unwrap_or(String::new())) {
            paths.push(p.to_path_buf());
        }
        paths.push(Path::new(&env_var("TARGET_RPATH_DIR")).to_path_buf());
        cmd.env("PATH", env::join_paths(paths.iter()).unwrap());
    }

    cmd
}

/// Run a built binary and make sure it succeeds.
#[track_caller]
pub fn run(name: &str) -> CompletedProcess {
    let caller = panic::Location::caller();
    let mut cmd = run_common(name, None);
    let output = cmd.run();
    if !output.status().success() {
        handle_failed_output(&cmd, output, caller.line());
    }
    output
}

/// Run a built binary with one or more argument(s) and make sure it succeeds.
#[track_caller]
pub fn run_with_args(name: &str, args: &[&str]) -> CompletedProcess {
    let caller = panic::Location::caller();
    let mut cmd = run_common(name, Some(args));
    let output = cmd.run();
    if !output.status().success() {
        handle_failed_output(&cmd, output, caller.line());
    }
    output
}

/// Run a built binary and make sure it fails.
#[track_caller]
pub fn run_fail(name: &str) -> CompletedProcess {
    let caller = panic::Location::caller();
    let mut cmd = run_common(name, None);
    let output = cmd.run_fail();
    if output.status().success() {
        handle_failed_output(&cmd, output, caller.line());
    }
    output
}

/// Create a new custom [`Command`]. This should be preferred to creating [`std::process::Command`]
/// directly.
#[track_caller]
pub fn cmd<S: AsRef<OsStr>>(program: S) -> Command {
    let mut command = Command::new(program);
    set_host_rpath(&mut command);
    command.env("LC_ALL", "C"); // force english locale
    command
}
