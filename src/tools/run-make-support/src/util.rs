use std::path::PathBuf;

use crate::command::{Command, CompletedProcess};
use crate::env::env_var;
use crate::path_helpers::cwd;

/// If a given [`Command`] failed (as indicated by its [`CompletedProcess`]), verbose print the
/// executed command, failure location, output status and stdout/stderr, and abort the process with
/// exit code `1`.
pub(crate) fn handle_failed_output(
    cmd: &Command,
    output: CompletedProcess,
    caller_line_number: u32,
) -> ! {
    if output.status().success() {
        eprintln!("command unexpectedly succeeded at line {caller_line_number}");
    } else {
        eprintln!("command failed at line {caller_line_number}");
    }
    eprintln!("{cmd:?}");
    eprintln!("output status: `{}`", output.status());
    eprintln!("=== STDOUT ===\n{}\n\n", output.stdout_utf8());
    eprintln!("=== STDERR ===\n{}\n\n", output.stderr_utf8());
    std::process::exit(1)
}

/// Set the runtime library paths as needed for running the host compilers (rustc/rustdoc/etc).
pub(crate) fn set_host_compiler_dylib_path(cmd: &mut Command) {
    let ld_lib_path_envvar = env_var("LD_LIB_PATH_ENVVAR");
    cmd.env(&ld_lib_path_envvar, {
        let mut paths = vec![];
        paths.push(cwd());
        paths.push(PathBuf::from(env_var("HOST_RUSTC_DYLIB_PATH")));
        for p in std::env::split_paths(&env_var(&ld_lib_path_envvar)) {
            paths.push(p.to_path_buf());
        }
        std::env::join_paths(paths.iter()).unwrap()
    });
}
