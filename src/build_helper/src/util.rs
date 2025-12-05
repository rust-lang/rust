use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;

use crate::ci::CiEnv;

/// Invokes `build_helper::util::detail_exit` with `cfg!(test)`
///
/// This is a macro instead of a function so that it uses `cfg(test)` in the *calling* crate, not in build helper.
#[macro_export]
macro_rules! exit {
    ($code:expr) => {
        $crate::util::detail_exit($code, cfg!(test));
    };
}

/// If code is not 0 (successful exit status), exit status is 101 (rust's default error code.)
/// If `is_test` true and code is an error code, it will cause a panic.
pub fn detail_exit(code: i32, is_test: bool) -> ! {
    // if in test and code is an error code, panic with status code provided
    if is_test {
        panic!("status code: {code}");
    } else {
        // If we're in CI, print the current bootstrap invocation command, to make it easier to
        // figure out what exactly has failed.
        if CiEnv::is_ci() {
            // Skip the first argument, as it will be some absolute path to the bootstrap binary.
            let bootstrap_args =
                std::env::args().skip(1).map(|a| a.to_string()).collect::<Vec<_>>().join(" ");
            eprintln!("Bootstrap failed while executing `{bootstrap_args}`");
        }

        // otherwise, exit with provided status code
        std::process::exit(code);
    }
}

pub fn fail(s: &str) -> ! {
    eprintln!("\n\n{s}\n\n");
    detail_exit(1, cfg!(test));
}

pub fn try_run(cmd: &mut Command, print_cmd_on_fail: bool) -> Result<(), ()> {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {cmd:?}\nerror: {e}")),
    };
    if !status.success() {
        if print_cmd_on_fail {
            println!(
                "\n\ncommand did not execute successfully: {cmd:?}\n\
                 expected success, got: {status}\n\n"
            );
        }
        Err(())
    } else {
        Ok(())
    }
}

/// Returns the submodule paths from the `.gitmodules` file in the given directory.
pub fn parse_gitmodules(target_dir: &Path) -> Vec<String> {
    let gitmodules = target_dir.join(".gitmodules");
    assert!(gitmodules.exists(), "'{}' file is missing.", gitmodules.display());

    let file = File::open(gitmodules).unwrap();

    let mut submodules_paths = vec![];
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.starts_with("path") {
            let actual_path = line.split(' ').next_back().expect("Couldn't get value of path");
            submodules_paths.push(actual_path.to_owned());
        }
    }

    submodules_paths
}
