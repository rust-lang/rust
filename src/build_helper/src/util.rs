use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

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
        panic!("status code: {}", code);
    } else {
        // otherwise,exit with provided status code
        std::process::exit(code);
    }
}

pub fn fail(s: &str) -> ! {
    eprintln!("\n\n{}\n\n", s);
    detail_exit(1, cfg!(test));
}

pub fn try_run(cmd: &mut Command, print_cmd_on_fail: bool) -> Result<(), ()> {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !status.success() {
        if print_cmd_on_fail {
            println!(
                "\n\ncommand did not execute successfully: {:?}\n\
                 expected success, got: {}\n\n",
                cmd, status
            );
        }
        Err(())
    } else {
        Ok(())
    }
}

/// Returns the submodule paths from the `.gitmodules` file in the given directory.
pub fn parse_gitmodules(target_dir: &Path) -> &[String] {
    static SUBMODULES_PATHS: OnceLock<Vec<String>> = OnceLock::new();
    let gitmodules = target_dir.join(".gitmodules");
    assert!(gitmodules.exists(), "'{}' file is missing.", gitmodules.display());

    let init_submodules_paths = || {
        let file = File::open(gitmodules).unwrap();

        let mut submodules_paths = vec![];
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let line = line.trim();
            if line.starts_with("path") {
                let actual_path = line.split(' ').last().expect("Couldn't get value of path");
                submodules_paths.push(actual_path.to_owned());
            }
        }

        submodules_paths
    };

    SUBMODULES_PATHS.get_or_init(|| init_submodules_paths())
}
