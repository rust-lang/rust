//! Small helper program to install a git hook to automatically run
//! `x.py test tidy --bless` before each commit.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let root_path: PathBuf = env::args_os().nth(1).expect("need path to root of repo").into();
    let script_path: PathBuf = root_path.join("src/tools/install-git-hook/src/pre-commit.sh");
    let hook_path: PathBuf = root_path.join(".git/hooks/pre-commit");

    fs::copy(&script_path, &hook_path).expect(
        format!("Failed to copy pre-commit script to {}", &hook_path.to_string_lossy()).as_str(),
    );
}
