use std::fs;
use std::path::Path;

use super::verify_inside_clippy_dir;

/// Rusts setup uses `git rev-parse --git-common-dir` to get the root directory of the repo.
/// I've decided against this for the sake of simplicity and to make sure that it doesn't install
/// the hook if `clippy_dev` would be used in the rust tree. The hook also references this tool
/// for formatting and should therefore only be used in a normal clone of clippy
const REPO_GIT_DIR: &str = ".git";
const HOOK_SOURCE_FILE: &str = "util/etc/pre-commit.sh";
const HOOK_TARGET_FILE: &str = ".git/hooks/pre-commit";

pub fn install_hook(force_override: bool) {
    if !check_precondition(force_override) {
        return;
    }

    // So a little bit of a funny story. Git on unix requires the pre-commit file
    // to have the `execute` permission to be set. The Rust functions for modifying
    // these flags doesn't seem to work when executed with normal user permissions.
    //
    // However, there is a little hack that is also being used by Rust itself in their
    // setup script. Git saves the `execute` flag when syncing files. This means
    // that we can check in a file with execution permissions and the sync it to create
    // a file with the flag set. We then copy this file here. The copy function will also
    // include the `execute` permission.
    match fs::copy(HOOK_SOURCE_FILE, HOOK_TARGET_FILE) {
        Ok(_) => {
            println!("info: the hook can be removed with `cargo dev remove git-hook`");
            println!("git hook successfully installed");
        },
        Err(err) => eprintln!("error: unable to copy `{HOOK_SOURCE_FILE}` to `{HOOK_TARGET_FILE}` ({err})"),
    }
}

fn check_precondition(force_override: bool) -> bool {
    if !verify_inside_clippy_dir() {
        return false;
    }

    // Make sure that we can find the git repository
    let git_path = Path::new(REPO_GIT_DIR);
    if !git_path.exists() || !git_path.is_dir() {
        eprintln!("error: clippy_dev was unable to find the `.git` directory");
        return false;
    }

    // Make sure that we don't override an existing hook by accident
    let path = Path::new(HOOK_TARGET_FILE);
    if path.exists() {
        if force_override {
            return delete_git_hook_file(path);
        }

        eprintln!("error: there is already a pre-commit hook installed");
        println!("info: use the `--force-override` flag to override the existing hook");
        return false;
    }

    true
}

pub fn remove_hook() {
    let path = Path::new(HOOK_TARGET_FILE);
    if path.exists() {
        if delete_git_hook_file(path) {
            println!("git hook successfully removed");
        }
    } else {
        println!("no pre-commit hook was found");
    }
}

fn delete_git_hook_file(path: &Path) -> bool {
    if let Err(err) = fs::remove_file(path) {
        eprintln!("error: unable to delete existing pre-commit git hook ({err})");
        false
    } else {
        true
    }
}
