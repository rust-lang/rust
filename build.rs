extern crate walkdir;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use walkdir::WalkDir;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("git_info.rs");
    let mut f = File::create(&dest_path).unwrap();

    writeln!(f,
             "const COMMIT_HASH: Option<&'static str> = {:?};",
             git_head_sha1())
        .unwrap();
    writeln!(f,
             "const WORKTREE_CLEAN: Option<bool> = {:?};",
             git_tree_is_clean())
        .unwrap();

    // cargo:rerun-if-changed requires one entry per individual file.
    for entry in WalkDir::new("src") {
        let entry = entry.unwrap();
        println!("cargo:rerun-if-changed={}", entry.path().display());
    }
}

// Returns `None` if git is not available.
fn git_head_sha1() -> Option<String> {
    Command::new("git")
        .arg("rev-parse")
        .arg("--short")
        .arg("HEAD")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|mut s| {
            let len = s.trim_right().len();
            s.truncate(len);
            s
        })
}

// Returns `None` if git is not available.
fn git_tree_is_clean() -> Option<bool> {
    Command::new("git")
        .arg("status")
        .arg("--porcelain")
        .arg("--untracked-files=no")
        .output()
        .ok()
        .map(|o| o.stdout.is_empty())
}
