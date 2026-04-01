use std::env;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Only check .git/HEAD dirty status if it exists - doing so when
    // building dependent crates may lead to false positives and rebuilds
    if Path::new(".git/HEAD").exists() {
        println!("cargo:rerun-if-changed=.git/HEAD");
    }

    println!("cargo:rerun-if-env-changed=CFG_RELEASE_CHANNEL");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    File::create(out_dir.join("commit-info.txt"))
        .unwrap()
        .write_all(commit_info().as_bytes())
        .unwrap();
}

// Try to get hash and date of the last commit on a best effort basis. If anything goes wrong
// (git not installed or if this is not a git repository) just return an empty string.
fn commit_info() -> String {
    match (channel(), commit_hash(), commit_date()) {
        (channel, Some(hash), Some(date)) => format!("{} ({} {})", channel, hash, date),
        _ => String::new(),
    }
}

fn channel() -> String {
    if let Ok(channel) = env::var("CFG_RELEASE_CHANNEL") {
        channel
    } else {
        "nightly".to_owned()
    }
}

fn commit_hash() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    let mut stdout = output.status.success().then_some(output.stdout)?;
    stdout.truncate(10);
    String::from_utf8(stdout).ok()
}

fn commit_date() -> Option<String> {
    let output = Command::new("git")
        .args(["log", "-1", "--date=short", "--pretty=format:%cd"])
        .output()
        .ok()?;
    let stdout = output.status.success().then_some(output.stdout)?;
    String::from_utf8(stdout).ok()
}
