use std::path::PathBuf;
use std::process::Command;
use std::str;

#[must_use]
fn get_output(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    let mut stdout = output.status.success().then_some(output.stdout)?;
    // Remove trailing newlines.
    while stdout.last().copied() == Some(b'\n') {
        stdout.pop();
    }
    String::from_utf8(stdout).ok()
}

#[must_use]
pub fn rerun_if_git_changes() -> Option<()> {
    // Make sure we get rerun when the git commit changes.
    // We want to watch two files: HEAD, which tracks which branch we are on,
    // and the file for that branch that tracks which commit is checked out.

    // First, find the `HEAD` file. This should work even with worktrees.
    let git_head_file = PathBuf::from(get_output("git", &["rev-parse", "--git-path", "HEAD"])?);
    if git_head_file.exists() {
        println!("cargo::rerun-if-changed={}", git_head_file.display());
    }

    // Determine the name of the current ref.
    // This will quit if HEAD is detached.
    let git_head_ref = get_output("git", &["symbolic-ref", "-q", "HEAD"])?;
    // Ask git where this ref is stored.
    let git_head_ref_file =
        PathBuf::from(get_output("git", &["rev-parse", "--git-path", &git_head_ref])?);
    // If this ref is packed, the file does not exist. However, the checked-out branch is never (?)
    // packed, so we should always be able to find this file.
    if git_head_ref_file.exists() {
        println!("cargo::rerun-if-changed={}", git_head_ref_file.display());
    }

    Some(())
}

#[must_use]
pub fn get_commit_hash() -> Option<String> {
    let mut stdout = get_output("git", &["rev-parse", "HEAD"])?;
    stdout.truncate(10);
    Some(stdout)
}

#[must_use]
pub fn get_commit_date() -> Option<String> {
    get_output("git", &["log", "-1", "--date=short", "--pretty=format:%cd"])
}

fn main() {
    // Don't rebuild miri when nothing changed.
    println!("cargo:rerun-if-changed=build.rs");
    // gather version info
    let _ = rerun_if_git_changes();
    println!("cargo:rustc-env=GIT_COMMIT_HASH={}", get_commit_hash().unwrap_or_default());
    println!("cargo:rustc-env=GIT_COMMIT_DATE={}", get_commit_date().unwrap_or_default());
}
