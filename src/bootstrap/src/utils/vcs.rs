use std::path::Path;

use crate::utils::exec::ExecutionContext;
use crate::utils::helpers::{git, jj, t};

/// Returns a hash of the sources, using (if available) jj or (as a fallback) git.
///
/// On git, this returns the hash of the current commit if the working tree is clean, or the tree
/// hash otherwise.
///
/// On jj, this returns the hash of the current commit (`@`).
pub(crate) fn git_or_jj_hash(source_dir: Option<&Path>) -> String {
    let exec_ctx = ExecutionContext::new(0, true);

    // Determine whether we should use git or jj.
    let output = jj(source_dir).allow_failure().arg("root").run_capture(&exec_ctx);

    if output.is_failure() {
        let output = git(source_dir)
            .allow_failure()
            .args(["diff-index", "HEAD", "--exit-code"])
            .run_capture(&exec_ctx);

        if output.is_success() {
            // `git diff-index HEAD` exits with 0 means working tree is clean.
            t!(git(source_dir)
                .args(["rev-parse", "--short=12", "HEAD"])
                .run_capture_stdout(exec_ctx)
                .stdout_if_ok()
                .ok_or("failed to get git tree hash"))
            .trim()
            .to_string()
        } else {
            t!(git(source_dir)
                .arg("write-tree")
                .run_capture_stdout(exec_ctx)
                .stdout_if_ok()
                .ok_or("failed to get git tree hash"))
            .trim()
            .to_string()
        }
    } else {
        t!(jj(source_dir)
            .arg("log")
            .args(["--revision", "@"])
            .arg("--no-graph")
            .args(["--template", "commit_id.short()"])
            .run_capture_stdout(exec_ctx)
            .stdout_if_ok()
            .ok_or("failed to get current jj commit hash"))
    }
}
