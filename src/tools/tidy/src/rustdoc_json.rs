//! Tidy check to ensure that `FORMAT_VERSION` was correctly updated if `rustdoc-json-types` was
//! updated as well.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

use build_helper::ci::CiEnv;
use build_helper::git::{GitConfig, get_closest_upstream_commit};
use build_helper::stage0_parser::parse_stage0_file;

fn git_diff<S: AsRef<OsStr>>(base_commit: &str, extra_arg: S) -> Option<String> {
    let output = Command::new("git").arg("diff").arg(base_commit).arg(extra_arg).output().ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into())
}

pub fn check(src_path: &Path, bad: &mut bool) {
    println!("Checking tidy rustdoc_json...");
    let stage0 = parse_stage0_file();
    let base_commit = match get_closest_upstream_commit(
        None,
        &GitConfig {
            nightly_branch: &stage0.config.nightly_branch,
            git_merge_commit_email: &stage0.config.git_merge_commit_email,
        },
        CiEnv::current(),
    ) {
        Ok(Some(commit)) => commit,
        Ok(None) => {
            *bad = true;
            eprintln!("error: no base commit found for rustdoc_json check");
            return;
        }
        Err(error) => {
            *bad = true;
            eprintln!(
                "error: failed to retrieve base commit for rustdoc_json check because of `{error}`"
            );
            return;
        }
    };

    // First we check that `src/rustdoc-json-types` was modified.
    match git_diff(&base_commit, "--name-status") {
        Some(output) => {
            if !output
                .lines()
                .any(|line| line.starts_with("M") && line.contains("src/rustdoc-json-types"))
            {
                // `rustdoc-json-types` was not modified so nothing more to check here.
                println!("`rustdoc-json-types` was not modified.");
                return;
            }
        }
        None => {
            *bad = true;
            eprintln!("error: failed to run `git diff` in rustdoc_json check");
            return;
        }
    }
    // Then we check that if `FORMAT_VERSION` was updated, the `Latest feature:` was also updated.
    match git_diff(&base_commit, src_path.join("rustdoc-json-types")) {
        Some(output) => {
            let mut format_version_updated = false;
            let mut latest_feature_comment_updated = false;
            for line in output.lines() {
                if line.starts_with("+pub const FORMAT_VERSION: u32 =") {
                    format_version_updated = true;
                } else if line.starts_with("+// Latest feature:") {
                    latest_feature_comment_updated = true;
                }
            }
            if format_version_updated != latest_feature_comment_updated {
                *bad = true;
                if latest_feature_comment_updated {
                    eprintln!(
                        "error: `Latest feature` comment was updated whereas `FORMAT_VERSION` wasn't"
                    );
                } else {
                    eprintln!(
                        "error: `Latest feature` comment was not updated whereas `FORMAT_VERSION` was"
                    );
                }
            }
        }
        None => {
            *bad = true;
            eprintln!("error: failed to run `git diff` in rustdoc_json check");
        }
    }
}
