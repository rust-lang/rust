//! Tidy check to ensure that `FORMAT_VERSION` was correctly updated if `rustdoc-json-types` was
//! updated as well.

use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;
use std::str::FromStr;

use build_helper::ci::CiEnv;
use build_helper::git::{GitConfig, get_closest_upstream_commit};
use build_helper::stage0_parser::parse_stage0_file;

const RUSTDOC_JSON_TYPES: &str = "src/rustdoc-json-types";

fn git_diff<S: AsRef<OsStr>>(base_commit: &str, extra_arg: S) -> Option<String> {
    let output = Command::new("git").arg("diff").arg(base_commit).arg(extra_arg).output().ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into())
}

fn error_if_in_ci(ci_env: CiEnv, msg: &str, bad: &mut bool) {
    if ci_env.is_running_in_ci() {
        *bad = true;
        eprintln!("error in `rustdoc_json` tidy check: {msg}");
    } else {
        eprintln!("{msg}. Skipping `rustdoc_json` tidy check");
    }
}

pub fn check(src_path: &Path, bad: &mut bool) {
    println!("Checking tidy rustdoc_json...");
    let stage0 = parse_stage0_file();
    let ci_env = CiEnv::current();
    let base_commit = match get_closest_upstream_commit(
        None,
        &GitConfig {
            nightly_branch: &stage0.config.nightly_branch,
            git_merge_commit_email: &stage0.config.git_merge_commit_email,
        },
        ci_env,
    ) {
        Ok(Some(commit)) => commit,
        Ok(None) => {
            error_if_in_ci(ci_env, "no base commit found", bad);
            return;
        }
        Err(error) => {
            error_if_in_ci(ci_env, &format!("failed to retrieve base commit: {error}"), bad);
            return;
        }
    };

    // First we check that `src/rustdoc-json-types` was modified.
    match git_diff(&base_commit, "--name-status") {
        Some(output) => {
            if !output
                .lines()
                .any(|line| line.starts_with("M") && line.contains(RUSTDOC_JSON_TYPES))
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
            let mut new_version = None;
            let mut old_version = None;
            for line in output.lines() {
                if line.starts_with("+pub const FORMAT_VERSION: u32 =") {
                    format_version_updated = true;
                    new_version = line
                        .split('=')
                        .nth(1)
                        .and_then(|s| s.trim().split(';').next())
                        .and_then(|s| u32::from_str(s.trim()).ok());
                } else if line.starts_with("-pub const FORMAT_VERSION: u32 =") {
                    old_version = line
                        .split('=')
                        .nth(1)
                        .and_then(|s| s.trim().split(';').next())
                        .and_then(|s| u32::from_str(s.trim()).ok());
                } else if line.starts_with("+// Latest feature:") {
                    latest_feature_comment_updated = true;
                }
            }
            if format_version_updated != latest_feature_comment_updated {
                *bad = true;
                if latest_feature_comment_updated {
                    eprintln!(
                        "error in `rustdoc_json` tidy check: `Latest feature` comment was updated \
                         whereas `FORMAT_VERSION` wasn't in `{RUSTDOC_JSON_TYPES}/lib.rs`"
                    );
                } else {
                    eprintln!(
                        "error in `rustdoc_json` tidy check: `Latest feature` comment was not \
                         updated whereas `FORMAT_VERSION` was in `{RUSTDOC_JSON_TYPES}/lib.rs`"
                    );
                }
            }
            match (new_version, old_version) {
                (Some(new_version), Some(old_version)) if new_version != old_version + 1 => {
                    *bad = true;
                    eprintln!(
                        "error in `rustdoc_json` tidy check: invalid `FORMAT_VERSION` increase in \
                         `{RUSTDOC_JSON_TYPES}/lib.rs`, should be `{}`, found `{new_version}`",
                        old_version + 1,
                    );
                }
                _ => {}
            }
        }
        None => {
            *bad = true;
            eprintln!("error: failed to run `git diff` in rustdoc_json check");
        }
    }
}
