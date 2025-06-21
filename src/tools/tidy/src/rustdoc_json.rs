//! Tidy check to ensure that `FORMAT_VERSION` was correctly updated if `rustdoc-json-types` was
//! updated as well.

use std::path::Path;

const RUSTDOC_JSON_TYPES: &str = "src/rustdoc-json-types";

pub fn check(src_path: &Path, ci_info: &crate::CiInfo, bad: &mut bool) {
    println!("Checking tidy rustdoc_json...");
    let Some(base_commit) = &ci_info.base_commit else {
        eprintln!("No base commit, skipping rustdoc_json check");
        return;
    };

    // First we check that `src/rustdoc-json-types` was modified.
    match crate::git_diff(&base_commit, "--name-status") {
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
    match crate::git_diff(&base_commit, src_path.join("rustdoc-json-types")) {
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
        }
        None => {
            *bad = true;
            eprintln!("error: failed to run `git diff` in rustdoc_json check");
        }
    }
}
