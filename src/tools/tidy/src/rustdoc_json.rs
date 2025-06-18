//! Tidy check to ensure that `FORMAT_VERSION` was correctly updated if `rustdoc-json-types` was
//! updated as well.

use std::process::Command;

fn git_diff(base_commit: &str, extra_arg: &str) -> Option<String> {
    let output = Command::new("git").arg("diff").arg(base_commit).arg(extra_arg).output().ok()?;
    Some(String::from_utf8_lossy(&output.stdout).into())
}

pub fn check(bad: &mut bool) {
    let Ok(base_commit) = std::env::var("BASE_COMMIT") else {
        // Not in CI so nothing we can check here.
        println!("not checking rustdoc JSON `FORMAT_VERSION` update");
        return;
    };

    // First we check that `src/rustdoc-json-types` was modified.
    match git_diff(&base_commit, "--name-status") {
        Some(output) => {
            if !output
                .lines()
                .any(|line| line.starts_with("M") && line.contains("src/rustdoc-json-types"))
            {
                // `rustdoc-json-types` was not modified so nothing more to check here.
                return;
            }
        }
        None => {
            *bad = true;
            eprintln!("Failed to run `git diff`");
            return;
        }
    }
    // Then we check that if `FORMAT_VERSION` was updated, the `Latest feature:` was also updated.
    match git_diff(&base_commit, "src/rustdoc-json-types") {
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
                        "`Latest feature` comment was updated whereas `FORMAT_VERSION` wasn't"
                    );
                } else {
                    eprintln!(
                        "`Latest feature` comment was not updated whereas `FORMAT_VERSION` was"
                    );
                }
            }
        }
        None => {
            *bad = true;
            eprintln!("Failed to run `git diff`");
            return;
        }
    }
}
