use std::process::ExitCode;

use build_helper::git::{GitConfig, get_git_modified_files};
use suggest_tests::get_suggestions;

fn main() -> ExitCode {
    let modified_files = get_git_modified_files(
        &GitConfig {
            nightly_branch: &env("SUGGEST_TESTS_NIGHTLY_BRANCH"),
            git_merge_commit_email: &env("SUGGEST_TESTS_MERGE_COMMIT_EMAIL"),
        },
        None,
        &Vec::new(),
    );
    let modified_files = match modified_files {
        Ok(files) => files,
        Err(err) => {
            eprintln!("Could not get modified files from git: \"{err}\"");
            return ExitCode::FAILURE;
        }
    };

    let suggestions = get_suggestions(&modified_files);

    for sug in &suggestions {
        println!("{sug}");
    }

    ExitCode::SUCCESS
}

fn env(key: &str) -> String {
    match std::env::var(key) {
        Ok(var) => var,
        Err(err) => {
            eprintln!("suggest-tests: failed to read environment variable {key}: {err}");
            std::process::exit(1);
        }
    }
}
