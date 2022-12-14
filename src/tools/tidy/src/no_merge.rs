//! This check makes sure that no accidental merge commits are introduced to the repository.
//! It forbids all merge commits that are not caused by rollups/bors or subtree syncs.

use std::process::Command;

macro_rules! try_unwrap_in_ci {
    ($expr:expr) => {
        match $expr {
            Ok(value) => value,
            Err(err) if CiEnv::is_ci() => {
                panic!("Encountered error while testing Git status: {:?}", err)
            }
            Err(_) => return,
        }
    };
}

pub fn check(_: (), bad: &mut bool) {
    let remote = try_unwrap_in_ci!(get_rust_lang_rust_remote());
    let merge_commits = try_unwrap_in_ci!(find_merge_commits(&remote));

    let mut bad_merge_commits = merge_commits.lines().filter(|commit| {
        !(
            // Bors is the ruler of merge commits.
            commit.starts_with("Auto merge of") || commit.starts_with("Rollup merge of")
        )
    });

    if let Some(merge) = bad_merge_commits.next() {
        tidy_error!(
            bad,
            "found a merge commit in the history: `{merge}`.
To resolve the issue, see this: https://rustc-dev-guide.rust-lang.org/git.html#i-made-a-merge-commit-by-accident.
If you're doing a subtree sync, add your tool to the list in the code that emitted this error."
        );
    }
}

/// Finds the remote for rust-lang/rust.
/// For example for these remotes it will return `upstream`.
/// ```text
/// origin  https://github.com/Nilstrieb/rust.git (fetch)
/// origin  https://github.com/Nilstrieb/rust.git (push)
/// upstream        https://github.com/rust-lang/rust (fetch)
/// upstream        https://github.com/rust-lang/rust (push)
/// ```
fn get_rust_lang_rust_remote() -> Result<String, String> {
    let mut git = Command::new("git");
    git.args(["config", "--local", "--get-regex", "remote\\..*\\.url"]);

    let output = git.output().map_err(|err| format!("{err:?}"))?;
    if !output.status.success() {
        return Err(format!(
            "failed to execute git config command: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|err| format!("{err:?}"))?;

    let rust_lang_remote = stdout
        .lines()
        .find(|remote| remote.contains("rust-lang"))
        .ok_or_else(|| "rust-lang/rust remote not found".to_owned())?;

    let remote_name =
        rust_lang_remote.split('.').nth(1).ok_or_else(|| "remote name not found".to_owned())?;
    Ok(remote_name.into())
}

/// Runs `git log --merges --format=%s $REMOTE/master..HEAD` and returns all commits
fn find_merge_commits(remote: &str) -> Result<String, String> {
    let mut git = Command::new("git");
    git.args([
        "log",
        "--merges",
        "--format=%s",
        &format!("{remote}/master..HEAD"),
        // Ignore subtree syncs. Add your new subtrees here.
        ":!src/tools/miri",
        ":!src/tools/rust-analyzer",
        ":!compiler/rustc_smir",
        ":!library/portable-simd",
        ":!compiler/rustc_codegen_gcc",
        ":!src/tools/rustfmt",
        ":!compiler/rustc_codegen_cranelift",
        ":!src/tools/clippy",
    ]);

    let output = git.output().map_err(|err| format!("{err:?}"))?;
    if !output.status.success() {
        return Err(format!(
            "failed to execute git log command: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|err| format!("{err:?}"))?;

    Ok(stdout)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum CiEnv {
    /// Not a CI environment.
    None,
    /// The Azure Pipelines environment, for Linux (including Docker), Windows, and macOS builds.
    AzurePipelines,
    /// The GitHub Actions environment, for Linux (including Docker), Windows and macOS builds.
    GitHubActions,
}

impl CiEnv {
    /// Obtains the current CI environment.
    pub fn current() -> CiEnv {
        if std::env::var("TF_BUILD").map_or(false, |e| e == "True") {
            CiEnv::AzurePipelines
        } else if std::env::var("GITHUB_ACTIONS").map_or(false, |e| e == "true") {
            CiEnv::GitHubActions
        } else {
            CiEnv::None
        }
    }

    pub fn is_ci() -> bool {
        Self::current() != CiEnv::None
    }
}
