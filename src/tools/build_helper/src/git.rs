use std::{path::Path, process::Command};

/// Finds the remote for rust-lang/rust.
/// For example for these remotes it will return `upstream`.
/// ```text
/// origin  https://github.com/Nilstrieb/rust.git (fetch)
/// origin  https://github.com/Nilstrieb/rust.git (push)
/// upstream        https://github.com/rust-lang/rust (fetch)
/// upstream        https://github.com/rust-lang/rust (push)
/// ```
pub fn get_rust_lang_rust_remote(git_dir: Option<&Path>) -> Result<String, String> {
    let mut git = Command::new("git");
    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }
    git.args(["config", "--local", "--get-regex", "remote\\..*\\.url"]);

    let output = git.output().map_err(|err| format!("{err:?}"))?;
    if !output.status.success() {
        return Err("failed to execute git config command".to_owned());
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

pub fn rev_exists(rev: &str, git_dir: Option<&Path>) -> Result<bool, String> {
    let mut git = Command::new("git");
    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }
    git.args(["rev-parse", rev]);
    let output = git.output().map_err(|err| format!("{err:?}"))?;

    match output.status.code() {
        Some(0) => Ok(true),
        Some(128) => Ok(false),
        None => {
            return Err(format!(
                "git didn't exit properly: {}",
                String::from_utf8(output.stderr).map_err(|err| format!("{err:?}"))?
            ));
        }
        Some(code) => {
            return Err(format!(
                "git command exited with status code: {code}: {}",
                String::from_utf8(output.stderr).map_err(|err| format!("{err:?}"))?
            ));
        }
    }
}

/// Returns the master branch from which we can take diffs to see changes.
/// This will usually be rust-lang/rust master, but sometimes this might not exist.
/// This could be because the user is updating their forked master branch using the GitHub UI
/// and therefore doesn't need an upstream master branch checked out.
/// We will then fall back to origin/master in the hope that at least this exists.
pub fn updated_master_branch(git_dir: Option<&Path>) -> Result<String, String> {
    let upstream_remote = get_rust_lang_rust_remote(git_dir)?;
    let upstream_master = format!("{upstream_remote}/master");
    if rev_exists(&upstream_master, git_dir)? {
        return Ok(upstream_master);
    }

    // We could implement smarter logic here in the future.
    Ok("origin/master".into())
}
