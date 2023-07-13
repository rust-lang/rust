use std::process::Stdio;
use std::{path::Path, process::Command};

/// Runs a command and returns the output
fn output_result(cmd: &mut Command) -> Result<String, String> {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => return Err(format!("failed to run command: {:?}: {}", cmd, e)),
    };
    if !output.status.success() {
        return Err(format!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}\n{}",
            cmd,
            output.status,
            String::from_utf8(output.stderr).map_err(|err| format!("{err:?}"))?
        ));
    }
    Ok(String::from_utf8(output.stdout).map_err(|err| format!("{err:?}"))?)
}

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
    let stdout = output_result(&mut git)?;

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

/// Returns the files that have been modified in the current branch compared to the master branch.
/// The `extensions` parameter can be used to filter the files by their extension.
/// If `extensions` is empty, all files will be returned.
pub fn get_git_modified_files(
    git_dir: Option<&Path>,
    extensions: &Vec<&str>,
) -> Result<Option<Vec<String>>, String> {
    let Ok(updated_master) = updated_master_branch(git_dir) else {
        return Ok(None);
    };

    let git = || {
        let mut git = Command::new("git");
        if let Some(git_dir) = git_dir {
            git.current_dir(git_dir);
        }
        git
    };

    let merge_base = output_result(git().arg("merge-base").arg(&updated_master).arg("HEAD"))?;
    let files = output_result(git().arg("diff-index").arg("--name-only").arg(merge_base.trim()))?
        .lines()
        .map(|s| s.trim().to_owned())
        .filter(|f| {
            Path::new(f).extension().map_or(false, |ext| {
                extensions.is_empty() || extensions.contains(&ext.to_str().unwrap())
            })
        })
        .collect();
    Ok(Some(files))
}

/// Returns the files that haven't been added to git yet.
pub fn get_git_untracked_files(git_dir: Option<&Path>) -> Result<Option<Vec<String>>, String> {
    let Ok(_updated_master) = updated_master_branch(git_dir) else {
        return Ok(None);
    };
    let mut git = Command::new("git");
    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }

    let files = output_result(git.arg("ls-files").arg("--others").arg("--exclude-standard"))?
        .lines()
        .map(|s| s.trim().to_owned())
        .collect();
    Ok(Some(files))
}
