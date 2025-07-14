use std::path::Path;
use std::process::{Command, Stdio};

use crate::ci::CiEnv;

#[derive(Debug)]
pub struct GitConfig<'a> {
    pub nightly_branch: &'a str,
    pub git_merge_commit_email: &'a str,
}

/// Runs a command and returns the output
pub fn output_result(cmd: &mut Command) -> Result<String, String> {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => return Err(format!("failed to run command: {cmd:?}: {e}")),
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
    String::from_utf8(output.stdout).map_err(|err| format!("{err:?}"))
}

/// Represents the result of checking whether a set of paths
/// have been modified locally or not.
#[derive(PartialEq, Debug, Clone)]
pub enum PathFreshness {
    /// Artifacts should be downloaded from this upstream commit,
    /// there are no local modifications.
    LastModifiedUpstream { upstream: String },
    /// There are local modifications to a certain set of paths.
    /// "Local" essentially means "not-upstream" here.
    /// `upstream` is the latest upstream merge commit that made modifications to the
    /// set of paths.
    HasLocalModifications { upstream: String },
    /// No upstream commit was found.
    /// This should not happen in most reasonable circumstances, but one never knows.
    MissingUpstream,
}

/// This function figures out if a set of paths was last modified upstream or
/// if there are some local modifications made to them.
/// It can be used to figure out if we should download artifacts from CI or rather
/// build them locally.
///
/// The function assumes that at least a single upstream bors merge commit is in the
/// local git history.
///
/// `target_paths` should be a non-empty slice of paths (git `pathspec`s) relative to `git_dir`
/// whose modifications would invalidate the artifact.
/// Each pathspec can also be a negative match, i.e. `:!foo`. This matches changes outside
/// the `foo` directory.
/// See <https://git-scm.com/docs/gitglossary#Documentation/gitglossary.txt-aiddefpathspecapathspec>
/// for how git `pathspec` works.
///
/// The function behaves differently in CI and outside CI.
///
/// - Outside CI, we want to find out if `target_paths` were modified in some local commit on
///   top of the latest upstream commit that is available in local git history.
///   If not, we try to find the most recent upstream commit (which we assume are commits
///   made by bors) that modified `target_paths`.
///   We don't want to simply take the latest master commit to avoid changing the output of
///   this function frequently after rebasing on the latest master branch even if `target_paths`
///   were not modified upstream in the meantime. In that case we would be redownloading CI
///   artifacts unnecessarily.
///
/// - In CI, we use a shallow clone of depth 2, i.e., we fetch only a single parent commit
///   (which will be the most recent bors merge commit) and do not have access
///   to the full git history. Luckily, we only need to distinguish between two situations:
///   1) The current PR made modifications to `target_paths`.
///      In that case, a build is typically necessary.
///   2) The current PR did not make modifications to `target_paths`.
///      In that case we simply take the latest upstream commit, because on CI there is no need to avoid
///      redownloading.
pub fn check_path_modifications(
    git_dir: &Path,
    config: &GitConfig<'_>,
    target_paths: &[&str],
    ci_env: CiEnv,
) -> Result<PathFreshness, String> {
    assert!(!target_paths.is_empty());
    for path in target_paths {
        assert!(Path::new(path.trim_start_matches(":!")).is_relative());
    }

    let upstream_sha = if matches!(ci_env, CiEnv::GitHubActions) {
        // Here the situation is different for PR CI and try/auto CI.
        // For PR CI, we have the following history:
        // <merge commit made by GitHub>
        // 1-N PR commits
        // upstream merge commit made by bors
        //
        // For try/auto CI, we have the following history:
        // <**non-upstream** merge commit made by bors>
        // 1-N PR commits
        // upstream merge commit made by bors
        //
        // But on both cases, HEAD should be a merge commit.
        // So if HEAD contains modifications of `target_paths`, our PR has modified
        // them. If not, we can use the only available upstream commit for downloading
        // artifacts.

        // Do not include HEAD, as it is never an upstream commit
        // If we do not find an upstream commit in CI, something is seriously wrong.
        Some(
            get_closest_upstream_commit(Some(git_dir), config, ci_env)?
                .expect("No upstream commit was found on CI"),
        )
    } else {
        // Outside CI, we want to find the most recent upstream commit that
        // modified the set of paths, to have an upstream reference that does not change
        // unnecessarily often.
        // However, if such commit is not found, we can fall back to the latest upstream commit
        let upstream_with_modifications =
            get_latest_upstream_commit_that_modified_files(git_dir, config, target_paths)?;
        match upstream_with_modifications {
            Some(sha) => Some(sha),
            None => get_closest_upstream_commit(Some(git_dir), config, ci_env)?,
        }
    };

    let Some(upstream_sha) = upstream_sha else {
        return Ok(PathFreshness::MissingUpstream);
    };

    // For local environments, we want to find out if something has changed
    // from the latest upstream commit.
    // However, that should be equivalent to checking if something has changed
    // from the latest upstream commit *that modified `target_paths`*, and
    // with this approach we do not need to invoke git an additional time.
    if has_changed_since(git_dir, &upstream_sha, target_paths) {
        Ok(PathFreshness::HasLocalModifications { upstream: upstream_sha })
    } else {
        Ok(PathFreshness::LastModifiedUpstream { upstream: upstream_sha })
    }
}

/// Returns true if any of the passed `paths` have changed since the `base` commit.
pub fn has_changed_since(git_dir: &Path, base: &str, paths: &[&str]) -> bool {
    let mut git = Command::new("git");
    git.current_dir(git_dir);

    git.args(["diff-index", "--quiet", base, "--"]).args(paths);

    // Exit code 0 => no changes
    // Exit code 1 => some changes were detected
    !git.status().expect("cannot run git diff-index").success()
}

/// Returns the latest upstream commit that modified `target_paths`, or `None` if no such commit
/// was found.
fn get_latest_upstream_commit_that_modified_files(
    git_dir: &Path,
    git_config: &GitConfig<'_>,
    target_paths: &[&str],
) -> Result<Option<String>, String> {
    let mut git = Command::new("git");
    git.current_dir(git_dir);

    // In theory, we could just use
    // `git rev-list --first-parent HEAD --author=<merge-bot> -- <paths>`
    // to find the latest upstream commit that modified `<paths>`.
    // However, this does not work if you are in a subtree sync branch that contains merge commits
    // which have the subtree history as their first parent, and the rustc history as second parent:
    // `--first-parent` will just walk up the subtree history and never see a single rustc commit.
    // We thus have to take a two-pronged approach. First lookup the most recent upstream commit
    // by *date* (this should work even in a subtree sync branch), and then start the lookup for
    // modified paths starting from that commit.
    //
    // See https://github.com/rust-lang/rust/pull/138591#discussion_r2037081858 for more details.
    let upstream = get_closest_upstream_commit(Some(git_dir), git_config, CiEnv::None)?
        .unwrap_or_else(|| "HEAD".to_string());

    git.args([
        "rev-list",
        "--first-parent",
        "-n1",
        &upstream,
        "--author",
        git_config.git_merge_commit_email,
    ]);

    if !target_paths.is_empty() {
        git.arg("--").args(target_paths);
    }
    let output = output_result(&mut git)?.trim().to_owned();
    if output.is_empty() { Ok(None) } else { Ok(Some(output)) }
}

/// Returns the most recent (ordered chronologically) commit found in the local history that
/// should exist upstream. We identify upstream commits by the e-mail of the commit
/// author.
///
/// If we are in CI, we simply return our first parent.
pub fn get_closest_upstream_commit(
    git_dir: Option<&Path>,
    config: &GitConfig<'_>,
    env: CiEnv,
) -> Result<Option<String>, String> {
    let base = match env {
        CiEnv::None => "HEAD",
        CiEnv::GitHubActions => {
            // On CI, we should always have a non-upstream merge commit at the tip,
            // and our first parent should be the most recently merged upstream commit.
            // We thus simply return our first parent.
            return resolve_commit_sha(git_dir, "HEAD^1").map(Some);
        }
    };

    let mut git = Command::new("git");

    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }

    // We do not use `--first-parent`, because we can be in a situation (outside CI) where we have
    // a subtree merge that actually has the main rustc history as its second parent.
    // Using `--first-parent` would recurse into the history of the subtree, which could have some
    // old bors commits that are not relevant to us.
    // With `--author-date-order`, git recurses into all parent subtrees, and returns the most
    // chronologically recent bors commit.
    // Here we assume that none of our subtrees use bors anymore, and that all their old bors
    // commits are way older than recent rustc bors commits!
    git.args([
        "rev-list",
        "--author-date-order",
        &format!("--author={}", config.git_merge_commit_email),
        "-n1",
        base,
    ]);

    let output = output_result(&mut git)?.trim().to_owned();
    if output.is_empty() { Ok(None) } else { Ok(Some(output)) }
}

/// Resolve the commit SHA of `commit_ref`.
fn resolve_commit_sha(git_dir: Option<&Path>, commit_ref: &str) -> Result<String, String> {
    let mut git = Command::new("git");

    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }

    git.args(["rev-parse", commit_ref]);

    Ok(output_result(&mut git)?.trim().to_owned())
}

/// Returns the files that have been modified in the current branch compared to the master branch.
/// This includes committed changes, uncommitted changes, and changes that are not even staged.
///
/// The `extensions` parameter can be used to filter the files by their extension.
/// Does not include removed files.
/// If `extensions` is empty, all files will be returned.
pub fn get_git_modified_files(
    config: &GitConfig<'_>,
    git_dir: Option<&Path>,
    extensions: &[&str],
) -> Result<Vec<String>, String> {
    let Some(merge_base) = get_closest_upstream_commit(git_dir, config, CiEnv::None)? else {
        return Err("No upstream commit was found".to_string());
    };

    let mut git = Command::new("git");
    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }
    let files = output_result(git.args(["diff-index", "--name-status", merge_base.trim()]))?
        .lines()
        .filter_map(|f| {
            let (status, name) = f.trim().split_once(char::is_whitespace).unwrap();
            if status == "D" {
                None
            } else if Path::new(name).extension().map_or(extensions.is_empty(), |ext| {
                // If there is no extension, we allow the path if `extensions` is empty
                // If there is an extension, we allow it if `extension` is empty or it contains the
                // extension.
                extensions.is_empty() || extensions.contains(&ext.to_str().unwrap())
            }) {
                Some(name.to_owned())
            } else {
                None
            }
        })
        .collect();
    Ok(files)
}

/// Returns the files that haven't been added to git yet.
pub fn get_git_untracked_files(git_dir: Option<&Path>) -> Result<Option<Vec<String>>, String> {
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
