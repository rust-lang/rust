use std::path::{Path, PathBuf};
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
    HasLocalModifications { upstream: String, modifications: Vec<PathBuf> },
    /// No upstream commit was found.
    /// This should not happen in most reasonable circumstances, but one never knows.
    MissingUpstream,
}

/// This function figures out if a set of paths was last modified upstream or
/// if there are some local modifications made to them.
/// It can be used to figure out if we should download artifacts from CI or rather
/// build them locally.
///
/// If no upstream bors merge commit can be found in the available local git history,
/// the function returns `MissingUpstream` so callers can conservatively avoid using
/// CI artifacts.
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
/// - In CI, we prefer a shallow merge-parent fast path when `HEAD` is a CI-generated merge
///   commit. However, fork push workflows can also run in shallow clones where `HEAD` is just the
///   branch tip, so blindly using `HEAD^1` there would pick a fork commit instead of the upstream
///   base. In those cases we fall back to the fetched nightly branch ref, and only then to the
///   normal upstream search logic.
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
        // CI may be running on a synthetic merge ref or a shallow fork push ref.
        // `get_closest_upstream_commit` handles the trusted merge-parent fast path and falls back
        // to the fetched nightly branch ref when the merge-parent assumption is not valid. If a
        // fork push clone does not have that ref either, conservatively report `MissingUpstream`
        // so callers disable CI downloads and build locally instead of panicking.
        get_closest_upstream_commit(Some(git_dir), config, ci_env)?
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
    let modifications = changes_since(git_dir, &upstream_sha, target_paths)?;
    if !modifications.is_empty() {
        Ok(PathFreshness::HasLocalModifications { upstream: upstream_sha, modifications })
    } else {
        Ok(PathFreshness::LastModifiedUpstream { upstream: upstream_sha })
    }
}

/// Returns true if any of the passed `paths` have changed since the `base` commit.
pub fn changes_since(git_dir: &Path, base: &str, paths: &[&str]) -> Result<Vec<PathBuf>, String> {
    use std::io::BufRead;

    run_git_diff_index(Some(git_dir), |cmd| {
        cmd.args([base, "--name-only", "--"]).args(paths);

        let output = cmd.stderr(Stdio::inherit()).output().expect("cannot run git diff-index");
        if !output.status.success() {
            return Err(format!("failed to run: {cmd:?}: {:?}", output.status));
        }

        output
            .stdout
            .lines()
            .map(|res| match res {
                Ok(line) => Ok(PathBuf::from(line)),
                Err(e) => Err(format!("invalid UTF-8 in diff-index: {e:?}")),
            })
            .collect()
    })
}

// Temporary e-mail used by new bors for merge commits for a few days, until it learned how to reuse
// the original homu e-mail
// FIXME: remove in Q2 2026
const TEMPORARY_BORS_EMAIL: &str = "122020455+rust-bors[bot]@users.noreply.github.com";

/// Escape characters from the git user e-mail, so that git commands do not interpret it as regex
/// special characters.
fn escape_email_git_regex(text: &str) -> String {
    text.replace("[", "\\[").replace("]", "\\]").replace(".", "\\.")
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
        &escape_email_git_regex(git_config.git_merge_commit_email),
    ]);

    // Also search for temporary bors account
    if git_config.git_merge_commit_email != TEMPORARY_BORS_EMAIL {
        git.args(["--author", &escape_email_git_regex(TEMPORARY_BORS_EMAIL)]);
    }

    if !target_paths.is_empty() {
        git.arg("--").args(target_paths);
    }
    let output = output_result(&mut git)?.trim().to_owned();
    if output.is_empty() { Ok(None) } else { Ok(Some(output)) }
}

/// Returns the most recent (ordered chronologically) commit found in the local history that
/// should exist upstream. We identify upstream commits by the e-mail of the commit
/// author.
pub fn get_closest_upstream_commit(
    git_dir: Option<&Path>,
    config: &GitConfig<'_>,
    env: CiEnv,
) -> Result<Option<String>, String> {
    match env {
        CiEnv::None => get_closest_upstream_commit_from_ref(git_dir, config, "HEAD"),
        CiEnv::GitHubActions => {
            // CI-generated PR and auto-merge refs put a synthetic merge commit at HEAD, so the
            // first parent is usually the most recent upstream merge commit. Fork push workflows
            // do not have that shape, though, and in shallow clones `HEAD^1` can just be the
            // previous fork commit. Only trust the fast path when it points at an actual upstream
            // merge-bot commit, otherwise fall back to the fetched nightly branch.
            if is_merge_commit(git_dir, "HEAD")? {
                let parent = resolve_commit_sha(git_dir, "HEAD^1")?;
                if is_upstream_merge_commit(git_dir, &parent, config)? {
                    return Ok(Some(parent));
                }
            }

            let nightly_ref = format!("refs/remotes/origin/{}", config.nightly_branch);
            if git_ref_exists(git_dir, &nightly_ref)? {
                if let Some(upstream) =
                    get_closest_upstream_commit_from_ref(git_dir, config, &nightly_ref)?
                {
                    return Ok(Some(upstream));
                }
            }

            get_closest_upstream_commit_from_ref(git_dir, config, "HEAD")
        }
    }
}

fn get_closest_upstream_commit_from_ref(
    git_dir: Option<&Path>,
    config: &GitConfig<'_>,
    base: &str,
) -> Result<Option<String>, String> {
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
        &format!("--author={}", &escape_email_git_regex(config.git_merge_commit_email),),
        "-n1",
        base,
    ]);

    // Also search for temporary bors account
    if config.git_merge_commit_email != TEMPORARY_BORS_EMAIL {
        git.args(["--author", &escape_email_git_regex(TEMPORARY_BORS_EMAIL)]);
    }

    let output = output_result(&mut git)?.trim().to_owned();
    if output.is_empty() { Ok(None) } else { Ok(Some(output)) }
}

fn is_merge_commit(git_dir: Option<&Path>, commit_ref: &str) -> Result<bool, String> {
    let mut git = Command::new("git");
    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }

    let output = git
        .args(["rev-parse", "--verify", "--quiet", &format!("{commit_ref}^2")])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .map_err(|e| format!("failed to run command: {git:?}: {e}"))?;
    Ok(output.success())
}

fn git_ref_exists(git_dir: Option<&Path>, refname: &str) -> Result<bool, String> {
    let mut git = Command::new("git");
    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }

    let output = git
        .args(["rev-parse", "--verify", "--quiet", refname])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .map_err(|e| format!("failed to run command: {git:?}: {e}"))?;
    Ok(output.success())
}

fn is_upstream_merge_commit(
    git_dir: Option<&Path>,
    commit_ref: &str,
    config: &GitConfig<'_>,
) -> Result<bool, String> {
    let mut git = Command::new("git");
    if let Some(git_dir) = git_dir {
        git.current_dir(git_dir);
    }

    git.args(["show", "-s", "--format=%ae", commit_ref]);
    let author_email = output_result(&mut git)?.trim().to_owned();
    let merge_bot_email = extract_author_email(config.git_merge_commit_email);
    Ok(author_email == merge_bot_email || author_email == TEMPORARY_BORS_EMAIL)
}

fn extract_author_email(author: &str) -> &str {
    author
        .split_once('<')
        .and_then(|(_, email)| email.trim().strip_suffix('>'))
        .map(str::trim)
        .unwrap_or_else(|| author.trim())
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

    let files = run_git_diff_index(git_dir, |cmd| {
        output_result(cmd.args(["--name-status", merge_base.trim()]))
    })?
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

/// diff-index can return outdated information, because it does not update the git index.
/// This function uses `update-index` to update the index first, and then provides `func` with a
/// command prepared to run `git diff-index`.
fn run_git_diff_index<F, T>(git_dir: Option<&Path>, func: F) -> T
where
    F: FnOnce(&mut Command) -> T,
{
    let git = || {
        let mut git = Command::new("git");
        if let Some(git_dir) = git_dir {
            git.current_dir(git_dir);
        }
        git
    };

    // We ignore the exit code, as it errors out when some files are modified.
    let _ = output_result(git().args(["update-index", "--refresh", "-q"]));
    func(git().arg("diff-index"))
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
