use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::process::Command;

use crate::ci::CiEnv;
use crate::git::{GitConfig, PathFreshness, check_path_modifications};

#[test]
fn test_pr_ci_unchanged_anywhere() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_nonupstream_merge(&["b"]);
        let src = ctx.check_modifications(&["c"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::LastModifiedUpstream { upstream: sha });
    });
}

#[test]
fn test_pr_ci_changed_in_pr() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_nonupstream_merge(&["b"]);
        let src = ctx.check_modifications(&["b"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::HasLocalModifications { upstream: sha });
    });
}

#[test]
fn test_auto_ci_unchanged_anywhere_select_parent() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_upstream_merge(&["b"]);
        let src = ctx.check_modifications(&["c"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::LastModifiedUpstream { upstream: sha });
    });
}

#[test]
fn test_auto_ci_changed_in_pr() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_upstream_merge(&["b", "c"]);
        let src = ctx.check_modifications(&["c", "d"], CiEnv::GitHubActions);
        assert_eq!(src, PathFreshness::HasLocalModifications { upstream: sha });
    });
}

#[test]
fn test_local_uncommitted_modifications() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_branch("feature");
        ctx.modify("a");

        assert_eq!(
            ctx.check_modifications(&["a", "d"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: sha }
        );
    });
}

#[test]
fn test_local_committed_modifications() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a"]);
        ctx.create_upstream_merge(&["b", "c"]);
        ctx.create_branch("feature");
        ctx.modify("x");
        ctx.commit();
        ctx.modify("a");
        ctx.commit();

        assert_eq!(
            ctx.check_modifications(&["a", "d"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: sha }
        );
    });
}

#[test]
fn test_local_committed_modifications_subdirectory() {
    git_test(|ctx| {
        let sha = ctx.create_upstream_merge(&["a/b/c"]);
        ctx.create_upstream_merge(&["b", "c"]);
        ctx.create_branch("feature");
        ctx.modify("a/b/d");
        ctx.commit();

        assert_eq!(
            ctx.check_modifications(&["a/b"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: sha }
        );
    });
}

#[test]
fn test_local_changes_in_head_upstream() {
    git_test(|ctx| {
        // We want to resolve to the upstream commit that made modifications to a,
        // even if it is currently HEAD
        let sha = ctx.create_upstream_merge(&["a"]);
        assert_eq!(
            ctx.check_modifications(&["a", "d"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: sha }
        );
    });
}

#[test]
fn test_local_changes_in_previous_upstream() {
    git_test(|ctx| {
        // We want to resolve to this commit, which modified a
        let sha = ctx.create_upstream_merge(&["a", "e"]);
        // Not to this commit, which is the latest upstream commit
        ctx.create_upstream_merge(&["b", "c"]);
        ctx.create_branch("feature");
        ctx.modify("d");
        ctx.commit();
        assert_eq!(
            ctx.check_modifications(&["a"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: sha }
        );
    });
}

#[test]
fn test_local_no_upstream_commit_with_changes() {
    git_test(|ctx| {
        ctx.create_upstream_merge(&["a", "e"]);
        ctx.create_upstream_merge(&["a", "e"]);
        // We want to fall back to this commit, because there are no commits
        // that modified `x`.
        let sha = ctx.create_upstream_merge(&["a", "e"]);
        ctx.create_branch("feature");
        ctx.modify("d");
        ctx.commit();
        assert_eq!(
            ctx.check_modifications(&["x"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: sha }
        );
    });
}

#[test]
fn test_local_no_upstream_commit() {
    git_test(|ctx| {
        let src = ctx.check_modifications(&["c", "d"], CiEnv::None);
        assert_eq!(src, PathFreshness::MissingUpstream);
    });
}

#[test]
fn test_local_changes_negative_path() {
    git_test(|ctx| {
        let upstream = ctx.create_upstream_merge(&["a"]);
        ctx.create_branch("feature");
        ctx.modify("b");
        ctx.modify("d");
        ctx.commit();

        assert_eq!(
            ctx.check_modifications(&[":!b", ":!d"], CiEnv::None),
            PathFreshness::LastModifiedUpstream { upstream: upstream.clone() }
        );
        assert_eq!(
            ctx.check_modifications(&[":!c"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream: upstream.clone() }
        );
        assert_eq!(
            ctx.check_modifications(&[":!d", ":!x"], CiEnv::None),
            PathFreshness::HasLocalModifications { upstream }
        );
    });
}

struct GitCtx {
    dir: tempfile::TempDir,
    git_repo: String,
    nightly_branch: String,
    merge_bot_email: String,
}

impl GitCtx {
    fn new() -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        let ctx = Self {
            dir,
            git_repo: "rust-lang/rust".to_string(),
            nightly_branch: "nightly".to_string(),
            merge_bot_email: "Merge bot <merge-bot@rust-lang.org>".to_string(),
        };
        ctx.run_git(&["init"]);
        ctx.run_git(&["config", "user.name", "Tester"]);
        ctx.run_git(&["config", "user.email", "tester@rust-lang.org"]);
        ctx.modify("README.md");
        ctx.commit();
        ctx.run_git(&["branch", "-m", "main"]);
        ctx
    }

    fn check_modifications(&self, target_paths: &[&str], ci_env: CiEnv) -> PathFreshness {
        check_path_modifications(Some(self.dir.path()), &self.git_config(), target_paths, ci_env)
            .unwrap()
    }

    fn create_upstream_merge(&self, modified_files: &[&str]) -> String {
        self.create_branch_and_merge("previous-pr", modified_files, &self.merge_bot_email)
    }

    fn create_nonupstream_merge(&self, modified_files: &[&str]) -> String {
        self.create_branch_and_merge("pr", modified_files, "Tester <tester@rust-lang.org>")
    }

    fn create_branch_and_merge(
        &self,
        branch: &str,
        modified_files: &[&str],
        author: &str,
    ) -> String {
        self.create_branch(branch);
        for file in modified_files {
            self.modify(file);
        }
        self.commit();
        self.switch_to_branch("main");
        self.merge(branch, author);
        self.run_git(&["branch", "-d", branch]);
        self.get_current_commit()
    }

    fn get_current_commit(&self) -> String {
        self.run_git(&["rev-parse", "HEAD"])
    }

    fn merge(&self, branch: &str, author: &str) {
        self.run_git(&["merge", "--no-commit", "--no-ff", branch]);
        self.run_git(&[
            "commit".to_string(),
            "-m".to_string(),
            "Merge of {branch}".to_string(),
            "--author".to_string(),
            author.to_string(),
        ]);
    }

    fn modify(&self, path: &str) {
        use std::io::Write;

        let path = self.dir.path().join(path);
        std::fs::create_dir_all(&path.parent().unwrap()).unwrap();

        let mut file = OpenOptions::new().create(true).append(true).open(path).unwrap();
        writeln!(file, "line").unwrap();
    }

    fn commit(&self) -> String {
        self.run_git(&["add", "."]);
        self.run_git(&["commit", "-m", "commit message"]);
        self.get_current_commit()
    }

    fn switch_to_branch(&self, name: &str) {
        self.run_git(&["switch", name]);
    }

    /// Creates a branch and switches to it.
    fn create_branch(&self, name: &str) {
        self.run_git(&["checkout", "-b", name]);
    }

    fn run_git<S: AsRef<OsStr>>(&self, args: &[S]) -> String {
        let mut cmd = self.git_cmd();
        cmd.args(args);
        eprintln!("Running {cmd:?}");
        let output = cmd.output().unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap().trim().to_string();
        let stderr = String::from_utf8(output.stderr).unwrap().trim().to_string();
        if !output.status.success() {
            panic!("Git command `{cmd:?}` failed\nStdout\n{stdout}\nStderr\n{stderr}");
        }
        stdout
    }

    fn git_cmd(&self) -> Command {
        let mut cmd = Command::new("git");
        cmd.current_dir(&self.dir);
        cmd
    }

    fn git_config(&self) -> GitConfig<'_> {
        GitConfig {
            git_repository: &self.git_repo,
            nightly_branch: &self.nightly_branch,
            git_merge_commit_email: &self.merge_bot_email,
        }
    }
}

fn git_test<F>(test_fn: F)
where
    F: FnOnce(&GitCtx),
{
    let ctx = GitCtx::new();
    test_fn(&ctx);
}
