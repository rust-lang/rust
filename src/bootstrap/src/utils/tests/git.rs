use std::ffi::OsStr;
use std::fs::OpenOptions;
use std::path::Path;
use std::process::Command;

use build_helper::ci::CiEnv;
use build_helper::git::{GitConfig, PathFreshness, check_path_modifications};

pub struct GitCtx {
    dir: tempfile::TempDir,
    pub git_repo: String,
    pub nightly_branch: String,
    pub merge_bot_email: String,
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

    pub fn get_path(&self) -> &Path {
        self.dir.path()
    }

    pub fn check_modifications(&self, target_paths: &[&str], ci_env: CiEnv) -> PathFreshness {
        check_path_modifications(self.dir.path(), &self.git_config(), target_paths, ci_env).unwrap()
    }

    pub fn create_upstream_merge(&self, modified_files: &[&str]) -> String {
        self.create_branch_and_merge("previous-pr", modified_files, &self.merge_bot_email)
    }

    pub fn create_nonupstream_merge(&self, modified_files: &[&str]) -> String {
        self.create_branch_and_merge("pr", modified_files, "Tester <tester@rust-lang.org>")
    }

    pub fn create_branch_and_merge(
        &self,
        branch: &str,
        modified_files: &[&str],
        author: &str,
    ) -> String {
        let current_branch = self.get_current_branch();

        self.create_branch(branch);
        for file in modified_files {
            self.modify(file);
        }
        self.commit();
        self.switch_to_branch(&current_branch);
        self.merge(branch, author);
        self.run_git(&["branch", "-d", branch]);
        self.get_current_commit()
    }

    pub fn get_current_commit(&self) -> String {
        self.run_git(&["rev-parse", "HEAD"])
    }

    pub fn get_current_branch(&self) -> String {
        self.run_git(&["rev-parse", "--abbrev-ref", "HEAD"])
    }

    pub fn merge(&self, branch: &str, author: &str) {
        self.run_git(&["merge", "--no-commit", "--no-ff", branch]);
        self.run_git(&[
            "commit".to_string(),
            "-m".to_string(),
            format!("Merge of {branch} into {}", self.get_current_branch()),
            "--author".to_string(),
            author.to_string(),
        ]);
    }

    pub fn modify(&self, path: &str) {
        self.write(path, "line");
    }

    pub fn write(&self, path: &str, data: &str) {
        use std::io::Write;

        let path = self.dir.path().join(path);
        std::fs::create_dir_all(&path.parent().unwrap()).unwrap();

        let mut file = OpenOptions::new().create(true).append(true).open(path).unwrap();
        writeln!(file, "{data}").unwrap();
    }

    pub fn commit(&self) -> String {
        self.run_git(&["add", "."]);
        self.run_git(&["commit", "-m", "commit message"]);
        self.get_current_commit()
    }

    pub fn switch_to_branch(&self, name: &str) {
        self.run_git(&["switch", name]);
    }

    /// Creates a branch and switches to it.
    pub fn create_branch(&self, name: &str) {
        self.run_git(&["checkout", "-b", name]);
    }

    pub fn run_git<S: AsRef<OsStr>>(&self, args: &[S]) -> String {
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
            nightly_branch: &self.nightly_branch,
            git_merge_commit_email: &self.merge_bot_email,
        }
    }
}

/// Run an end-to-end test that allows testing git logic.
pub fn git_test<F>(test_fn: F)
where
    F: FnOnce(&mut GitCtx),
{
    let mut ctx = GitCtx::new();
    test_fn(&mut ctx);
}
