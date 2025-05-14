use std::io::Write;
use std::ops::Not;
use std::path::PathBuf;
use std::time::Duration;
use std::{env, net, process};

use anyhow::{Context, anyhow, bail};
use xshell::{Shell, cmd};

/// Used for rustc syncs.
const JOSH_FILTER: &str = ":/src/doc/rustc-dev-guide";
const JOSH_PORT: u16 = 42042;
const UPSTREAM_REPO: &str = "rust-lang/rust";

pub enum RustcPullError {
    /// No changes are available to be pulled.
    NothingToPull,
    /// A rustc-pull has failed, probably a git operation error has occurred.
    PullFailed(anyhow::Error),
}

impl<E> From<E> for RustcPullError
where
    E: Into<anyhow::Error>,
{
    fn from(error: E) -> Self {
        Self::PullFailed(error.into())
    }
}

pub struct GitSync {
    dir: PathBuf,
}

/// This code was adapted from the miri repository
/// (https://github.com/rust-lang/miri/blob/6a68a79f38064c3bc30617cca4bdbfb2c336b140/miri-script/src/commands.rs#L236).
impl GitSync {
    pub fn from_current_dir() -> anyhow::Result<Self> {
        Ok(Self { dir: std::env::current_dir()? })
    }

    pub fn rustc_pull(&self, commit: Option<String>) -> Result<(), RustcPullError> {
        let sh = Shell::new()?;
        sh.change_dir(&self.dir);
        let commit = commit.map(Ok).unwrap_or_else(|| {
            let rust_repo_head =
                cmd!(sh, "git ls-remote https://github.com/{UPSTREAM_REPO}/ HEAD").read()?;
            rust_repo_head
                .split_whitespace()
                .next()
                .map(|front| front.trim().to_owned())
                .ok_or_else(|| anyhow!("Could not obtain Rust repo HEAD from remote."))
        })?;
        // Make sure the repo is clean.
        if cmd!(sh, "git status --untracked-files=no --porcelain").read()?.is_empty().not() {
            return Err(anyhow::anyhow!(
                "working directory must be clean before performing rustc pull"
            )
            .into());
        }
        // Make sure josh is running.
        let josh = Self::start_josh()?;
        let josh_url =
            format!("http://localhost:{JOSH_PORT}/{UPSTREAM_REPO}.git@{commit}{JOSH_FILTER}.git");

        let previous_base_commit = sh.read_file("rust-version")?.trim().to_string();
        if previous_base_commit == commit {
            return Err(RustcPullError::NothingToPull);
        }

        // Update rust-version file. As a separate commit, since making it part of
        // the merge has confused the heck out of josh in the past.
        // We pass `--no-verify` to avoid running git hooks.
        // We do this before the merge so that if there are merge conflicts, we have
        // the right rust-version file while resolving them.
        sh.write_file("rust-version", format!("{commit}\n"))?;
        const PREPARING_COMMIT_MESSAGE: &str = "Preparing for merge from rustc";
        cmd!(sh, "git commit rust-version --no-verify -m {PREPARING_COMMIT_MESSAGE}")
            .run()
            .context("FAILED to commit rust-version file, something went wrong")?;

        // Fetch given rustc commit.
        cmd!(sh, "git fetch {josh_url}")
            .run()
            .inspect_err(|_| {
                // Try to un-do the previous `git commit`, to leave the repo in the state we found it.
                cmd!(sh, "git reset --hard HEAD^")
                    .run()
                    .expect("FAILED to clean up again after failed `git fetch`, sorry for that");
            })
            .context("FAILED to fetch new commits, something went wrong (committing the rust-version file has been undone)")?;

        // This should not add any new root commits. So count those before and after merging.
        let num_roots = || -> anyhow::Result<u32> {
            Ok(cmd!(sh, "git rev-list HEAD --max-parents=0 --count")
                .read()
                .context("failed to determine the number of root commits")?
                .parse::<u32>()?)
        };
        let num_roots_before = num_roots()?;

        let sha =
            cmd!(sh, "git rev-parse HEAD").output().context("FAILED to get current commit")?.stdout;

        // Merge the fetched commit.
        const MERGE_COMMIT_MESSAGE: &str = "Merge from rustc";
        cmd!(sh, "git merge FETCH_HEAD --no-verify --no-ff -m {MERGE_COMMIT_MESSAGE}")
            .run()
            .context("FAILED to merge new commits, something went wrong")?;

        let current_sha =
            cmd!(sh, "git rev-parse HEAD").output().context("FAILED to get current commit")?.stdout;
        if current_sha == sha {
            cmd!(sh, "git reset --hard HEAD^")
                .run()
                .expect("FAILED to clean up after creating the preparation commit");
            eprintln!(
                "No merge was performed, no changes to pull were found. Rolled back the preparation commit."
            );
            return Err(RustcPullError::NothingToPull);
        }

        // Check that the number of roots did not increase.
        if num_roots()? != num_roots_before {
            return Err(anyhow::anyhow!(
                "Josh created a new root commit. This is probably not the history you want."
            )
            .into());
        }

        drop(josh);
        Ok(())
    }

    pub fn rustc_push(&self, github_user: String, branch: String) -> anyhow::Result<()> {
        let sh = Shell::new()?;
        sh.change_dir(&self.dir);
        let base = sh.read_file("rust-version")?.trim().to_owned();
        // Make sure the repo is clean.
        if cmd!(sh, "git status --untracked-files=no --porcelain").read()?.is_empty().not() {
            bail!("working directory must be clean before running `rustc-push`");
        }
        // Make sure josh is running.
        let josh = Self::start_josh()?;
        let josh_url =
            format!("http://localhost:{JOSH_PORT}/{github_user}/rust.git{JOSH_FILTER}.git");

        // Find a repo we can do our preparation in.
        if let Ok(rustc_git) = env::var("RUSTC_GIT") {
            // If rustc_git is `Some`, we'll use an existing fork for the branch updates.
            sh.change_dir(rustc_git);
        } else {
            // Otherwise, do this in the local repo.
            println!(
                "This will pull a copy of the rust-lang/rust history into this checkout, growing it by about 1GB."
            );
            print!(
                "To avoid that, abort now and set the `RUSTC_GIT` environment variable to an existing rustc checkout. Proceed? [y/N] "
            );
            std::io::stdout().flush()?;
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer)?;
            if answer.trim().to_lowercase() != "y" {
                std::process::exit(1);
            }
        };
        // Prepare the branch. Pushing works much better if we use as base exactly
        // the commit that we pulled from last time, so we use the `rust-version`
        // file to find out which commit that would be.
        println!("Preparing {github_user}/rust (base: {base})...");
        if cmd!(sh, "git fetch https://github.com/{github_user}/rust {branch}")
            .ignore_stderr()
            .read()
            .is_ok()
        {
            println!(
                "The branch '{branch}' seems to already exist in 'https://github.com/{github_user}/rust'. Please delete it and try again."
            );
            std::process::exit(1);
        }
        cmd!(sh, "git fetch https://github.com/{UPSTREAM_REPO} {base}").run()?;
        cmd!(sh, "git push https://github.com/{github_user}/rust {base}:refs/heads/{branch}")
            .ignore_stdout()
            .ignore_stderr() // silence the "create GitHub PR" message
            .run()?;
        println!();

        // Do the actual push.
        sh.change_dir(&self.dir);
        println!("Pushing changes...");
        cmd!(sh, "git push {josh_url} HEAD:{branch}").run()?;
        println!();

        // Do a round-trip check to make sure the push worked as expected.
        cmd!(sh, "git fetch {josh_url} {branch}").ignore_stderr().read()?;
        let head = cmd!(sh, "git rev-parse HEAD").read()?;
        let fetch_head = cmd!(sh, "git rev-parse FETCH_HEAD").read()?;
        if head != fetch_head {
            bail!(
                "Josh created a non-roundtrip push! Do NOT merge this into rustc!\n\
                Expected {head}, got {fetch_head}."
            );
        }
        println!(
            "Confirmed that the push round-trips back to rustc-dev-guide properly. Please create a rustc PR:"
        );
        println!(
            // Open PR with `subtree update` title to silence the `no-merges` triagebot check
            "    https://github.com/{UPSTREAM_REPO}/compare/{github_user}:{branch}?quick_pull=1&title=rustc-dev-guide+subtree+update&body=r?+@ghost"
        );

        drop(josh);
        Ok(())
    }

    fn start_josh() -> anyhow::Result<impl Drop> {
        // Determine cache directory.
        let local_dir = {
            let user_dirs =
                directories::ProjectDirs::from("org", "rust-lang", "rustc-dev-guide-josh").unwrap();
            user_dirs.cache_dir().to_owned()
        };

        // Start josh, silencing its output.
        let mut cmd = process::Command::new("josh-proxy");
        cmd.arg("--local").arg(local_dir);
        cmd.arg("--remote").arg("https://github.com");
        cmd.arg("--port").arg(JOSH_PORT.to_string());
        cmd.arg("--no-background");
        cmd.stdout(process::Stdio::null());
        cmd.stderr(process::Stdio::null());
        let josh = cmd.spawn().context("failed to start josh-proxy, make sure it is installed")?;

        // Create a wrapper that stops it on drop.
        struct Josh(process::Child);
        impl Drop for Josh {
            fn drop(&mut self) {
                #[cfg(unix)]
                {
                    // Try to gracefully shut it down.
                    process::Command::new("kill")
                        .args(["-s", "INT", &self.0.id().to_string()])
                        .output()
                        .expect("failed to SIGINT josh-proxy");
                    // Sadly there is no "wait with timeout"... so we just give it some time to finish.
                    std::thread::sleep(Duration::from_millis(100));
                    // Now hopefully it is gone.
                    if self.0.try_wait().expect("failed to wait for josh-proxy").is_some() {
                        return;
                    }
                }
                // If that didn't work (or we're not on Unix), kill it hard.
                eprintln!(
                    "I have to kill josh-proxy the hard way, let's hope this does not break anything."
                );
                self.0.kill().expect("failed to SIGKILL josh-proxy");
            }
        }

        // Wait until the port is open. We try every 10ms until 1s passed.
        for _ in 0..100 {
            // This will generally fail immediately when the port is still closed.
            let josh_ready = net::TcpStream::connect_timeout(
                &net::SocketAddr::from(([127, 0, 0, 1], JOSH_PORT)),
                Duration::from_millis(1),
            );
            if josh_ready.is_ok() {
                return Ok(Josh(josh));
            }
            // Not ready yet.
            std::thread::sleep(Duration::from_millis(10));
        }
        bail!("Even after waiting for 1s, josh-proxy is still not available.")
    }
}
