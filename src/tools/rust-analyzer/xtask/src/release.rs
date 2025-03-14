mod changelog;

use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{bail, Context as _};
use directories::ProjectDirs;
use stdx::JodChild;
use xshell::{cmd, Shell};

use crate::{date_iso, flags, is_release_tag, project_root};

impl flags::Release {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        if !self.dry_run {
            cmd!(sh, "git switch release").run()?;
            cmd!(sh, "git fetch upstream --tags --force").run()?;
            cmd!(sh, "git reset --hard tags/nightly").run()?;
            // The `release` branch sometimes has a couple of cherry-picked
            // commits for patch releases. If that's the case, just overwrite
            // it. As we are setting `release` branch to an up-to-date `nightly`
            // tag, this shouldn't be problematic in general.
            //
            // Note that, as we tag releases, we don't worry about "losing"
            // commits -- they'll be kept alive by the tag. More generally, we
            // don't care about historic releases all that much, it's fine even
            // to delete old tags.
            cmd!(sh, "git push --force").run()?;
        }

        let website_root = project_root().join("../rust-analyzer.github.io");
        {
            let _dir = sh.push_dir(&website_root);
            cmd!(sh, "git switch src").run()?;
            cmd!(sh, "git pull").run()?;
        }
        let changelog_dir = website_root.join("./thisweek/_posts");

        let today = date_iso(sh)?;
        let commit = cmd!(sh, "git rev-parse HEAD").read()?;
        let changelog_n = sh
            .read_dir(changelog_dir.as_path())?
            .into_iter()
            .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().to_string()))
            .filter_map(|s| s.splitn(5, '-').last().map(|n| n.replace('-', ".")))
            .filter_map(|s| s.parse::<f32>().ok())
            .map(|n| 1 + n.floor() as usize)
            .max()
            .unwrap_or_default();

        let tags = cmd!(sh, "git tag --list").read()?;
        let prev_tag = tags.lines().filter(|line| is_release_tag(line)).next_back().unwrap();

        let contents = changelog::get_changelog(sh, changelog_n, &commit, prev_tag, &today)?;
        let path = changelog_dir.join(format!("{today}-changelog-{changelog_n}.adoc"));
        sh.write_file(path, contents)?;

        Ok(())
    }
}

// git sync implementation adapted from https://github.com/rust-lang/miri/blob/62039ac/miri-script/src/commands.rs
impl flags::RustcPull {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        sh.change_dir(project_root());
        let commit = self.commit.map(Result::Ok).unwrap_or_else(|| {
            let rust_repo_head =
                cmd!(sh, "git ls-remote https://github.com/rust-lang/rust/ HEAD").read()?;
            rust_repo_head
                .split_whitespace()
                .next()
                .map(|front| front.trim().to_owned())
                .ok_or_else(|| anyhow::format_err!("Could not obtain Rust repo HEAD from remote."))
        })?;
        // Make sure the repo is clean.
        if !cmd!(sh, "git status --untracked-files=no --porcelain").read()?.is_empty() {
            bail!("working directory must be clean before running `cargo xtask pull`");
        }
        // This should not add any new root commits. So count those before and after merging.
        let num_roots = || -> anyhow::Result<u32> {
            Ok(cmd!(sh, "git rev-list HEAD --max-parents=0 --count")
                .read()
                .context("failed to determine the number of root commits")?
                .parse::<u32>()?)
        };
        let num_roots_before = num_roots()?;
        // Make sure josh is running.
        let josh = start_josh()?;

        // Update rust-version file. As a separate commit, since making it part of
        // the merge has confused the heck out of josh in the past.
        // We pass `--no-verify` to avoid running any git hooks that might exist,
        // in case they dirty the repository.
        sh.write_file("rust-version", format!("{commit}\n"))?;
        const PREPARING_COMMIT_MESSAGE: &str = "Preparing for merge from rust-lang/rust";
        cmd!(sh, "git commit rust-version --no-verify -m {PREPARING_COMMIT_MESSAGE}")
            .run()
            .context("FAILED to commit rust-version file, something went wrong")?;

        // Fetch given rustc commit.
        cmd!(sh, "git fetch http://localhost:{JOSH_PORT}/rust-lang/rust.git@{commit}{JOSH_FILTER}.git")
            .run()
            .inspect_err(|_| {
                // Try to un-do the previous `git commit`, to leave the repo in the state we found it it.
                cmd!(sh, "git reset --hard HEAD^")
                    .run()
                    .expect("FAILED to clean up again after failed `git fetch`, sorry for that");
            })
            .context("FAILED to fetch new commits, something went wrong (committing the rust-version file has been undone)")?;

        // Merge the fetched commit.
        const MERGE_COMMIT_MESSAGE: &str = "Merge from rust-lang/rust";
        cmd!(sh, "git merge FETCH_HEAD --no-verify --no-ff -m {MERGE_COMMIT_MESSAGE}")
            .run()
            .context("FAILED to merge new commits, something went wrong")?;

        // Check that the number of roots did not increase.
        if num_roots()? != num_roots_before {
            bail!("Josh created a new root commit. This is probably not the history you want.");
        }

        drop(josh);
        Ok(())
    }
}

impl flags::RustcPush {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        let branch = self.branch.as_deref().unwrap_or("sync-from-ra");
        let rust_path = self.rust_path;
        let rust_fork = self.rust_fork;

        sh.change_dir(project_root());
        let base = sh.read_file("rust-version")?.trim().to_owned();
        // Make sure the repo is clean.
        if !cmd!(sh, "git status --untracked-files=no --porcelain").read()?.is_empty() {
            bail!("working directory must be clean before running `cargo xtask push`");
        }
        // Make sure josh is running.
        let josh = start_josh()?;

        // Find a repo we can do our preparation in.
        sh.change_dir(rust_path);

        // Prepare the branch. Pushing works much better if we use as base exactly
        // the commit that we pulled from last time, so we use the `rust-version`
        // file to find out which commit that would be.
        println!("Preparing {rust_fork} (base: {base})...");
        if cmd!(sh, "git fetch https://github.com/{rust_fork} {branch}")
            .ignore_stderr()
            .read()
            .is_ok()
        {
            bail!(
                "The branch `{branch}` seems to already exist in `https://github.com/{rust_fork}`. Please delete it and try again."
            );
        }
        cmd!(sh, "git fetch https://github.com/rust-lang/rust {base}").run()?;
        cmd!(sh, "git push https://github.com/{rust_fork} {base}:refs/heads/{branch}")
            .ignore_stdout()
            .ignore_stderr() // silence the "create GitHub PR" message
            .run()?;
        println!();

        // Do the actual push.
        sh.change_dir(project_root());
        println!("Pushing rust-analyzer changes...");
        cmd!(
            sh,
            "git push http://localhost:{JOSH_PORT}/{rust_fork}.git{JOSH_FILTER}.git HEAD:{branch}"
        )
        .run()?;
        println!();

        // Do a round-trip check to make sure the push worked as expected.
        cmd!(
            sh,
            "git fetch http://localhost:{JOSH_PORT}/{rust_fork}.git{JOSH_FILTER}.git {branch}"
        )
        .ignore_stderr()
        .read()?;
        let head = cmd!(sh, "git rev-parse HEAD").read()?;
        let fetch_head = cmd!(sh, "git rev-parse FETCH_HEAD").read()?;
        if head != fetch_head {
            bail!(
                "Josh created a non-roundtrip push! Do NOT merge this into rustc!\n\
                Expected {head}, got {fetch_head}."
            );
        }
        println!("Confirmed that the push round-trips back to rust-analyzer properly. Please create a rustc PR:");
        // https://github.com/github-linguist/linguist/compare/master...octocat:linguist:master
        let fork_path = rust_fork.replace('/', ":");
        println!(
            "    https://github.com/rust-lang/rust/compare/{fork_path}:{branch}?quick_pull=1&title=Subtree+update+of+rust-analyzer&body=r?+@ghost"
        );

        drop(josh);
        Ok(())
    }
}

/// Used for rustc syncs.
const JOSH_FILTER: &str =
    ":rev(55d9a533b309119c8acd13061581b43ae8840823:prefix=src/tools/rust-analyzer):/src/tools/rust-analyzer";
const JOSH_PORT: &str = "42042";

fn start_josh() -> anyhow::Result<impl Drop> {
    // Determine cache directory.
    let local_dir = {
        let user_dirs = ProjectDirs::from("org", "rust-lang", "rust-analyzer-josh").unwrap();
        user_dirs.cache_dir().to_owned()
    };

    // Start josh, silencing its output.
    let mut cmd = Command::new("josh-proxy");
    cmd.arg("--local").arg(local_dir);
    cmd.arg("--remote").arg("https://github.com");
    cmd.arg("--port").arg(JOSH_PORT);
    cmd.arg("--no-background");
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    let josh = cmd.spawn().context("failed to start josh-proxy, make sure it is installed")?;
    // Give it some time so hopefully the port is open. (100ms was not enough.)
    thread::sleep(Duration::from_millis(200));

    Ok(JodChild(josh))
}
