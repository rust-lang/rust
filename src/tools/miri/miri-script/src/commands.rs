use std::env;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::ops::Not;
use std::ops::Range;
use std::path::PathBuf;
use std::process;
use std::thread;
use std::time;

use anyhow::{anyhow, bail, Context, Result};
use path_macro::path;
use walkdir::WalkDir;
use xshell::{cmd, Shell};

use crate::util::*;
use crate::Command;

/// Used for rustc syncs.
const JOSH_FILTER: &str =
    ":rev(75dd959a3a40eb5b4574f8d2e23aa6efbeb33573:prefix=src/tools/miri):/src/tools/miri";
const JOSH_PORT: &str = "42042";

impl MiriEnv {
    /// Returns the location of the sysroot.
    ///
    /// If the target is None the sysroot will be built for the host machine.
    fn build_miri_sysroot(
        &mut self,
        quiet: bool,
        target: Option<impl AsRef<OsStr>>,
    ) -> Result<PathBuf> {
        if let Some(miri_sysroot) = self.sh.var_os("MIRI_SYSROOT") {
            // Sysroot already set, use that.
            return Ok(miri_sysroot.into());
        }
        let manifest_path = path!(self.miri_dir / "cargo-miri" / "Cargo.toml");
        let Self { toolchain, cargo_extra_flags, .. } = &self;

        // Make sure everything is built. Also Miri itself.
        self.build(path!(self.miri_dir / "Cargo.toml"), &[], quiet)?;
        self.build(&manifest_path, &[], quiet)?;

        let target_flag = if let Some(target) = &target {
            vec![OsStr::new("--target"), target.as_ref()]
        } else {
            vec![]
        };
        let target_flag = &target_flag;

        if !quiet {
            eprint!("$ cargo miri setup");
            if let Some(target) = &target {
                eprint!(" --target {target}", target = target.as_ref().to_string_lossy());
            }
            eprintln!();
        }

        let mut cmd = cmd!(self.sh,
            "cargo +{toolchain} --quiet run {cargo_extra_flags...} --manifest-path {manifest_path} --
             miri setup --print-sysroot {target_flag...}"
        );
        cmd.set_quiet(quiet);
        let output = cmd.read()?;
        self.sh.set_var("MIRI_SYSROOT", &output);
        Ok(output.into())
    }
}

impl Command {
    fn auto_actions() -> Result<()> {
        if env::var_os("MIRI_AUTO_OPS").is_some_and(|x| x == "no") {
            return Ok(());
        }

        let miri_dir = miri_dir()?;
        let auto_everything = path!(miri_dir / ".auto-everything").exists();
        let auto_toolchain = auto_everything || path!(miri_dir / ".auto-toolchain").exists();
        let auto_fmt = auto_everything || path!(miri_dir / ".auto-fmt").exists();
        let auto_clippy = auto_everything || path!(miri_dir / ".auto-clippy").exists();

        // `toolchain` goes first as it could affect the others
        if auto_toolchain {
            Self::toolchain(vec![])?;
        }
        if auto_fmt {
            Self::fmt(vec![])?;
        }
        if auto_clippy {
            Self::clippy(vec![])?;
        }

        Ok(())
    }

    fn start_josh() -> Result<impl Drop> {
        // Determine cache directory.
        let local_dir = {
            let user_dirs =
                directories::ProjectDirs::from("org", "rust-lang", "miri-josh").unwrap();
            user_dirs.cache_dir().to_owned()
        };

        // Start josh, silencing its output.
        let mut cmd = process::Command::new("josh-proxy");
        cmd.arg("--local").arg(local_dir);
        cmd.arg("--remote").arg("https://github.com");
        cmd.arg("--port").arg(JOSH_PORT);
        cmd.arg("--no-background");
        cmd.stdout(process::Stdio::null());
        cmd.stderr(process::Stdio::null());
        let josh = cmd.spawn().context("failed to start josh-proxy, make sure it is installed")?;
        // Give it some time so hopefully the port is open. (100ms was not enough.)
        thread::sleep(time::Duration::from_millis(200));

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
                    thread::sleep(time::Duration::from_millis(100));
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

        Ok(Josh(josh))
    }

    pub fn exec(self) -> Result<()> {
        // First, and crucially only once, run the auto-actions -- but not for all commands.
        match &self {
            Command::Install { .. }
            | Command::Build { .. }
            | Command::Check { .. }
            | Command::Test { .. }
            | Command::Run { .. }
            | Command::Fmt { .. }
            | Command::Clippy { .. }
            | Command::Cargo { .. } => Self::auto_actions()?,
            | Command::Toolchain { .. }
            | Command::Bench { .. }
            | Command::RustcPull { .. }
            | Command::RustcPush { .. } => {}
        }
        // Then run the actual command.
        match self {
            Command::Install { flags } => Self::install(flags),
            Command::Build { flags } => Self::build(flags),
            Command::Check { flags } => Self::check(flags),
            Command::Test { bless, flags, target } => Self::test(bless, flags, target),
            Command::Run { dep, verbose, many_seeds, target, edition, flags } =>
                Self::run(dep, verbose, many_seeds, target, edition, flags),
            Command::Fmt { flags } => Self::fmt(flags),
            Command::Clippy { flags } => Self::clippy(flags),
            Command::Cargo { flags } => Self::cargo(flags),
            Command::Bench { target, benches } => Self::bench(target, benches),
            Command::Toolchain { flags } => Self::toolchain(flags),
            Command::RustcPull { commit } => Self::rustc_pull(commit.clone()),
            Command::RustcPush { github_user, branch } => Self::rustc_push(github_user, branch),
        }
    }

    fn toolchain(flags: Vec<String>) -> Result<()> {
        // Make sure rustup-toolchain-install-master is installed.
        which::which("rustup-toolchain-install-master")
            .context("Please install rustup-toolchain-install-master by running 'cargo install rustup-toolchain-install-master'")?;
        let sh = Shell::new()?;
        sh.change_dir(miri_dir()?);
        let new_commit = sh.read_file("rust-version")?.trim().to_owned();
        let current_commit = {
            let rustc_info = cmd!(sh, "rustc +miri --version -v").read();
            if rustc_info.is_err() {
                None
            } else {
                let metadata = rustc_version::version_meta_for(&rustc_info.unwrap())?;
                Some(
                    metadata
                        .commit_hash
                        .ok_or_else(|| anyhow!("rustc metadata did not contain commit hash"))?,
                )
            }
        };
        // Check if we already are at that commit.
        if current_commit.as_ref() == Some(&new_commit) {
            if active_toolchain()? != "miri" {
                cmd!(sh, "rustup override set miri").run()?;
            }
            return Ok(());
        }
        // Install and setup new toolchain.
        cmd!(sh, "rustup toolchain uninstall miri").run()?;

        cmd!(sh, "rustup-toolchain-install-master -n miri -c cargo -c rust-src -c rustc-dev -c llvm-tools -c rustfmt -c clippy {flags...} -- {new_commit}").run()?;
        cmd!(sh, "rustup override set miri").run()?;
        // Cleanup.
        cmd!(sh, "cargo clean").run()?;
        // Call `cargo metadata` on the sources in case that changes the lockfile
        // (which fails under some setups when it is done from inside vscode).
        let sysroot = cmd!(sh, "rustc --print sysroot").read()?;
        let sysroot = sysroot.trim();
        cmd!(sh, "cargo metadata --format-version 1 --manifest-path {sysroot}/lib/rustlib/rustc-src/rust/compiler/rustc/Cargo.toml").ignore_stdout().run()?;
        Ok(())
    }

    fn rustc_pull(commit: Option<String>) -> Result<()> {
        let sh = Shell::new()?;
        sh.change_dir(miri_dir()?);
        let commit = commit.map(Result::Ok).unwrap_or_else(|| {
            let rust_repo_head =
                cmd!(sh, "git ls-remote https://github.com/rust-lang/rust/ HEAD").read()?;
            rust_repo_head
                .split_whitespace()
                .next()
                .map(|front| front.trim().to_owned())
                .ok_or_else(|| anyhow!("Could not obtain Rust repo HEAD from remote."))
        })?;
        // Make sure the repo is clean.
        if cmd!(sh, "git status --untracked-files=no --porcelain").read()?.is_empty().not() {
            bail!("working directory must be clean before running `./miri rustc-pull`");
        }
        // Make sure josh is running.
        let josh = Self::start_josh()?;

        // Update rust-version file. As a separate commit, since making it part of
        // the merge has confused the heck out of josh in the past.
        // We pass `--no-verify` to avoid running git hooks like `./miri fmt` that could in turn
        // trigger auto-actions.
        // We do this before the merge so that if there are merge conflicts, we have
        // the right rust-version file while resolving them.
        sh.write_file("rust-version", format!("{commit}\n"))?;
        const PREPARING_COMMIT_MESSAGE: &str = "Preparing for merge from rustc";
        cmd!(sh, "git commit rust-version --no-verify -m {PREPARING_COMMIT_MESSAGE}")
            .run()
            .context("FAILED to commit rust-version file, something went wrong")?;

        // Fetch given rustc commit.
        cmd!(sh, "git fetch http://localhost:{JOSH_PORT}/rust-lang/rust.git@{commit}{JOSH_FILTER}.git")
            .run()
            .map_err(|e| {
                // Try to un-do the previous `git commit`, to leave the repo in the state we found it.
                cmd!(sh, "git reset --hard HEAD^")
                    .run()
                    .expect("FAILED to clean up again after failed `git fetch`, sorry for that");
                e
            })
            .context("FAILED to fetch new commits, something went wrong (committing the rust-version file has been undone)")?;

        // This should not add any new root commits. So count those before and after merging.
        let num_roots = || -> Result<u32> {
            Ok(cmd!(sh, "git rev-list HEAD --max-parents=0 --count")
                .read()
                .context("failed to determine the number of root commits")?
                .parse::<u32>()?)
        };
        let num_roots_before = num_roots()?;

        // Merge the fetched commit.
        const MERGE_COMMIT_MESSAGE: &str = "Merge from rustc";
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

    fn rustc_push(github_user: String, branch: String) -> Result<()> {
        let sh = Shell::new()?;
        sh.change_dir(miri_dir()?);
        let base = sh.read_file("rust-version")?.trim().to_owned();
        // Make sure the repo is clean.
        if cmd!(sh, "git status --untracked-files=no --porcelain").read()?.is_empty().not() {
            bail!("working directory must be clean before running `./miri rustc-push`");
        }
        // Make sure josh is running.
        let josh = Self::start_josh()?;

        // Find a repo we can do our preparation in.
        if let Ok(rustc_git) = env::var("RUSTC_GIT") {
            // If rustc_git is `Some`, we'll use an existing fork for the branch updates.
            sh.change_dir(rustc_git);
        } else {
            // Otherwise, do this in the local Miri repo.
            println!(
                "This will pull a copy of the rust-lang/rust history into this Miri checkout, growing it by about 1GB."
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
        cmd!(sh, "git fetch https://github.com/rust-lang/rust {base}").run()?;
        cmd!(sh, "git push https://github.com/{github_user}/rust {base}:refs/heads/{branch}")
            .ignore_stdout()
            .ignore_stderr() // silence the "create GitHub PR" message
            .run()?;
        println!();

        // Do the actual push.
        sh.change_dir(miri_dir()?);
        println!("Pushing miri changes...");
        cmd!(
            sh,
            "git push http://localhost:{JOSH_PORT}/{github_user}/rust.git{JOSH_FILTER}.git HEAD:{branch}"
        )
        .run()?;
        println!();

        // Do a round-trip check to make sure the push worked as expected.
        cmd!(
            sh,
            "git fetch http://localhost:{JOSH_PORT}/{github_user}/rust.git{JOSH_FILTER}.git {branch}"
        )
        .ignore_stderr()
        .read()?;
        let head = cmd!(sh, "git rev-parse HEAD").read()?;
        let fetch_head = cmd!(sh, "git rev-parse FETCH_HEAD").read()?;
        if head != fetch_head {
            bail!("Josh created a non-roundtrip push! Do NOT merge this into rustc!");
        }
        println!(
            "Confirmed that the push round-trips back to Miri properly. Please create a rustc PR:"
        );
        println!(
            // Open PR with `subtree update` title to silence the `no-merges` triagebot check
            // See https://github.com/rust-lang/rust/pull/114157
            "    https://github.com/rust-lang/rust/compare/{github_user}:{branch}?quick_pull=1&title=Miri+subtree+update&body=r?+@ghost"
        );

        drop(josh);
        Ok(())
    }

    fn bench(target: Option<String>, benches: Vec<String>) -> Result<()> {
        // The hyperfine to use
        let hyperfine = env::var("HYPERFINE");
        let hyperfine = hyperfine.as_deref().unwrap_or("hyperfine -w 1 -m 5 --shell=none");
        let hyperfine = shell_words::split(hyperfine)?;
        let Some((program_name, args)) = hyperfine.split_first() else {
            bail!("expected HYPERFINE environment variable to be non-empty");
        };
        // Make sure we have an up-to-date Miri installed and selected the right toolchain.
        Self::install(vec![])?;

        let sh = Shell::new()?;
        sh.change_dir(miri_dir()?);
        let benches_dir = "bench-cargo-miri";
        let benches: Vec<OsString> = if benches.is_empty() {
            sh.read_dir(benches_dir)?
                .into_iter()
                .filter(|path| path.is_dir())
                .map(Into::into)
                .collect()
        } else {
            benches.into_iter().map(Into::into).collect()
        };
        let target_flag = if let Some(target) = target {
            let mut flag = OsString::from("--target=");
            flag.push(target);
            flag
        } else {
            OsString::new()
        };
        let target_flag = &target_flag;
        // Run the requested benchmarks
        for bench in benches {
            let current_bench = path!(benches_dir / bench / "Cargo.toml");
            // We don't attempt to escape `current_bench`, but we wrap it in quotes.
            // That seems to make Windows CI happy.
            cmd!(
                sh,
                "{program_name} {args...} 'cargo miri run '{target_flag}' --manifest-path \"'{current_bench}'\"'"
            )
            .run()?;
        }
        Ok(())
    }

    fn install(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.install_to_sysroot(e.miri_dir.clone(), &flags)?;
        e.install_to_sysroot(path!(e.miri_dir / "cargo-miri"), &flags)?;
        Ok(())
    }

    fn build(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.build(path!(e.miri_dir / "Cargo.toml"), &flags, /* quiet */ false)?;
        e.build(path!(e.miri_dir / "cargo-miri" / "Cargo.toml"), &flags, /* quiet */ false)?;
        Ok(())
    }

    fn check(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.check(path!(e.miri_dir / "Cargo.toml"), &flags)?;
        e.check(path!(e.miri_dir / "cargo-miri" / "Cargo.toml"), &flags)?;
        Ok(())
    }

    fn clippy(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.clippy(path!(e.miri_dir / "Cargo.toml"), &flags)?;
        e.clippy(path!(e.miri_dir / "cargo-miri" / "Cargo.toml"), &flags)?;
        e.clippy(path!(e.miri_dir / "miri-script" / "Cargo.toml"), &flags)?;
        Ok(())
    }

    fn cargo(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        let toolchain = &e.toolchain;
        // We carefully kept the working dir intact, so this will run cargo *on the workspace in the
        // current working dir*, not on the main Miri workspace. That is exactly what RA needs.
        cmd!(e.sh, "cargo +{toolchain} {flags...}").run()?;
        Ok(())
    }

    fn test(bless: bool, mut flags: Vec<String>, target: Option<String>) -> Result<()> {
        let mut e = MiriEnv::new()?;

        // Prepare a sysroot.
        e.build_miri_sysroot(/* quiet */ false, target.as_deref())?;

        // Forward information to test harness.
        if bless {
            e.sh.set_var("RUSTC_BLESS", "Gesundheit");
        }
        if let Some(target) = target {
            // Tell the harness which target to test.
            e.sh.set_var("MIRI_TEST_TARGET", target);
        }

        // Make sure the flags are going to the test harness, not cargo.
        flags.insert(0, "--".into());

        // Then test, and let caller control flags.
        // Only in root project as `cargo-miri` has no tests.
        e.test(path!(e.miri_dir / "Cargo.toml"), &flags)?;
        Ok(())
    }

    fn run(
        dep: bool,
        verbose: bool,
        many_seeds: Option<Range<u32>>,
        target: Option<String>,
        edition: Option<String>,
        flags: Vec<String>,
    ) -> Result<()> {
        let mut e = MiriEnv::new()?;
        // More flags that we will pass before `flags`
        // (because `flags` may contain `--`).
        let mut early_flags = Vec::<OsString>::new();

        // Add target, edition to flags.
        if let Some(target) = &target {
            early_flags.push("--target".into());
            early_flags.push(target.into());
        }
        if verbose {
            early_flags.push("--verbose".into());
        }
        early_flags.push("--edition".into());
        early_flags.push(edition.as_deref().unwrap_or("2021").into());

        // Prepare a sysroot, add it to the flags.
        let miri_sysroot = e.build_miri_sysroot(/* quiet */ !verbose, target.as_deref())?;
        early_flags.push("--sysroot".into());
        early_flags.push(miri_sysroot.into());

        // Compute everything needed to run the actual command. Also add MIRIFLAGS.
        let miri_manifest = path!(e.miri_dir / "Cargo.toml");
        let miri_flags = e.sh.var("MIRIFLAGS").unwrap_or_default();
        let miri_flags = flagsplit(&miri_flags);
        let toolchain = &e.toolchain;
        let extra_flags = &e.cargo_extra_flags;
        let quiet_flag = if verbose { None } else { Some("--quiet") };
        // This closure runs the command with the given `seed_flag` added between the MIRIFLAGS and
        // the `flags` given on the command-line.
        let run_miri = |sh: &Shell, seed_flag: Option<String>| -> Result<()> {
            // The basic command that executes the Miri driver.
            let mut cmd = if dep {
                cmd!(
                    sh,
                    "cargo +{toolchain} {quiet_flag...} test {extra_flags...} --manifest-path {miri_manifest} --test ui -- --miri-run-dep-mode"
                )
            } else {
                cmd!(
                    sh,
                    "cargo +{toolchain} {quiet_flag...} run {extra_flags...} --manifest-path {miri_manifest} --"
                )
            };
            cmd.set_quiet(!verbose);
            // Add Miri flags
            let mut cmd = cmd.args(&miri_flags).args(&seed_flag).args(&early_flags).args(&flags);
            // For `--dep` we also need to set the env var.
            if dep {
                if let Some(target) = &target {
                    cmd = cmd.env("MIRI_TEST_TARGET", target);
                }
            }
            // And run the thing.
            Ok(cmd.run()?)
        };
        // Run the closure once or many times.
        if let Some(seed_range) = many_seeds {
            e.run_many_times(seed_range, |sh, seed| {
                eprintln!("Trying seed: {seed}");
                run_miri(sh, Some(format!("-Zmiri-seed={seed}"))).map_err(|err| {
                    eprintln!("FAILING SEED: {seed}");
                    err
                })
            })?;
        } else {
            run_miri(&e.sh, None)?;
        }
        Ok(())
    }

    fn fmt(flags: Vec<String>) -> Result<()> {
        use itertools::Itertools;

        let e = MiriEnv::new()?;
        let config_path = path!(e.miri_dir / "rustfmt.toml");

        // Collect each rust file in the miri repo.
        let files = WalkDir::new(&e.miri_dir)
            .into_iter()
            .filter_entry(|entry| {
                let name = entry.file_name().to_string_lossy();
                let ty = entry.file_type();
                if ty.is_file() {
                    name.ends_with(".rs")
                } else {
                    // dir or symlink. skip `target` and `.git`.
                    &name != "target" && &name != ".git"
                }
            })
            .filter_ok(|item| item.file_type().is_file())
            .map_ok(|item| item.into_path());

        e.format_files(files, &e.toolchain[..], &config_path, &flags)
    }
}
