use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::ops::{Not, Range};
use std::path::PathBuf;
use std::time::Duration;
use std::{env, net, process};

use anyhow::{Context, Result, anyhow, bail};
use path_macro::path;
use walkdir::WalkDir;
use xshell::{Shell, cmd};

use crate::Command;
use crate::util::*;

/// Used for rustc syncs.
const JOSH_FILTER: &str =
    ":rev(75dd959a3a40eb5b4574f8d2e23aa6efbeb33573:prefix=src/tools/miri):/src/tools/miri";
const JOSH_PORT: u16 = 42042;

impl MiriEnv {
    /// Prepares the environment: builds miri and cargo-miri and a sysroot.
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

        // Make sure everything is built. Also Miri itself.
        self.build(".", &[], quiet)?;
        self.build("cargo-miri", &[], quiet)?;

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

        let mut cmd = self
            .cargo_cmd("cargo-miri", "run")
            .arg("--quiet")
            .arg("--")
            .args(&["miri", "setup", "--print-sysroot"])
            .args(target_flag);
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

    pub fn exec(self) -> Result<()> {
        // First, and crucially only once, run the auto-actions -- but not for all commands.
        match &self {
            Command::Install { .. }
            | Command::Build { .. }
            | Command::Check { .. }
            | Command::Test { .. }
            | Command::Run { .. }
            | Command::Fmt { .. }
            | Command::Doc { .. }
            | Command::Clippy { .. } => Self::auto_actions()?,
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
            Command::Doc { flags } => Self::doc(flags),
            Command::Fmt { flags } => Self::fmt(flags),
            Command::Clippy { flags } => Self::clippy(flags),
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
        let josh_url =
            format!("http://localhost:{JOSH_PORT}/rust-lang/rust.git@{commit}{JOSH_FILTER}.git");

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
        let josh_url =
            format!("http://localhost:{JOSH_PORT}/{github_user}/rust.git{JOSH_FILTER}.git");

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
        e.build(".", &flags, /* quiet */ false)?;
        e.build("cargo-miri", &flags, /* quiet */ false)?;
        Ok(())
    }

    fn check(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.check(".", &flags)?;
        e.check("cargo-miri", &flags)?;
        Ok(())
    }

    fn doc(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.doc(".", &flags)?;
        e.doc("cargo-miri", &flags)?;
        Ok(())
    }

    fn clippy(flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.clippy(".", &flags)?;
        e.clippy("cargo-miri", &flags)?;
        e.clippy("miri-script", &flags)?;
        Ok(())
    }

    fn test(bless: bool, mut flags: Vec<String>, target: Option<String>) -> Result<()> {
        let mut e = MiriEnv::new()?;

        // Prepare a sysroot. (Also builds cargo-miri, which we need.)
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
        e.test(".", &flags)?;
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

        // Preparation: get a sysroot, and get the miri binary.
        let miri_sysroot = e.build_miri_sysroot(/* quiet */ !verbose, target.as_deref())?;
        let miri_bin =
            e.build_get_binary(".").context("failed to get filename of miri executable")?;

        // More flags that we will pass before `flags`
        // (because `flags` may contain `--`).
        let mut early_flags = Vec::<OsString>::new();

        // In `dep` mode, the target is already passed via `MIRI_TEST_TARGET`
        if !dep {
            if let Some(target) = &target {
                early_flags.push("--target".into());
                early_flags.push(target.into());
            }
        }
        early_flags.push("--edition".into());
        early_flags.push(edition.as_deref().unwrap_or("2021").into());
        early_flags.push("--sysroot".into());
        early_flags.push(miri_sysroot.into());

        // Compute everything needed to run the actual command. Also add MIRIFLAGS.
        let miri_flags = e.sh.var("MIRIFLAGS").unwrap_or_default();
        let miri_flags = flagsplit(&miri_flags);
        let quiet_flag = if verbose { None } else { Some("--quiet") };
        // This closure runs the command with the given `seed_flag` added between the MIRIFLAGS and
        // the `flags` given on the command-line.
        let run_miri = |e: &MiriEnv, seed_flag: Option<String>| -> Result<()> {
            // The basic command that executes the Miri driver.
            let mut cmd = if dep {
                // We invoke the test suite as that has all the logic for running with dependencies.
                e.cargo_cmd(".", "test")
                    .args(&["--test", "ui"])
                    .args(quiet_flag)
                    .arg("--")
                    .args(&["--miri-run-dep-mode"])
            } else {
                cmd!(e.sh, "{miri_bin}")
            };
            cmd.set_quiet(!verbose);
            // Add Miri flags
            let mut cmd = cmd.args(&miri_flags).args(&seed_flag).args(&early_flags).args(&flags);
            // For `--dep` we also need to set the target in the env var.
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
            e.run_many_times(seed_range, |e, seed| {
                eprintln!("Trying seed: {seed}");
                run_miri(e, Some(format!("-Zmiri-seed={seed}"))).inspect_err(|_| {
                    eprintln!("FAILING SEED: {seed}");
                })
            })?;
        } else {
            run_miri(&e, None)?;
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

        e.format_files(files, &config_path, &flags)
    }
}
