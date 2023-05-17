use std::collections::BTreeMap;

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use dunce::canonicalize;
use path_macro::path;
use xshell::{cmd, Shell};

use walkdir::WalkDir;

use crate::arg::Subcommands;

/// Used for rustc syncs.
const JOSH_FILTER: &str =
    ":rev(75dd959a3a40eb5b4574f8d2e23aa6efbeb33573:prefix=src/tools/miri):/src/tools/miri";

fn detect_miri_dir() -> std::io::Result<PathBuf> {
    const MIRI_SCRIPT_ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");
    Ok(canonicalize(MIRI_SCRIPT_ROOT_DIR)?.parent().unwrap().into())
}

/// Queries an active toolchain for `dir` via `rustup`.
fn get_active_toolchain(dir: &Path) -> Result<String> {
    let sh = Shell::new()?;
    sh.change_dir(dir);
    let stdout = cmd!(sh, "rustup show active-toolchain").read()?;
    Ok(stdout.split_whitespace().next().context("Could not obtain active Rust toolchain")?.into())
}

#[derive(Clone, Debug)]
pub(super) struct MiriRunner<'a> {
    /// miri_dir is the root of the miri repository checkout we are working in.
    miri_dir: PathBuf,
    /// active_toolchain is passed as `+toolchain` argument to cargo/rustc invocations.
    active_toolchain: String,
    cargo_extra_flags: Vec<String>,
    command: &'a super::Subcommands,
    /// Environment variables passed to child processes.
    env: BTreeMap<OsString, OsString>,
    /// Additional variables used by environment-altering commands.
    /// These should be accessed by corresponding methods (e.g. `sysroot()`) and not directly.
    sysroot: Option<PathBuf>,
}

fn shell_with_parent_env() -> Result<Shell> {
    let sh = Shell::new()?;
    // xshell does not propagate parent's env variables by default.
    for (k, v) in std::env::vars_os() {
        sh.set_var(k, v);
    }
    Ok(sh)
}

impl MiriRunner<'_> {
    pub(super) fn exec(command: &super::Subcommands) -> Result<()> {
        Self::exec_inner(command, true)
    }
    fn exec_inner(command: &super::Subcommands, run_auto_things: bool) -> Result<()> {
        let miri_dir = detect_miri_dir()?;
        let active_toolchain = get_active_toolchain(&miri_dir)?;
        let config = command.get_config(&miri_dir);
        // CARGO_EXTRA_FLAGS do not have to be a valid UTF-8, but that's what shell_words' expects.
        let cargo_extra_flags = std::env::var("CARGO_EXTRA_FLAGS").unwrap_or_default();
        let cargo_extra_flags = shell_words::split(&cargo_extra_flags)?;
        let env = BTreeMap::new();

        let mut runner = MiriRunner {
            miri_dir,
            active_toolchain,
            command,
            env,
            cargo_extra_flags,
            sysroot: None,
        };
        if let Some(config) = config {
            // Run the auto-things.
            if run_auto_things {
                if config.toolchain {
                    // Run this first, so that the toolchain doesn't change after
                    // other code has run.
                    let command = Subcommands::Toolchain { flags: vec![] };
                    Self::exec_inner(&command, false)?;
                    // Let's make sure to actually use that toolchain, too.
                    runner.active_toolchain = "miri".to_owned();
                }
                if config.fmt {
                    let command = Subcommands::Fmt { flags: vec![] };
                    Self::exec_inner(&command, false)?;
                }
                if config.clippy {
                    let command = Subcommands::Clippy {
                        flags: ["--", "-D", "warnings"].into_iter().map(OsString::from).collect(),
                    };
                    Self::exec_inner(&command, false)?;
                }
            }

            // Prepare the environment
            // Determine some toolchain properties
            let libdir = runner.libdir()?;
            if !libdir.exists() {
                println!("Something went wrong determining the library dir.");
                println!("I got {} but that does not exist.", libdir.display());
                println!("Please report a bug at https://github.com/rust-lang/miri/issues.");
                std::process::exit(2);
            }
            // Share target dir between `miri` and `cargo-miri`.
            let target_dir = std::env::var_os("CARGO_TARGET_DIR")
                .filter(|val| !val.is_empty())
                .unwrap_or_else(|| {
                    let target_dir = path!(runner.miri_dir / "target");
                    target_dir.into()
                });
            runner.set_env("CARGO_TARGET_DIR", target_dir);

            // We configure dev builds to not be unusably slow.
            let devel_opt_level = std::env::var_os("CARGO_PROFILE_DEV_OPT_LEVEL")
                .filter(|val| !val.is_empty())
                .unwrap_or_else(|| "2".into());
            runner.set_env("CARGO_PROFILE_DEV_OPT_LEVEL", devel_opt_level);
            let rustflags = {
                let env = std::env::var_os("RUSTFLAGS");
                let mut flags_with_warnings = OsString::from(
                    "-Zunstable-options -Wrustc::internal -Wrust_2018_idioms -Wunused_lifetimes -Wsemicolon_in_expressions_from_macros ",
                );
                if let Some(value) = env {
                    flags_with_warnings.push(value);
                }
                // We set the rpath so that Miri finds the private rustc libraries it needs.
                let mut flags_with_compiler_settings = OsString::from("-C link-args=-Wl,-rpath,");
                flags_with_compiler_settings.push(&libdir);
                flags_with_compiler_settings.push(flags_with_warnings);
                flags_with_compiler_settings
            };
            runner.set_env("RUSTFLAGS", rustflags);
        }
        runner.execute()
    }
    fn execute(&mut self) -> Result<()> {
        // Run command.
        match self.command {
            Subcommands::Install { flags } => self.install(flags),
            Subcommands::Build { flags } => self.build(flags),
            Subcommands::Check { flags } => self.check(flags),
            Subcommands::Test { bless, flags } => self.test(*bless, flags),
            Subcommands::Run { dep, flags } => self.run(*dep, flags),
            Subcommands::Fmt { flags } => self.fmt(flags),
            Subcommands::Clippy { flags } => self.clippy(flags),
            Subcommands::Cargo { flags } => self.cargo(flags),
            Subcommands::ManySeeds { command, seed_start, seeds } =>
                self.many_seeds(command, *seed_start, *seeds),
            Subcommands::Bench { benches } => self.bench(benches),
            Subcommands::Toolchain { flags } => self.toolchain(flags),
            Subcommands::RustcPull { commit } => self.rustc_pull(commit.clone()),
            Subcommands::RustcPush { github_user, branch } => self.rustc_push(github_user, branch),
        }
    }

    fn set_env(
        &mut self,
        key: impl Into<OsString>,
        value: impl Into<OsString>,
    ) -> Option<OsString> {
        self.env.insert(key.into(), value.into())
    }

    /// Prepare and set MIRI_SYSROOT. Respects `MIRI_TEST_TARGET` and takes into account
    /// locally built vs. distributed rustc.
    fn find_miri_sysroot(&mut self) -> Result<()> {
        let current_sysroot = std::env::var_os("MIRI_SYSROOT").unwrap_or_default();

        if !current_sysroot.is_empty() {
            // Sysroot already set, use that.
            let current_value = self.set_env("MIRI_SYSROOT", &current_sysroot);
            assert!(current_value.is_none() || current_value.unwrap() == current_sysroot);
            return Ok(());
        }
        // We need to build a sysroot.
        let target = std::env::var_os("MIRI_TEST_TARGET").filter(|target| !target.is_empty());
        let sysroot = self.build_miri_sysroot(target.as_deref())?;
        self.set_env("MIRI_SYSROOT", sysroot);
        Ok(())
    }

    /// Build a sysroot and set MIRI_SYSROOT to use it. Arguments are passed to `cargo miri setup`.
    fn build_miri_sysroot(&self, target: Option<&OsStr>) -> Result<String> {
        let manifest_path = path!(self.miri_dir / "cargo-miri" / "Cargo.toml");
        let Self { active_toolchain, cargo_extra_flags, .. } = &self;
        let target_prefix: Option<&OsStr> = target.map(|_| "--target".as_ref());
        let sh = self.shell()?;
        let output = cmd!(sh, "cargo +{active_toolchain} --quiet run {cargo_extra_flags...} --manifest-path {manifest_path} -- miri setup --print-sysroot {target_prefix...} {target...}").read();
        if output.is_err() {
            // Run it again (without `--print-sysroot`) so the user can see the error.
            cmd!(sh, "cargo +{active_toolchain} --quiet run {cargo_extra_flags...} --manifest-path {manifest_path} -- miri setup {target_prefix...} {target...}").run().with_context(|| "`cargo miri setup` failed")?;
        }

        Ok(output?)
    }
    fn build_package(
        // Path to Cargo.toml file of a package to build.
        path: &OsStr,
        toolchain: impl AsRef<OsStr>,
        extra_flags: &[String],
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> Result<()> {
        let sh = Shell::new()?;
        cmd!(sh, "cargo +{toolchain} build {extra_flags...} --manifest-path {path} {args...}")
            .run()?;
        Ok(())
    }
    fn shell(&self) -> Result<Shell> {
        let sh = shell_with_parent_env()?;
        for (k, v) in &self.env {
            sh.set_var(k, v);
        }

        Ok(sh)
    }

    fn libdir(&self) -> Result<PathBuf> {
        let sh = shell_with_parent_env()?;
        let toolchain = &self.active_toolchain;
        let target_output = cmd!(sh, "rustc +{toolchain} --version --verbose").read()?;
        let rustc_meta = rustc_version::version_meta_for(&target_output)?;
        let target = rustc_meta.host;

        let sysroot = cmd!(sh, "rustc +{toolchain} --print sysroot").read()?;

        let sysroot = PathBuf::from(sysroot);
        let libdir = path!(sysroot / "lib" / "rustlib" / target / "lib");
        Ok(libdir)
    }
    fn sysroot(&mut self) -> Result<PathBuf> {
        if let Some(sysroot) = self.sysroot.as_ref() {
            Ok(sysroot.clone())
        } else {
            let sh = shell_with_parent_env()?;
            let toolchain = &self.active_toolchain;

            let sysroot: PathBuf = cmd!(sh, "rustc +{toolchain} --print sysroot").read()?.into();
            self.sysroot = Some(sysroot.clone());
            Ok(sysroot)
        }
    }
    fn install_to_dir(
        &mut self,
        sh: &Shell,
        path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> Result<()> {
        let sysroot = self.sysroot()?;
        let toolchain = &self.active_toolchain;
        let extra_flags = &self.cargo_extra_flags;
        // "--locked" to respect the Cargo.lock file if it exists.
        // Install binaries to the miri toolchain's sysroot so they do not interact with other toolchains.
        cmd!(sh, "cargo +{toolchain} install {extra_flags...} --path {path} --force --root {sysroot} {args...}").run()?;
        Ok(())
    }
}

impl MiriRunner<'_> {
    fn bench(&self, benches: &[OsString]) -> Result<()> {
        // The hyperfine to use
        let hyperfine = std::env::var("HYPERFINE");
        let hyperfine = hyperfine.as_deref().unwrap_or("hyperfine -w 1 -m 5 --shell=none");
        let hyperfine = shell_words::split(hyperfine).unwrap();
        let Some((program_name, args)) = hyperfine.split_first() else {
            bail!("Expected HYPERFINE environment variable to be non-empty");
        };
        // Make sure we have an up-to-date Miri installed
        Self::exec_inner(&Subcommands::Install { flags: vec![] }, false)?;
        let benches_dir = path!(self.miri_dir / "bench-cargo-miri");
        let benches = if benches.is_empty() {
            std::fs::read_dir(&benches_dir)?
                .filter_map(|path| {
                    path.ok()
                        .filter(|dir| dir.file_type().map(|t| t.is_dir()).unwrap_or(false))
                        .map(|p| p.file_name())
                })
                .collect()
        } else {
            benches.to_owned()
        };
        let sh = shell_with_parent_env()?;
        let toolchain = &self.active_toolchain;
        // Run the requested benchmarks
        for bench in benches {
            let current_bench_dir = path!(benches_dir / bench / "Cargo.toml");
            cmd!(
                sh,
                "{program_name} {args...} 'cargo +'{toolchain}' miri run --manifest-path \"'{current_bench_dir}'\"'"
            )
            .run()?;
        }
        Ok(())
    }

    fn toolchain(&self, flags: &[OsString]) -> Result<()> {
        // Make sure rustup-toolchain-install-master is installed.
        which::which("rustup-toolchain-install-master").context("Please install rustup-toolchain-install-master by running 'cargo install rustup-toolchain-install-master'")?;
        let sh = shell_with_parent_env()?;
        sh.change_dir(&self.miri_dir);
        let new_commit = Some(sh.read_file("rust-version")?.trim().to_owned());
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
        if current_commit == new_commit {
            println!("miri toolchain is already at commit {}.", current_commit.unwrap());
            cmd!(sh, "rustup override set miri").run()?;
            return Ok(());
        }
        // Install and setup new toolchain.
        cmd!(sh, "rustup toolchain uninstall miri").run()?;

        cmd!(sh, "rustup-toolchain-install-master -n miri -c cargo -c rust-src -c rustc-dev -c llvm-tools -c rustfmt -c clippy {flags...} -- {new_commit...}").run()?;
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

    fn rustc_pull(&self, commit: Option<String>) -> Result<()> {
        let sh = shell_with_parent_env()?;
        sh.change_dir(&self.miri_dir);
        let commit: String = commit.map(Result::Ok).unwrap_or_else(|| {
            let rust_repo_head =
                cmd!(sh, "git ls-remote https://github.com/rust-lang/rust/ HEAD").read()?;
            rust_repo_head
                .split_whitespace()
                .next()
                .map(|front| front.trim().to_owned())
                .ok_or_else(|| anyhow!("Could not obtain Rust repo HEAD from remote."))
        })?;
        // Update rust-version file. As a separate commit, since making it part of
        // the merge has confused the heck out of josh in the past.
        sh.write_file(path!(self.miri_dir / "rust-version"), &commit)?;
        const PREPARING_COMMIT_MESSAGE: &str = "Preparing for merge from rustc";
        cmd!(sh, "git commit rust-version -m {PREPARING_COMMIT_MESSAGE}")
            .run()
            .context("FAILED to commit rust-version file, something went wrong")?;
        // Fetch given rustc commit and note down which one that was
        cmd!(sh, "git fetch http://localhost:8000/rust-lang/rust.git@{commit}{JOSH_FILTER}.git")
            .run()
            .context("FAILED to fetch new commits, something went wrong")?;
        const MERGE_COMMIT_MESSAGE: &str = "Merge from rustc";
        cmd!(sh, "git merge FETCH_HEAD --no-ff -m {MERGE_COMMIT_MESSAGE}")
            .run()
            .context("FAILED to merge new commits, something went wrong")?;
        Ok(())
    }

    fn rustc_push(&self, github_user: &str, branch: &str) -> Result<()> {
        let rustc_git = std::env::var_os("RUSTC_GIT");
        let working_directory = if let Some(rustc_git) = rustc_git {
            rustc_git
        } else {
            // If rustc_git is `Some`, we'll use an existing fork for the branch updates.
            // Otherwise, do this in the local Miri repo.
            println!(
                "This will pull a copy of the rust-lang/rust history into this Miri checkout, growing it by about 1GB."
            );
            println!(
                "To avoid that, abort now and set the RUSTC_GIT environment variable to an existing rustc checkout. Proceed? [y/N] "
            );
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer)?;
            if answer.trim().to_lowercase() != "y" {
                std::process::exit(1);
            }
            self.miri_dir.clone().into()
        };
        // Prepare the branch. Pushing works much better if we use as base exactly
        // the commit that we pulled from last time, so we use the `rust-version`
        // file as a good approximation of that.
        let rust_version_path = path!(self.miri_dir / "rust-version");
        let base = std::fs::read_to_string(rust_version_path)?.trim().to_owned();
        println!("Preparing {github_user}/rust (base: {base})...)");
        let sh = shell_with_parent_env()?;
        sh.change_dir(working_directory);

        if cmd!(sh, "git fetch https://github.com/{github_user}").read().is_ok() {
            println!(
                "The branch '{branch}' seems to already exist in 'https://github.com/{github_user}'. Please delete it and try again."
            );
            std::process::exit(1);
        }

        cmd!(sh, "git fetch https://github.com/rust-lang/rust {base}").run()?;

        cmd!(sh, "git push https://github.com/{github_user}/rust {base}:refs/heads/{branch}")
            .run()?;
        println!();
        // Do the actual push.
        sh.change_dir(&self.miri_dir);
        println!("Pushing miri changes...");
        cmd!(
            sh,
            "git push http://localhost:8000/{github_user}/rust.git{JOSH_FILTER}.git HEAD:{branch}"
        )
        .run()?;
        // Do a round-trip check to make sure the push worked as expected.
        println!();
        cmd!(
            sh,
            "git fetch http://localhost:8000/{github_user}/rust.git{JOSH_FILTER}.git {branch}"
        )
        .read()?;
        let head = cmd!(sh, "git rev-parse HEAD").read()?;
        let fetch_head = cmd!(sh, "git rev-parse FETCH_HEAD").read()?;
        if head != fetch_head {
            println!("ERROR: Josh created a non-roundtrip push! Do NOT merge this into rustc!");
            std::process::exit(1);
        }
        println!(
            "Confirmed that the push round-trips back to Miri properly. Please create a rustc PR:"
        );
        println!("    https://github.com/{github_user}/rust/pull/new/{branch}");
        Ok(())
    }

    fn install(&mut self, flags: &[OsString]) -> Result<()> {
        let sh = self.shell()?;
        self.install_to_dir(&sh, self.miri_dir.clone(), flags)?;
        let cargo_miri_dir = path!(self.miri_dir / "cargo-miri");
        self.install_to_dir(&sh, cargo_miri_dir, flags)?;
        Ok(())
    }

    fn build(&self, flags: &[OsString]) -> Result<()> {
        // Build, and let caller control flags.
        let miri_manifest = path!(self.miri_dir / "Cargo.toml");
        let cargo_miri_manifest = path!(self.miri_dir / "cargo-miri" / "Cargo.toml");
        Self::build_package(
            miri_manifest.as_ref(),
            &self.active_toolchain,
            &self.cargo_extra_flags,
            flags,
        )?;
        Self::build_package(
            cargo_miri_manifest.as_ref(),
            &self.active_toolchain,
            &self.cargo_extra_flags,
            flags,
        )?;
        Ok(())
    }

    fn check(&self, flags: &[OsString]) -> Result<()> {
        fn check_package(
            // Path to Cargo.toml file of a package to check.
            path: &OsStr,
            toolchain: impl AsRef<OsStr>,
            extra_flags: &[String],
            all_targets: bool,
            args: impl IntoIterator<Item = impl AsRef<OsStr>>,
        ) -> Result<()> {
            let all_targets: Option<&OsStr> = all_targets.then_some("--all-targets".as_ref());
            let sh = Shell::new()?;
            cmd!(sh, "cargo +{toolchain} check {extra_flags...} --manifest-path {path} {all_targets...} {args...}").run()?;
            Ok(())
        }
        // Check, and let caller control flags.
        let miri_manifest = path!(self.miri_dir / "Cargo.toml");
        let cargo_miri_manifest = path!(self.miri_dir / "cargo-miri" / "Cargo.toml");
        check_package(
            miri_manifest.as_ref(),
            &self.active_toolchain,
            &self.cargo_extra_flags,
            true,
            flags,
        )?;
        check_package(
            cargo_miri_manifest.as_ref(),
            &self.active_toolchain,
            &self.cargo_extra_flags,
            false,
            flags,
        )?;
        Ok(())
    }

    fn test(&mut self, bless: bool, flags: &[OsString]) -> Result<()> {
        let miri_manifest = path!(self.miri_dir / "Cargo.toml");
        // First build and get a sysroot.
        Self::build_package(
            miri_manifest.as_ref(),
            &self.active_toolchain,
            &self.cargo_extra_flags,
            std::iter::empty::<OsString>(),
        )?;
        self.find_miri_sysroot()?;
        let extra_flags = &self.cargo_extra_flags;
        // Then test, and let caller control flags.
        // Only in root project as `cargo-miri` has no tests.
        let sh = self.shell()?;
        if bless {
            sh.set_var("MIRI_BLESS", "Gesundheit");
        }
        let toolchain: &OsStr = self.active_toolchain.as_ref();
        cmd!(
            sh,
            "cargo +{toolchain} test {extra_flags...} --manifest-path {miri_manifest} -- {flags...}"
        )
        .run()?;
        Ok(())
    }

    fn run(&mut self, dep: bool, flags: &[OsString]) -> Result<()> {
        use itertools::Itertools;
        // Scan for "--target" to overwrite the "MIRI_TEST_TARGET" env var so
        // that we set the MIRI_SYSROOT up the right way.
        let target = flags.iter().tuple_windows().find(|(first, _)| first == &"--target");
        if let Some((_, target)) = target {
            // Found it!
            self.set_env("MIRI_TEST_TARGET", target);
        } else if let Some(var) =
            std::env::var_os("MIRI_TEST_TARGET").filter(|target| !target.is_empty())
        {
            // Make sure miri actually uses this target.
            let entry = self.env.entry("MIRIFLAGS".into()).or_default();
            entry.push(" --target ");
            entry.push(var);
        }
        // First build and get a sysroot.
        let miri_manifest = path!(self.miri_dir / "Cargo.toml");
        Self::build_package(
            miri_manifest.as_ref(),
            &self.active_toolchain,
            &self.cargo_extra_flags,
            std::iter::empty::<OsString>(),
        )?;
        self.find_miri_sysroot()?;
        // Then run the actual command.
        let miri_flags = self.env.get(&OsString::from("MIRIFLAGS")).cloned().unwrap_or_default();
        let miri_flags: &OsStr = miri_flags.as_ref();
        let extra_flags = &self.cargo_extra_flags;
        let sh = self.shell()?;
        let toolchain: &OsStr = self.active_toolchain.as_ref();
        if dep {
            cmd!(
                sh,
                "cargo +{toolchain} --quiet test --test compiletest {extra_flags...} --manifest-path {miri_manifest} -- --miri-run-dep-mode {miri_flags} {flags...}"
            ).run()?;
        } else {
            cmd!(
                sh,
                "cargo +{toolchain} --quiet run {extra_flags...} --manifest-path {miri_manifest} -- {miri_flags} {flags...}"
            ).run()?;
        }
        Ok(())
    }

    fn fmt(&self, flags: &[OsString]) -> Result<()> {
        let toolchain = &self.active_toolchain;
        let config_path = path!(self.miri_dir / "rustfmt.toml");
        let sh = self.shell()?;
        for item in WalkDir::new(&self.miri_dir).into_iter().filter_entry(|entry| {
            let name: String = entry.file_name().to_string_lossy().into();
            let ty = entry.file_type();
            if ty.is_file() {
                name.ends_with(".rs")
            } else {
                // dir or symlink
                &name != "target"
            }
        }) {
            let item = item.unwrap(); // Should never panic as we've already filtered out failed entries.
            if item.file_type().is_file() {
                let path = item.path();
                cmd!(sh, "rustfmt +{toolchain} --edition=2021 --config-path {config_path} {flags...} {path}").quiet().run()?;
            }
        }
        Ok(())
    }

    fn clippy(&self, flags: &[OsString]) -> Result<()> {
        let toolchain_modifier = &self.active_toolchain;
        let extra_flags = &self.cargo_extra_flags;
        let miri_manifest = path!(self.miri_dir / "Cargo.toml");
        let sh = self.shell()?;
        cmd!(sh, "cargo +{toolchain_modifier} clippy {extra_flags...} --manifest-path {miri_manifest} --all-targets -- {flags...}").run()?;
        Ok(())
    }

    fn cargo(&self, flags: &[OsString]) -> Result<()> {
        // We carefully kept the working dir intact, so this will run cargo *on the workspace in the
        // current working dir*, not on the main Miri workspace. That is exactly what RA needs.
        let toolchain_modifier = &self.active_toolchain;
        let sh = self.shell()?;
        cmd!(sh, "cargo +{toolchain_modifier} {flags...}").run()?;
        Ok(())
    }
    fn many_seeds(&self, command: &[OsString], seed_start: u64, seed_count: u64) -> Result<()> {
        let seed_end = seed_start + seed_count;
        assert!(!command.is_empty());
        let (command_name, trailing_args) = command.split_first().unwrap();
        let sh = shell_with_parent_env()?;
        for seed in seed_start..seed_end {
            println!("Trying seed: {seed}");
            let mut miriflags = std::env::var_os("MIRIFLAGS").unwrap_or_default();
            miriflags.push(format!(" -Zlayout-seed={seed} -Zmiri-seed={seed}"));
            let status =
                cmd!(sh, "{command_name} {trailing_args...}").env("MIRIFLAGS", miriflags).run();
            if status.is_err() {
                println!("Failing seed: {seed}");
                break;
            }
        }
        Ok(())
    }
}
