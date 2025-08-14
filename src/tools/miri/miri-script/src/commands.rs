use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter};
use std::path::PathBuf;
use std::{env, process};

use anyhow::{Context, Result, anyhow, bail};
use path_macro::path;
use serde_derive::{Deserialize, Serialize};
use tempfile::TempDir;
use walkdir::WalkDir;
use xshell::{Shell, cmd};

use crate::Command;
use crate::util::*;

impl MiriEnv {
    /// Prepares the environment: builds miri and cargo-miri and a sysroot.
    /// Returns the location of the sysroot.
    ///
    /// If the target is None the sysroot will be built for the host machine.
    fn build_miri_sysroot(
        &mut self,
        quiet: bool,
        target: Option<impl AsRef<OsStr>>,
        features: &[String],
    ) -> Result<PathBuf> {
        if let Some(miri_sysroot) = self.sh.var_os("MIRI_SYSROOT") {
            // Sysroot already set, use that.
            return Ok(miri_sysroot.into());
        }

        // Make sure everything is built. Also Miri itself.
        self.build(".", features, &[], quiet)?;
        self.build("cargo-miri", &[], &[], quiet)?;

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
            .cargo_cmd("cargo-miri", "run", &[])
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
            // no features for auto actions, see
            // https://github.com/rust-lang/miri/pull/4396#discussion_r2149654845
            Self::clippy(vec![], vec![])?;
        }

        Ok(())
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
            | Command::Toolchain { .. } | Command::Bench { .. } | Command::Squash => {}
        }
        // Then run the actual command.
        match self {
            Command::Install { features, flags } => Self::install(features, flags),
            Command::Build { features, flags } => Self::build(features, flags),
            Command::Check { features, flags } => Self::check(features, flags),
            Command::Test { bless, target, coverage, features, flags } =>
                Self::test(bless, target, coverage, features, flags),
            Command::Run { dep, verbose, target, edition, features, flags } =>
                Self::run(dep, verbose, target, edition, features, flags),
            Command::Doc { features, flags } => Self::doc(features, flags),
            Command::Fmt { flags } => Self::fmt(flags),
            Command::Clippy { features, flags } => Self::clippy(features, flags),
            Command::Bench { target, no_install, save_baseline, load_baseline, benches } =>
                Self::bench(target, no_install, save_baseline, load_baseline, benches),
            Command::Toolchain { flags } => Self::toolchain(flags),
            Command::Squash => Self::squash(),
        }
    }

    fn toolchain(flags: Vec<String>) -> Result<()> {
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

        cmd!(sh, "rustup-toolchain-install-master -n miri -c cargo -c rust-src -c rustc-dev -c llvm-tools -c rustfmt -c clippy {flags...} -- {new_commit}")
            .run()
            .context("Failed to run rustup-toolchain-install-master. If it is not installed, run 'cargo install rustup-toolchain-install-master'.")?;
        cmd!(sh, "rustup override set miri").run()?;
        // Cleanup.
        cmd!(sh, "cargo clean").run()?;
        Ok(())
    }

    fn squash() -> Result<()> {
        let sh = Shell::new()?;
        sh.change_dir(miri_dir()?);
        // Figure out base wrt latest upstream master.
        // (We can't trust any of the local ones, they can all be outdated.)
        let origin_master = {
            cmd!(sh, "git fetch https://github.com/rust-lang/miri/")
                .quiet()
                .ignore_stdout()
                .ignore_stderr()
                .run()?;
            cmd!(sh, "git rev-parse FETCH_HEAD").read()?
        };
        let base = cmd!(sh, "git merge-base HEAD {origin_master}").read()?;
        // Rebase onto that, setting ourselves as the sequence editor so that we can edit the sequence programmatically.
        // We want to forward the host stdin so apparently we cannot use `cmd!`.
        let mut cmd = process::Command::new("git");
        cmd.arg("rebase").arg(&base).arg("--interactive");
        let current_exe = {
            if cfg!(windows) {
                // Apparently git-for-Windows gets confused by backslashes if we just use
                // `current_exe()` here. So replace them by forward slashes if this is not a "magic"
                // path starting with "\\". This is clearly a git bug but we work around it here.
                // Also see <https://github.com/rust-lang/miri/issues/4340>.
                let bin = env::current_exe()?;
                match bin.into_os_string().into_string() {
                    Err(not_utf8) => not_utf8.into(), // :shrug:
                    Ok(str) => {
                        if str.starts_with(r"\\") {
                            str.into() // don't touch these magic paths, they must use backslashes
                        } else {
                            str.replace('\\', "/").into()
                        }
                    }
                }
            } else {
                env::current_exe()?
            }
        };
        cmd.env("GIT_SEQUENCE_EDITOR", current_exe);
        cmd.env("MIRI_SCRIPT_IS_GIT_SEQUENCE_EDITOR", "1");
        cmd.current_dir(sh.current_dir());
        let result = cmd.status()?;
        if !result.success() {
            bail!("`git rebase` failed");
        }
        Ok(())
    }

    pub fn squash_sequence_editor() -> Result<()> {
        let sequence_file = env::args().nth(1).expect("git should pass us a filename");
        if sequence_file == "fmt" {
            // This is probably us being called as a git hook as part of the rebase. Let's just
            // ignore this. Sadly `git rebase` does not have a flag to skip running hooks.
            return Ok(());
        }
        // Read the provided sequence and adjust it.
        let rebase_sequence = {
            let mut rebase_sequence = String::new();
            let file = fs::File::open(&sequence_file).with_context(|| {
                format!("failed to read rebase sequence from {sequence_file:?}")
            })?;
            let file = io::BufReader::new(file);
            for line in file.lines() {
                let line = line?;
                // The first line is left unchanged.
                if rebase_sequence.is_empty() {
                    writeln!(rebase_sequence, "{line}").unwrap();
                    continue;
                }
                // If this is a "pick" like, make it "squash".
                if let Some(rest) = line.strip_prefix("pick ") {
                    writeln!(rebase_sequence, "squash {rest}").unwrap();
                    continue;
                }
                // We've reached the end of the relevant part of the sequence, and we can stop.
                break;
            }
            rebase_sequence
        };
        // Write out the adjusted sequence.
        fs::write(&sequence_file, rebase_sequence).with_context(|| {
            format!("failed to write adjusted rebase sequence to {sequence_file:?}")
        })?;
        Ok(())
    }

    fn bench(
        target: Option<String>,
        no_install: bool,
        save_baseline: Option<String>,
        load_baseline: Option<String>,
        benches: Vec<String>,
    ) -> Result<()> {
        if save_baseline.is_some() && load_baseline.is_some() {
            bail!("Only one of `--save-baseline` and `--load-baseline` can be set");
        }

        // The hyperfine to use
        let hyperfine = env::var("HYPERFINE");
        let hyperfine = hyperfine.as_deref().unwrap_or("hyperfine -w 1 -m 5 --shell=none");
        let hyperfine = shell_words::split(hyperfine)?;
        let Some((program_name, args)) = hyperfine.split_first() else {
            bail!("expected HYPERFINE environment variable to be non-empty");
        };

        if !no_install {
            // Make sure we have an up-to-date Miri installed and selected the right toolchain.
            Self::install(vec![], vec![])?;
        }
        let results_json_dir = if save_baseline.is_some() || load_baseline.is_some() {
            Some(TempDir::new()?)
        } else {
            None
        };

        let miri_dir = miri_dir()?;
        let sh = Shell::new()?;
        sh.change_dir(&miri_dir);
        let benches_dir = "bench-cargo-miri";
        let benches: Vec<String> = if benches.is_empty() {
            sh.read_dir(benches_dir)?
                .into_iter()
                .filter(|path| path.is_dir())
                // Only keep the basename: that matches the usage with a manual bench list,
                // and it ensure the path concatenations below work as intended.
                .map(|path| path.file_name().unwrap().to_owned().into_string().unwrap())
                .collect()
        } else {
            benches.into_iter().collect()
        };
        let target_flag = if let Some(target) = target {
            let mut flag = OsString::from("--target=");
            flag.push(target);
            flag
        } else {
            OsString::new()
        };
        let target_flag = &target_flag;
        let toolchain = active_toolchain()?;
        // Run the requested benchmarks
        for bench in &benches {
            let current_bench = path!(benches_dir / bench / "Cargo.toml");
            let mut export_json = None;
            if let Some(baseline_temp_dir) = &results_json_dir {
                export_json = Some(format!(
                    "--export-json={}",
                    path!(baseline_temp_dir / format!("{bench}.bench.json")).display()
                ));
            }
            // We don't attempt to escape `current_bench`, but we wrap it in quotes.
            // That seems to make Windows CI happy.
            cmd!(
                sh,
                "{program_name} {args...} {export_json...} 'cargo +'{toolchain}' miri run '{target_flag}' --manifest-path \"'{current_bench}'\"'"
            )
            .run()?;
        }

        // Gather/load results for baseline saving.

        #[derive(Serialize, Deserialize)]
        struct BenchResult {
            mean: f64,
            stddev: f64,
        }

        let gather_results = || -> Result<BTreeMap<&str, BenchResult>> {
            let baseline_temp_dir = results_json_dir.unwrap();
            let mut results = BTreeMap::new();
            for bench in &benches {
                let result = File::open(path!(baseline_temp_dir / format!("{bench}.bench.json")))
                    .context("failed to read hyperfine JSON")?;
                let mut result: serde_json::Value = serde_json::from_reader(BufReader::new(result))
                    .context("failed to parse hyperfine JSON")?;
                let result: BenchResult = serde_json::from_value(result["results"][0].take())
                    .context("failed to interpret hyperfine JSON")?;
                results.insert(bench as &str, result);
            }
            Ok(results)
        };

        if let Some(baseline_file) = save_baseline {
            let results = gather_results()?;
            let baseline = File::create(baseline_file)?;
            serde_json::to_writer_pretty(BufWriter::new(baseline), &results)?;
        } else if let Some(baseline_file) = load_baseline {
            let new_results = gather_results()?;
            let baseline_results: BTreeMap<String, BenchResult> = {
                let f = File::open(baseline_file)?;
                serde_json::from_reader(BufReader::new(f))?
            };
            println!(
                "Comparison with baseline (relative speed, lower is better for the new results):"
            );
            for (bench, new_result) in new_results {
                let Some(baseline_result) = baseline_results.get(bench) else { continue };

                // Compare results (inspired by hyperfine)
                let ratio = new_result.mean / baseline_result.mean;
                // https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae
                // Covariance asssumed to be 0, i.e. variables are assumed to be independent
                let ratio_stddev = ratio
                    * f64::sqrt(
                        (new_result.stddev / new_result.mean).powi(2)
                            + (baseline_result.stddev / baseline_result.mean).powi(2),
                    );

                println!("  {bench}: {ratio:.2} ± {ratio_stddev:.2}");
            }
        }

        Ok(())
    }

    fn install(features: Vec<String>, flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.install_to_sysroot(".", &features, &flags)?;
        e.install_to_sysroot("cargo-miri", &[], &flags)?;
        Ok(())
    }

    fn build(features: Vec<String>, flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.build(".", &features, &flags, /* quiet */ false)?;
        e.build("cargo-miri", &[], &flags, /* quiet */ false)?;
        Ok(())
    }

    fn check(features: Vec<String>, flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.check(".", &features, &flags)?;
        e.check("cargo-miri", &[], &flags)?;
        Ok(())
    }

    fn doc(features: Vec<String>, flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.doc(".", &features, &flags)?;
        e.doc("cargo-miri", &[], &flags)?;
        Ok(())
    }

    fn clippy(features: Vec<String>, flags: Vec<String>) -> Result<()> {
        let e = MiriEnv::new()?;
        e.clippy(".", &features, &flags)?;
        e.clippy("cargo-miri", &[], &flags)?;
        e.clippy("miri-script", &[], &flags)?;
        Ok(())
    }

    fn test(
        bless: bool,
        target: Option<String>,
        coverage: bool,
        features: Vec<String>,
        mut flags: Vec<String>,
    ) -> Result<()> {
        let mut e = MiriEnv::new()?;

        let coverage = coverage.then_some(crate::coverage::CoverageReport::new()?);

        if let Some(report) = &coverage {
            report.add_env_vars(&mut e)?;
        }

        // Prepare a sysroot. (Also builds cargo-miri, which we need.)
        e.build_miri_sysroot(/* quiet */ false, target.as_deref(), &features)?;

        // Forward information to test harness.
        if bless {
            e.sh.set_var("RUSTC_BLESS", "Gesundheit");
        }
        if e.sh.var("MIRI_TEST_TARGET").is_ok() {
            // Avoid trouble due to an incorrectly set env var.
            bail!("MIRI_TEST_TARGET must not be set when invoking `./miri test`");
        }
        if let Some(target) = target {
            // Tell the harness which target to test.
            e.sh.set_var("MIRI_TEST_TARGET", target);
        }

        // Make sure the flags are going to the test harness, not cargo.
        flags.insert(0, "--".into());

        // Then test, and let caller control flags.
        // Only in root project as `cargo-miri` has no tests.
        e.test(".", &features, &flags)?;

        if let Some(coverage) = &coverage {
            coverage.show_coverage_report(&e, &features)?;
        }

        Ok(())
    }

    fn run(
        dep: bool,
        verbose: bool,
        target: Option<String>,
        edition: Option<String>,
        features: Vec<String>,
        flags: Vec<String>,
    ) -> Result<()> {
        let mut e = MiriEnv::new()?;

        // Preparation: get a sysroot, and get the miri binary.
        let miri_sysroot =
            e.build_miri_sysroot(/* quiet */ !verbose, target.as_deref(), &features)?;
        let miri_bin = e
            .build_get_binary(".", &features)
            .context("failed to get filename of miri executable")?;

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

        // Compute flags.
        let miri_flags = e.sh.var("MIRIFLAGS").unwrap_or_default();
        let miri_flags = flagsplit(&miri_flags);
        let quiet_flag = if verbose { None } else { Some("--quiet") };

        // Run Miri.
        // The basic command that executes the Miri driver.
        let mut cmd = if dep {
            // We invoke the test suite as that has all the logic for running with dependencies.
            e.cargo_cmd(".", "test", &features)
                .args(&["--test", "ui"])
                .args(quiet_flag)
                .arg("--")
                .args(&["--miri-run-dep-mode"])
        } else {
            cmd!(e.sh, "{miri_bin}")
        };
        cmd.set_quiet(!verbose);
        // Add Miri flags
        let mut cmd = cmd.args(&miri_flags).args(&early_flags).args(&flags);
        // For `--dep` we also need to set the target in the env var.
        if dep {
            if let Some(target) = &target {
                cmd = cmd.env("MIRI_TEST_TARGET", target);
            }
        }
        // Finally, run the thing.
        Ok(cmd.run()?)
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
                    // dir or symlink. skip `target`, `.git` and `genmc-src*`
                    &name != "target" && &name != ".git" && !name.starts_with("genmc-src")
                }
            })
            .filter_ok(|item| item.file_type().is_file())
            .map_ok(|item| item.into_path());

        e.format_files(files, &config_path, &flags)
    }
}
