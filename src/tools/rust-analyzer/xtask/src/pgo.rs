//! PGO (Profile-Guided Optimization) utilities.

use anyhow::Context;
use std::env::consts::EXE_EXTENSION;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use xshell::{Cmd, Shell, cmd};

use crate::flags::PgoTrainingCrate;

/// Decorates `ra_build_cmd` to add PGO instrumentation, and then runs the PGO instrumented
/// Rust Analyzer on itself to gather a PGO profile.
pub(crate) fn gather_pgo_profile<'a>(
    sh: &'a Shell,
    ra_build_cmd: Cmd<'a>,
    target: &str,
    train_crate: PgoTrainingCrate,
) -> anyhow::Result<PathBuf> {
    let pgo_dir = std::path::absolute("rust-analyzer-pgo")?;
    // Clear out any stale profiles
    if pgo_dir.is_dir() {
        std::fs::remove_dir_all(&pgo_dir)?;
    }
    std::fs::create_dir_all(&pgo_dir)?;

    // Figure out a path to `llvm-profdata`
    let target_libdir = cmd!(sh, "rustc --print=target-libdir")
        .read()
        .context("cannot resolve target-libdir from rustc")?;
    let target_bindir = PathBuf::from(target_libdir).parent().unwrap().join("bin");
    let llvm_profdata = target_bindir.join("llvm-profdata").with_extension(EXE_EXTENSION);

    // Build RA with PGO instrumentation
    let cmd_gather =
        ra_build_cmd.env("RUSTFLAGS", format!("-Cprofile-generate={}", pgo_dir.to_str().unwrap()));
    cmd_gather.run().context("cannot build rust-analyzer with PGO instrumentation")?;

    let (train_path, label) = match &train_crate {
        PgoTrainingCrate::RustAnalyzer => (PathBuf::from("."), "itself"),
        PgoTrainingCrate::GitHub(repo) => {
            (download_crate_for_training(sh, &pgo_dir, repo)?, repo.as_str())
        }
    };

    // Run RA either on itself or on a downloaded crate
    eprintln!("Training RA on {label}...");
    cmd!(
        sh,
        "target/{target}/release/rust-analyzer analysis-stats -q --run-all-ide-things {train_path}"
    )
    .run()
    .context("cannot generate PGO profiles")?;

    // Merge profiles into a single file
    let merged_profile = pgo_dir.join("merged.profdata");
    let profile_files = std::fs::read_dir(pgo_dir)?.filter_map(|entry| {
        let entry = entry.ok()?;
        if entry.path().extension() == Some(OsStr::new("profraw")) {
            Some(entry.path().to_str().unwrap().to_owned())
        } else {
            None
        }
    });
    cmd!(sh, "{llvm_profdata} merge {profile_files...} -o {merged_profile}").run().context(
        "cannot merge PGO profiles. Do you have the rustup `llvm-tools` component installed?",
    )?;

    Ok(merged_profile)
}

/// Downloads a crate from GitHub, stores it into `pgo_dir` and returns a path to it.
fn download_crate_for_training(sh: &Shell, pgo_dir: &Path, repo: &str) -> anyhow::Result<PathBuf> {
    let mut it = repo.splitn(2, '@');
    let repo = it.next().unwrap();
    let revision = it.next();

    // FIXME: switch to `--revision` here around 2035 or so
    let revision =
        if let Some(revision) = revision { &["--branch", revision] as &[&str] } else { &[] };

    let normalized_path = repo.replace("/", "-");
    let target_path = pgo_dir.join(normalized_path);
    cmd!(sh, "git clone --depth 1 https://github.com/{repo} {revision...} {target_path}")
        .run()
        .with_context(|| "cannot download PGO training crate from {repo}")?;

    Ok(target_path)
}

/// Helper function to create a build command for rust-analyzer
pub(crate) fn build_command<'a>(
    sh: &'a Shell,
    command: &str,
    target_name: &str,
    features: &[&str],
) -> Cmd<'a> {
    cmd!(
        sh,
        "cargo {command} --manifest-path ./crates/rust-analyzer/Cargo.toml --bin rust-analyzer --target {target_name} {features...} --release"
    )
}

pub(crate) fn apply_pgo_to_cmd<'a>(cmd: Cmd<'a>, profile_path: &Path) -> Cmd<'a> {
    cmd.env("RUSTFLAGS", format!("-Cprofile-use={}", profile_path.to_str().unwrap()))
}
