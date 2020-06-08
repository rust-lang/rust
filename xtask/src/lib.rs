//! Support library for `cargo xtask` command.
//!
//! See https://github.com/matklad/cargo-xtask/

pub mod not_bash;
pub mod install;
pub mod release;
pub mod dist;
pub mod pre_commit;

pub mod codegen;
mod ast_src;

use std::{
    env,
    path::{Path, PathBuf},
};

use walkdir::{DirEntry, WalkDir};

use crate::{
    codegen::Mode,
    not_bash::{fs2, pushd, pushenv, rm_rf, run},
};

pub use anyhow::{bail, Context as _, Result};

pub fn project_root() -> PathBuf {
    Path::new(
        &env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
    )
    .ancestors()
    .nth(1)
    .unwrap()
    .to_path_buf()
}

pub fn rust_files(path: &Path) -> impl Iterator<Item = PathBuf> {
    let iter = WalkDir::new(path);
    return iter
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
        .map(|e| e.unwrap())
        .filter(|e| !e.file_type().is_dir())
        .map(|e| e.into_path())
        .filter(|path| path.extension().map(|it| it == "rs").unwrap_or(false));

    fn is_hidden(entry: &DirEntry) -> bool {
        entry.file_name().to_str().map(|s| s.starts_with('.')).unwrap_or(false)
    }
}

pub fn run_rustfmt(mode: Mode) -> Result<()> {
    let _dir = pushd(project_root());
    let _e = pushenv("RUSTUP_TOOLCHAIN", "stable");
    ensure_rustfmt()?;
    match mode {
        Mode::Overwrite => run!("cargo fmt"),
        Mode::Verify => run!("cargo fmt -- --check"),
    }?;
    Ok(())
}

fn reformat(text: impl std::fmt::Display) -> Result<String> {
    let _e = pushenv("RUSTUP_TOOLCHAIN", "stable");
    ensure_rustfmt()?;
    let stdout = run!(
        "rustfmt --config-path {} --config fn_single_line=true", project_root().join("rustfmt.toml").display();
        <text.to_string().as_bytes()
    )?;
    let preamble = "Generated file, do not edit by hand, see `xtask/src/codegen`";
    Ok(format!("//! {}\n\n{}\n", preamble, stdout))
}

fn ensure_rustfmt() -> Result<()> {
    let out = run!("rustfmt --version")?;
    if !out.contains("stable") {
        bail!(
            "Failed to run rustfmt from toolchain 'stable'. \
             Please run `rustup component add rustfmt --toolchain stable` to install it.",
        )
    }
    Ok(())
}

pub fn run_clippy() -> Result<()> {
    if run!("cargo clippy --version").is_err() {
        bail!(
            "Failed run cargo clippy. \
            Please run `rustup component add clippy` to install it.",
        )
    }

    let allowed_lints = [
        "clippy::collapsible_if",
        "clippy::needless_pass_by_value",
        "clippy::nonminimal_bool",
        "clippy::redundant_pattern_matching",
    ];
    run!("cargo clippy --all-features --all-targets -- -A {}", allowed_lints.join(" -A "))?;
    Ok(())
}

pub fn run_fuzzer() -> Result<()> {
    let _d = pushd("./crates/ra_syntax");
    let _e = pushenv("RUSTUP_TOOLCHAIN", "nightly");
    if run!("cargo fuzz --help").is_err() {
        run!("cargo install cargo-fuzz")?;
    };

    // Expecting nightly rustc
    let out = run!("rustc --version")?;
    if !out.contains("nightly") {
        bail!("fuzz tests require nightly rustc")
    }

    run!("cargo fuzz run parser")?;
    Ok(())
}

/// Cleans the `./target` dir after the build such that only
/// dependencies are cached on CI.
pub fn run_pre_cache() -> Result<()> {
    let slow_tests_cookie = Path::new("./target/.slow_tests_cookie");
    if !slow_tests_cookie.exists() {
        panic!("slow tests were skipped on CI!")
    }
    rm_rf(slow_tests_cookie)?;

    for entry in Path::new("./target/debug").read_dir()? {
        let entry = entry?;
        if entry.file_type().map(|it| it.is_file()).ok() == Some(true) {
            // Can't delete yourself on windows :-(
            if !entry.path().ends_with("xtask.exe") {
                rm_rf(&entry.path())?
            }
        }
    }

    fs2::remove_file("./target/.rustc_info.json")?;
    let to_delete = ["ra_", "heavy_test", "xtask"];
    for &dir in ["./target/debug/deps", "target/debug/.fingerprint"].iter() {
        for entry in Path::new(dir).read_dir()? {
            let entry = entry?;
            if to_delete.iter().any(|&it| entry.path().display().to_string().contains(it)) {
                // Can't delete yourself on windows :-(
                if !entry.path().ends_with("xtask.exe") {
                    rm_rf(&entry.path())?
                }
            }
        }
    }

    Ok(())
}

fn is_release_tag(tag: &str) -> bool {
    tag.len() == "2020-02-24".len() && tag.starts_with(|c: char| c.is_ascii_digit())
}
