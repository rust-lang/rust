//! Support library for `cargo xtask` command.
//!
//! See https://github.com/matklad/cargo-xtask/

pub mod not_bash;
pub mod codegen;
mod ast_src;

pub mod install;
pub mod release;
pub mod dist;
pub mod pre_commit;
pub mod metrics;
pub mod pre_cache;

use std::{
    env,
    path::{Path, PathBuf},
};

use walkdir::{DirEntry, WalkDir};

use crate::{
    codegen::Mode,
    not_bash::{pushd, pushenv},
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
    let _d = pushd("./crates/syntax");
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

fn is_release_tag(tag: &str) -> bool {
    tag.len() == "2020-02-24".len() && tag.starts_with(|c: char| c.is_ascii_digit())
}
