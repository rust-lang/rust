//! See https://github.com/matklad/cargo-xtask/.
//!
//! This binary defines various auxiliary build commands, which are not
//! expressible with just `cargo`. Notably, it provides tests via `cargo test -p xtask`
//! for code generation and `cargo xtask install` for installation of
//! rust-analyzer server and client.
//!
//! This binary is integrated into the `cargo` command line by using an alias in
//! `.cargo/config`.
mod flags;

mod codegen;
mod ast_src;
#[cfg(test)]
mod tidy;

mod install;
mod release;
mod dist;
mod metrics;
mod pre_cache;

use anyhow::{bail, Result};
use std::{
    env,
    path::{Path, PathBuf},
};
use walkdir::{DirEntry, WalkDir};
use xshell::{cmd, cp, pushd, pushenv};

use crate::dist::DistCmd;

fn main() -> Result<()> {
    let _d = pushd(project_root())?;

    let flags = flags::Xtask::from_env()?;
    match flags.subcommand {
        flags::XtaskCmd::Help(_) => {
            println!("{}", flags::Xtask::HELP);
            Ok(())
        }
        flags::XtaskCmd::Install(cmd) => cmd.run(),
        flags::XtaskCmd::FuzzTests(_) => run_fuzzer(),
        flags::XtaskCmd::PreCache(cmd) => cmd.run(),
        flags::XtaskCmd::Release(cmd) => cmd.run(),
        flags::XtaskCmd::Promote(cmd) => cmd.run(),
        flags::XtaskCmd::Dist(flags) => {
            DistCmd { nightly: flags.nightly, client_version: flags.client }.run()
        }
        flags::XtaskCmd::Metrics(cmd) => cmd.run(),
        flags::XtaskCmd::Bb(cmd) => {
            {
                let _d = pushd("./crates/rust-analyzer")?;
                cmd!("cargo build --release --features jemalloc").run()?;
            }
            cp("./target/release/rust-analyzer", format!("./target/rust-analyzer-{}", cmd.suffix))?;
            Ok(())
        }
    }
}

fn project_root() -> PathBuf {
    Path::new(
        &env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
    )
    .ancestors()
    .nth(1)
    .unwrap()
    .to_path_buf()
}

fn rust_files() -> impl Iterator<Item = PathBuf> {
    rust_files_in(&project_root().join("crates"))
}

#[cfg(test)]
fn cargo_files() -> impl Iterator<Item = PathBuf> {
    files_in(&project_root(), "toml")
        .filter(|path| path.file_name().map(|it| it == "Cargo.toml").unwrap_or(false))
}

fn rust_files_in(path: &Path) -> impl Iterator<Item = PathBuf> {
    files_in(path, "rs")
}

fn ensure_rustfmt() -> Result<()> {
    let out = cmd!("rustfmt --version").read()?;
    if !out.contains("stable") {
        bail!(
            "Failed to run rustfmt from toolchain 'stable'. \
             Please run `rustup component add rustfmt --toolchain stable` to install it.",
        )
    }
    Ok(())
}

fn run_fuzzer() -> Result<()> {
    let _d = pushd("./crates/syntax")?;
    let _e = pushenv("RUSTUP_TOOLCHAIN", "nightly");
    if cmd!("cargo fuzz --help").read().is_err() {
        cmd!("cargo install cargo-fuzz").run()?;
    };

    // Expecting nightly rustc
    let out = cmd!("rustc --version").read()?;
    if !out.contains("nightly") {
        bail!("fuzz tests require nightly rustc")
    }

    cmd!("cargo fuzz run parser").run()?;
    Ok(())
}

fn date_iso() -> Result<String> {
    let res = cmd!("date --iso --utc").read()?;
    Ok(res)
}

fn is_release_tag(tag: &str) -> bool {
    tag.len() == "2020-02-24".len() && tag.starts_with(|c: char| c.is_ascii_digit())
}

fn files_in(path: &Path, ext: &'static str) -> impl Iterator<Item = PathBuf> {
    let iter = WalkDir::new(path);
    return iter
        .into_iter()
        .filter_entry(|e| !is_hidden(e))
        .map(|e| e.unwrap())
        .filter(|e| !e.file_type().is_dir())
        .map(|e| e.into_path())
        .filter(move |path| path.extension().map(|it| it == ext).unwrap_or(false));

    fn is_hidden(entry: &DirEntry) -> bool {
        entry.file_name().to_str().map(|s| s.starts_with('.')).unwrap_or(false)
    }
}
