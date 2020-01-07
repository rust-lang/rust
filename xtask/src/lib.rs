//! FIXME: write short doc here

mod cmd;
pub mod codegen;
pub mod install;
pub mod pre_commit;
mod ast_src;

use anyhow::Context;
use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use crate::codegen::Mode;

pub use anyhow::Result;
pub use cmd::{run, run_with_output, Cmd};

const TOOLCHAIN: &str = "stable";

pub fn project_root() -> PathBuf {
    Path::new(
        &env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
    )
    .ancestors()
    .nth(1)
    .unwrap()
    .to_path_buf()
}

pub fn run_rustfmt(mode: Mode) -> Result<()> {
    match Command::new("rustup")
        .args(&["run", TOOLCHAIN, "--", "cargo", "fmt", "--version"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => install_rustfmt().context("install rustfmt")?,
    };

    if mode == Mode::Verify {
        run(&format!("rustup run {} -- cargo fmt -- --check", TOOLCHAIN), ".")?;
    } else {
        run(&format!("rustup run {} -- cargo fmt", TOOLCHAIN), ".")?;
    }
    Ok(())
}

fn install_rustfmt() -> Result<()> {
    run(&format!("rustup toolchain install {}", TOOLCHAIN), ".")?;
    run(&format!("rustup component add rustfmt --toolchain {}", TOOLCHAIN), ".")
}

pub fn run_clippy() -> Result<()> {
    match Command::new("rustup")
        .args(&["run", TOOLCHAIN, "--", "cargo", "clippy", "--version"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => install_clippy().context("install clippy")?,
    };

    let allowed_lints = [
        "clippy::collapsible_if",
        "clippy::map_clone", // FIXME: remove when Iterator::copied stabilizes (1.36.0)
        "clippy::needless_pass_by_value",
        "clippy::nonminimal_bool",
        "clippy::redundant_pattern_matching",
    ];
    run(
        &format!(
            "rustup run {} -- cargo clippy --all-features --all-targets -- -A {}",
            TOOLCHAIN,
            allowed_lints.join(" -A ")
        ),
        ".",
    )?;
    Ok(())
}

pub fn install_clippy() -> Result<()> {
    run(&format!("rustup toolchain install {}", TOOLCHAIN), ".")?;
    run(&format!("rustup component add clippy --toolchain {}", TOOLCHAIN), ".")
}

pub fn run_fuzzer() -> Result<()> {
    match Command::new("cargo")
        .args(&["fuzz", "--help"])
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => (),
        _ => run("cargo install cargo-fuzz", ".")?,
    };

    run("rustup run nightly -- cargo fuzz run parser", "./crates/ra_syntax")
}
