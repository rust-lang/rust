//! See <https://github.com/matklad/cargo-xtask/>.
//!
//! This binary defines various auxiliary build commands, which are not
//! expressible with just `cargo`. Notably, it provides tests via `cargo test -p xtask`
//! for code generation and `cargo xtask install` for installation of
//! rust-analyzer server and client.
//!
//! This binary is integrated into the `cargo` command line by using an alias in
//! `.cargo/config`.
mod flags;

mod install;
mod release;
mod dist;
mod metrics;

use anyhow::{bail, Result};
use std::{
    env,
    path::{Path, PathBuf},
};
use xshell::{cmd, cp, pushd, pushenv};

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
        flags::XtaskCmd::Release(cmd) => cmd.run(),
        flags::XtaskCmd::Promote(cmd) => cmd.run(),
        flags::XtaskCmd::Dist(cmd) => cmd.run(),
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
    let res = cmd!("date -u +%Y-%m-%d").read()?;
    Ok(res)
}

fn is_release_tag(tag: &str) -> bool {
    tag.len() == "2020-02-24".len() && tag.starts_with(|c: char| c.is_ascii_digit())
}
