//! See <https://github.com/matklad/cargo-xtask/>.
//!
//! This binary defines various auxiliary build commands, which are not
//! expressible with just `cargo`. Notably, it provides tests via `cargo test -p xtask`
//! for code generation and `cargo xtask install` for installation of
//! rust-analyzer server and client.
//!
//! This binary is integrated into the `cargo` command line by using an alias in
//! `.cargo/config`.

#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::disallowed_methods,
    clippy::disallowed_types
)]

mod flags;

mod codegen;
mod dist;
mod install;
mod metrics;
mod pgo;
mod publish;
mod release;
mod tidy;
mod util;

use anyhow::bail;
use std::{env, path::PathBuf};
use xshell::{Shell, cmd};

fn main() -> anyhow::Result<()> {
    let flags = flags::Xtask::from_env_or_exit();

    let sh = &Shell::new()?;
    sh.change_dir(project_root());

    match flags.subcommand {
        flags::XtaskCmd::Install(cmd) => cmd.run(sh),
        flags::XtaskCmd::FuzzTests(_) => run_fuzzer(sh),
        flags::XtaskCmd::Release(cmd) => cmd.run(sh),
        flags::XtaskCmd::Dist(cmd) => cmd.run(sh),
        flags::XtaskCmd::PublishReleaseNotes(cmd) => cmd.run(sh),
        flags::XtaskCmd::Metrics(cmd) => cmd.run(sh),
        flags::XtaskCmd::Codegen(cmd) => cmd.run(sh),
        flags::XtaskCmd::Bb(cmd) => {
            {
                let _d = sh.push_dir("./crates/rust-analyzer");
                cmd!(sh, "cargo build --release --features jemalloc").run()?;
            }
            sh.copy_file(
                "./target/release/rust-analyzer",
                format!("./target/rust-analyzer-{}", cmd.suffix),
            )?;
            Ok(())
        }
        flags::XtaskCmd::Tidy(cmd) => cmd.run(sh),
    }
}

/// Returns the path to the root directory of `rust-analyzer` project.
fn project_root() -> PathBuf {
    let dir =
        env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned());
    PathBuf::from(dir).parent().unwrap().to_owned()
}

fn run_fuzzer(sh: &Shell) -> anyhow::Result<()> {
    let _d = sh.push_dir("./crates/syntax");
    let _e = sh.push_env("RUSTUP_TOOLCHAIN", "nightly");
    if cmd!(sh, "cargo fuzz --help").read().is_err() {
        cmd!(sh, "cargo install cargo-fuzz").run()?;
    };

    // Expecting nightly rustc
    let out = cmd!(sh, "rustc --version").read()?;
    if !out.contains("nightly") {
        bail!("fuzz tests require nightly rustc")
    }

    cmd!(sh, "cargo fuzz run parser").run()?;
    Ok(())
}

fn date_iso(sh: &Shell) -> anyhow::Result<String> {
    let res = cmd!(sh, "date -u +%Y-%m-%d").read()?;
    Ok(res)
}

fn is_release_tag(tag: &str) -> bool {
    tag.len() == "2020-02-24".len() && tag.starts_with(|c: char| c.is_ascii_digit())
}
