//! See https://github.com/matklad/cargo-xtask/.
//!
//! This binary defines various auxiliary build commands, which are not
//! expressible with just `cargo`. Notably, it provides `cargo xtask codegen`
//! for code generation and `cargo xtask install` for installation of
//! rust-analyzer server and client.
//!
//! This binary is integrated into the `cargo` command line by using an alias in
//! `.cargo/config`.
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
use codegen::CodegenCmd;
use pico_args::Arguments;
use std::{
    env,
    path::{Path, PathBuf},
};
use walkdir::{DirEntry, WalkDir};
use xshell::{cmd, cp, pushd, pushenv};

use crate::{
    codegen::Mode,
    dist::DistCmd,
    install::{InstallCmd, Malloc, ServerOpt},
    metrics::MetricsCmd,
    pre_cache::PreCacheCmd,
    release::{PromoteCmd, ReleaseCmd},
};

fn main() -> Result<()> {
    let _d = pushd(project_root())?;

    let mut args = Arguments::from_env();
    let subcommand = args.subcommand()?.unwrap_or_default();

    match subcommand.as_str() {
        "install" => {
            if args.contains(["-h", "--help"]) {
                eprintln!(
                    "\
cargo xtask install
Install rust-analyzer server or editor plugin.

USAGE:
    cargo xtask install [FLAGS]

FLAGS:
        --client[=CLIENT] Install only VS Code plugin.
                          CLIENT is one of 'code', 'code-exploration', 'code-insiders', 'codium', or 'code-oss'
        --server          Install only the language server
        --mimalloc        Use mimalloc allocator for server
        --jemalloc        Use jemalloc allocator for server
    -h, --help            Prints help information
        "
                );
                return Ok(());
            }
            let server = args.contains("--server");
            let client_code = args.contains("--client");
            if server && client_code {
                eprintln!(
                    "error: The argument `--server` cannot be used with `--client`\n\n\
                     For more information try --help"
                );
                return Ok(());
            }

            let malloc = if args.contains("--mimalloc") {
                Malloc::Mimalloc
            } else if args.contains("--jemalloc") {
                Malloc::Jemalloc
            } else {
                Malloc::System
            };

            let client_opt = args.opt_value_from_str("--client")?;

            finish_args(args)?;

            InstallCmd {
                client: if server { None } else { Some(client_opt.unwrap_or_default()) },
                server: if client_code { None } else { Some(ServerOpt { malloc }) },
            }
            .run()
        }
        "codegen" => {
            let features = args.contains("--features");
            finish_args(args)?;
            CodegenCmd { features }.run()
        }
        "lint" => {
            finish_args(args)?;
            run_clippy()
        }
        "fuzz-tests" => {
            finish_args(args)?;
            run_fuzzer()
        }
        "pre-cache" => {
            finish_args(args)?;
            PreCacheCmd.run()
        }
        "release" => {
            let dry_run = args.contains("--dry-run");
            finish_args(args)?;
            ReleaseCmd { dry_run }.run()
        }
        "promote" => {
            let dry_run = args.contains("--dry-run");
            finish_args(args)?;
            PromoteCmd { dry_run }.run()
        }
        "dist" => {
            let nightly = args.contains("--nightly");
            let client_version: Option<String> = args.opt_value_from_str("--client")?;
            finish_args(args)?;
            DistCmd { nightly, client_version }.run()
        }
        "metrics" => {
            let dry_run = args.contains("--dry-run");
            finish_args(args)?;
            MetricsCmd { dry_run }.run()
        }
        "bb" => {
            let suffix: String = args.free_from_str()?;
            finish_args(args)?;
            {
                let _d = pushd("./crates/rust-analyzer")?;
                cmd!("cargo build --release --features jemalloc").run()?;
            }
            cp("./target/release/rust-analyzer", format!("./target/rust-analyzer-{}", suffix))?;
            Ok(())
        }
        _ => {
            eprintln!(
                "\
cargo xtask
Run custom build command.

USAGE:
    cargo xtask <SUBCOMMAND>

SUBCOMMANDS:
    fuzz-tests
    codegen
    install
    lint
    dist
    promote
    bb"
            );
            Ok(())
        }
    }
}

fn finish_args(args: Arguments) -> Result<()> {
    if !args.finish().is_empty() {
        bail!("Unused arguments.");
    }
    Ok(())
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

fn run_rustfmt(mode: Mode) -> Result<()> {
    let _dir = pushd(project_root())?;
    let _e = pushenv("RUSTUP_TOOLCHAIN", "stable");
    ensure_rustfmt()?;
    let check = match mode {
        Mode::Overwrite => &[][..],
        Mode::Verify => &["--", "--check"],
    };
    cmd!("cargo fmt {check...}").run()?;
    Ok(())
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

fn run_clippy() -> Result<()> {
    if cmd!("cargo clippy --version").read().is_err() {
        bail!(
            "Failed run cargo clippy. \
            Please run `rustup component add clippy` to install it.",
        )
    }

    let allowed_lints = "
        -A clippy::collapsible_if
        -A clippy::needless_pass_by_value
        -A clippy::nonminimal_bool
        -A clippy::redundant_pattern_matching
    "
    .split_ascii_whitespace();
    cmd!("cargo clippy --all-features --all-targets -- {allowed_lints...}").run()?;
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
