//! See https://github.com/matklad/cargo-xtask/.
//!
//! This binary defines various auxiliary build commands, which are not
//! expressible with just `cargo`. Notably, it provides `cargo xtask codegen`
//! for code generation and `cargo xtask install` for installation of
//! rust-analyzer server and client.
//!
//! This binary is integrated into the `cargo` command line by using an alias in
//! `.cargo/config`.

use std::env;

use anyhow::bail;
use codegen::CodegenCmd;
use pico_args::Arguments;
use xshell::{cmd, cp, pushd};
use xtask::{
    codegen::{self, Mode},
    dist::DistCmd,
    install::{InstallCmd, Malloc, ServerOpt},
    metrics::MetricsCmd,
    pre_cache::PreCacheCmd,
    pre_commit, project_root,
    release::{PromoteCmd, ReleaseCmd},
    run_clippy, run_fuzzer, run_rustfmt, Result,
};

fn main() -> Result<()> {
    if env::args().next().map(|it| it.contains("pre-commit")) == Some(true) {
        return pre_commit::run_hook();
    }

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
        "format" => {
            finish_args(args)?;
            run_rustfmt(Mode::Overwrite)
        }
        "install-pre-commit-hook" => {
            finish_args(args)?;
            pre_commit::install_hook()
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
            cmd!("cargo build --release").run()?;
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
    format
    install-pre-commit-hook
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
