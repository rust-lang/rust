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

use pico_args::Arguments;
use xtask::{
    codegen::{self, Mode},
    dist::run_dist,
    install::{ClientOpt, InstallCmd, ServerOpt},
    not_bash::pushd,
    pre_commit, project_root,
    release::ReleaseCmd,
    run_clippy, run_fuzzer, run_pre_cache, run_rustfmt, Result,
};

fn main() -> Result<()> {
    if env::args().next().map(|it| it.contains("pre-commit")) == Some(true) {
        return pre_commit::run_hook();
    }

    let _d = pushd(project_root());

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
        --client-code    Install only VS Code plugin
        --server         Install only the language server
        --jemalloc       Use jemalloc for server
    -h, --help           Prints help information
        "
                );
                return Ok(());
            }
            let server = args.contains("--server");
            let client_code = args.contains("--client-code");
            if server && client_code {
                eprintln!(
                    "error: The argument `--server` cannot be used with `--client-code`\n\n\
                     For more information try --help"
                );
                return Ok(());
            }

            let jemalloc = args.contains("--jemalloc");

            args.finish()?;

            InstallCmd {
                client: if server { None } else { Some(ClientOpt::VsCode) },
                server: if client_code { None } else { Some(ServerOpt { jemalloc }) },
            }
            .run()
        }
        "codegen" => {
            args.finish()?;
            codegen::generate_syntax(Mode::Overwrite)?;
            codegen::generate_parser_tests(Mode::Overwrite)?;
            codegen::generate_assists_tests(Mode::Overwrite)?;
            codegen::generate_assists_docs(Mode::Overwrite)?;
            codegen::generate_feature_docs(Mode::Overwrite)?;
            Ok(())
        }
        "format" => {
            args.finish()?;
            run_rustfmt(Mode::Overwrite)
        }
        "install-pre-commit-hook" => {
            args.finish()?;
            pre_commit::install_hook()
        }
        "lint" => {
            args.finish()?;
            run_clippy()
        }
        "fuzz-tests" => {
            args.finish()?;
            run_fuzzer()
        }
        "pre-cache" => {
            args.finish()?;
            run_pre_cache()
        }
        "release" => {
            let dry_run = args.contains("--dry-run");
            args.finish()?;
            ReleaseCmd { dry_run }.run()
        }
        "dist" => {
            let nightly = args.contains("--nightly");
            let client_version: Option<String> = args.opt_value_from_str("--client")?;
            args.finish()?;
            run_dist(nightly, client_version)
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
    dist"
            );
            Ok(())
        }
    }
}
