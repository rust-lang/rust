//! See https://github.com/matklad/cargo-xtask/.
//!
//! This binary defines various auxiliary build commands, which are not
//! expressible with just `cargo`. Notably, it provides `cargo xtask codegen`
//! for code generation and `cargo xtask install` for installation of
//! rust-analyzer server and client.
//!
//! This binary is integrated into the `cargo` command line by using an alias in
//! `.cargo/config`.
mod help;

use anyhow::Context;
use autocfg;
use core::fmt::Write;
use core::str;
use pico_args::Arguments;
use std::{env, path::PathBuf};
use xtask::{
    codegen::{self, Mode},
    install_format_hook, run, run_clippy, run_fuzzer, run_rustfmt, Cmd, Result,
};

// Latest stable, feel free to send a PR if this lags behind.
const REQUIRED_RUST_VERSION: (usize, usize) = (1, 39);

struct InstallOpt {
    client: Option<ClientOpt>,
    server: Option<ServerOpt>,
}

enum ClientOpt {
    VsCode,
}

struct ServerOpt {
    jemalloc: bool,
}

fn main() -> Result<()> {
    let subcommand = match std::env::args_os().nth(1) {
        None => {
            eprintln!("{}", help::GLOBAL_HELP);
            return Ok(());
        }
        Some(s) => s,
    };
    let mut matches = Arguments::from_vec(std::env::args_os().skip(2).collect());
    let subcommand = &*subcommand.to_string_lossy();
    match subcommand {
        "install" => {
            if matches.contains(["-h", "--help"]) {
                eprintln!("{}", help::INSTALL_HELP);
                return Ok(());
            }
            let server = matches.contains("--server");
            let client_code = matches.contains("--client-code");
            if server && client_code {
                eprintln!("{}", help::INSTALL_RA_CONFLICT);
                return Ok(());
            }
            let jemalloc = matches.contains("--jemalloc");
            matches.finish().or_else(handle_extra_flags)?;
            let opts = InstallOpt {
                client: if server { None } else { Some(ClientOpt::VsCode) },
                server: if client_code { None } else { Some(ServerOpt { jemalloc }) },
            };
            install(opts)?
        }
        "codegen" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            codegen::generate_syntax(Mode::Overwrite)?;
            codegen::generate_parser_tests(Mode::Overwrite)?;
            codegen::generate_assists_docs(Mode::Overwrite)?;
        }
        "format" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            run_rustfmt(Mode::Overwrite)?
        }
        "format-hook" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            install_format_hook()?
        }
        "lint" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            run_clippy()?
        }
        "fuzz-tests" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            run_fuzzer()?
        }
        _ => eprintln!("{}", help::GLOBAL_HELP),
    }
    Ok(())
}

fn handle_extra_flags(e: pico_args::Error) -> Result<()> {
    if let pico_args::Error::UnusedArgsLeft(flags) = e {
        let mut invalid_flags = String::new();
        for flag in flags {
            write!(&mut invalid_flags, "{}, ", flag)?;
        }
        let (invalid_flags, _) = invalid_flags.split_at(invalid_flags.len() - 2);
        anyhow::bail!("Invalid flags: {}", invalid_flags)
    } else {
        anyhow::bail!(e.to_string())
    }
}

fn install(opts: InstallOpt) -> Result<()> {
    if cfg!(target_os = "macos") {
        fix_path_for_mac().context("Fix path for mac")?
    }
    if let Some(server) = opts.server {
        install_server(server).context("install server")?;
    }
    if let Some(client) = opts.client {
        install_client(client).context("install client")?;
    }
    Ok(())
}

fn fix_path_for_mac() -> Result<()> {
    let mut vscode_path: Vec<PathBuf> = {
        const COMMON_APP_PATH: &str =
            r"/Applications/Visual Studio Code.app/Contents/Resources/app/bin";
        const ROOT_DIR: &str = "";
        let home_dir = match env::var("HOME") {
            Ok(home) => home,
            Err(e) => anyhow::bail!("Failed getting HOME from environment with error: {}.", e),
        };

        [ROOT_DIR, &home_dir]
            .iter()
            .map(|dir| String::from(*dir) + COMMON_APP_PATH)
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .collect()
    };

    if !vscode_path.is_empty() {
        let vars = match env::var_os("PATH") {
            Some(path) => path,
            None => anyhow::bail!("Could not get PATH variable from env."),
        };

        let mut paths = env::split_paths(&vars).collect::<Vec<_>>();
        paths.append(&mut vscode_path);
        let new_paths = env::join_paths(paths).context("build env PATH")?;
        env::set_var("PATH", &new_paths);
    }

    Ok(())
}

fn install_client(ClientOpt::VsCode: ClientOpt) -> Result<()> {
    let npm_version = Cmd {
        unix: r"npm --version",
        windows: r"cmd.exe /c npm.cmd --version",
        work_dir: "./editors/code",
    }
    .run();

    if npm_version.is_err() {
        eprintln!("\nERROR: `npm --version` failed, `npm` is required to build the VS Code plugin")
    }

    Cmd { unix: r"npm ci", windows: r"cmd.exe /c npm.cmd ci", work_dir: "./editors/code" }.run()?;
    Cmd {
        unix: r"npm run package --scripts-prepend-node-path",
        windows: r"cmd.exe /c npm.cmd run package",
        work_dir: "./editors/code",
    }
    .run()?;

    let code_binary = ["code", "code-insiders", "codium"].iter().find(|bin| {
        Cmd {
            unix: &format!("{} --version", bin),
            windows: &format!("cmd.exe /c {}.cmd --version", bin),
            work_dir: "./editors/code",
        }
        .run()
        .is_ok()
    });

    let code_binary = match code_binary {
        Some(it) => it,
        None => anyhow::bail!("Can't execute `code --version`. Perhaps it is not in $PATH?"),
    };

    Cmd {
        unix: &format!(r"{} --install-extension ./ra-lsp-0.0.1.vsix --force", code_binary),
        windows: &format!(
            r"cmd.exe /c {}.cmd --install-extension ./ra-lsp-0.0.1.vsix --force",
            code_binary
        ),
        work_dir: "./editors/code",
    }
    .run()?;

    let output = Cmd {
        unix: &format!(r"{} --list-extensions", code_binary),
        windows: &format!(r"cmd.exe /c {}.cmd --list-extensions", code_binary),
        work_dir: ".",
    }
    .run_with_output()?;

    if !str::from_utf8(&output.stdout)?.contains("ra-lsp") {
        anyhow::bail!(
            "Could not install the Visual Studio Code extension. \
             Please make sure you have at least NodeJS 10.x installed and try again."
        );
    }

    Ok(())
}

fn install_server(opts: ServerOpt) -> Result<()> {
    let target_dir = env::var_os("CARGO_TARGET_DIR").unwrap_or_else(|| "target".into());
    let ac = autocfg::AutoCfg::with_dir(target_dir)?;

    let old_rust = !ac.probe_rustc_version(REQUIRED_RUST_VERSION.0, REQUIRED_RUST_VERSION.1);

    if old_rust {
        eprintln!(
            "\nWARNING: at least rust {}.{}.0 is required to compile rust-analyzer\n",
            REQUIRED_RUST_VERSION.0, REQUIRED_RUST_VERSION.1
        )
    }

    let res = if opts.jemalloc {
        run("cargo install --path crates/ra_lsp_server --locked --force --features jemalloc", ".")
    } else {
        run("cargo install --path crates/ra_lsp_server --locked --force", ".")
    };

    if res.is_err() && old_rust {
        eprintln!(
            "\nWARNING: at least rust {}.{}.0 is required to compile rust-analyzer\n",
            REQUIRED_RUST_VERSION.0, REQUIRED_RUST_VERSION.1
        )
    }

    res
}
