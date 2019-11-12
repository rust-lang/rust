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

use core::fmt::Write;
use core::str;
use pico_args::Arguments;
use std::{env, path::PathBuf};
use xtask::{
    codegen::{self, Mode},
    install_format_hook, run, run_clippy, run_fuzzer, run_rustfmt, run_with_output, Cmd, Result,
};

// Latest stable, feel free to send a PR if this lags behind.
const REQUIRED_RUST_VERSION: u32 = 38;

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
        Err(format!("Invalid flags: {}", invalid_flags).into())
    } else {
        Err(e.to_string().into())
    }
}

fn install(opts: InstallOpt) -> Result<()> {
    if cfg!(target_os = "macos") {
        fix_path_for_mac()?
    }
    if let Some(server) = opts.server {
        install_server(server)?;
    }
    if let Some(client) = opts.client {
        install_client(client)?;
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
            Err(e) => Err(format!("Failed getting HOME from environment with error: {}.", e))?,
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
            None => Err("Could not get PATH variable from env.")?,
        };

        let mut paths = env::split_paths(&vars).collect::<Vec<_>>();
        paths.append(&mut vscode_path);
        let new_paths = env::join_paths(paths)?;
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
        None => Err("Can't execute `code --version`. Perhaps it is not in $PATH?")?,
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
        Err("Could not install the Visual Studio Code extension. \
             Please make sure you have at least NodeJS 10.x installed and try again.")?;
    }

    Ok(())
}

fn install_server(opts: ServerOpt) -> Result<()> {
    let mut old_rust = false;
    if let Ok(output) = run_with_output("cargo --version", ".") {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            println!("{}", stdout);
            if !check_version(&stdout, REQUIRED_RUST_VERSION) {
                old_rust = true;
            }
        }
    }

    if old_rust {
        eprintln!(
            "\nWARNING: at least rust 1.{}.0 is required to compile rust-analyzer\n",
            REQUIRED_RUST_VERSION
        )
    }

    let res = if opts.jemalloc {
        run("cargo install --path crates/ra_lsp_server --locked --force --features jemalloc", ".")
    } else {
        run("cargo install --path crates/ra_lsp_server --locked --force", ".")
    };

    if res.is_err() && old_rust {
        eprintln!(
            "\nWARNING: at least rust 1.{}.0 is required to compile rust-analyzer\n",
            REQUIRED_RUST_VERSION
        )
    }

    res
}

fn check_version(version_output: &str, min_minor_version: u32) -> bool {
    // Parse second the number out of
    //      cargo 1.39.0-beta (1c6ec66d5 2019-09-30)
    let minor: Option<u32> = version_output.split('.').nth(1).and_then(|it| it.parse().ok());
    match minor {
        None => true,
        Some(minor) => minor >= min_minor_version,
    }
}
