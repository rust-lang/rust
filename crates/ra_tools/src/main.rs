mod help;

use core::fmt::Write;
use core::str;
use pico_args::Arguments;
use ra_tools::{
    gen_tests, generate_boilerplate, install_format_hook, run, run_clippy, run_fuzzer, run_rustfmt,
    Cmd, Overwrite, Result,
};
use std::{env, path::PathBuf};

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
        "install-ra" | "install-code" => {
            if matches.contains(["-h", "--help"]) {
                eprintln!("{}", help::INSTALL_RA_HELP);
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
                server: if client_code { None } else { Some(ServerOpt { jemalloc: jemalloc }) },
            };
            install(opts)?
        }
        "gen-tests" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            gen_tests(Overwrite)?
        }
        "gen-syntax" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            generate_boilerplate(Overwrite)?
        }
        "format" => {
            if matches.contains(["-h", "--help"]) {
                help::print_no_param_subcommand_help(&subcommand);
                return Ok(());
            }
            run_rustfmt(Overwrite)?
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
    Cmd { unix: r"npm ci", windows: r"cmd.exe /c npm.cmd ci", work_dir: "./editors/code" }.run()?;
    Cmd {
        unix: r"npm run package",
        windows: r"cmd.exe /c npm.cmd run package",
        work_dir: "./editors/code",
    }
    .run()?;

    let code_in_path = Cmd {
        unix: r"code --version",
        windows: r"cmd.exe /c code.cmd --version",
        work_dir: "./editors/code",
    }
    .run()
    .is_ok();
    if !code_in_path {
        Err("Can't execute `code --version`. Perhaps it is not in $PATH?")?;
    }

    Cmd {
        unix: r"code --install-extension ./ra-lsp-0.0.1.vsix --force",
        windows: r"cmd.exe /c code.cmd --install-extension ./ra-lsp-0.0.1.vsix --force",
        work_dir: "./editors/code",
    }
    .run()?;

    let output = Cmd {
        unix: r"code --list-extensions",
        windows: r"cmd.exe /c code.cmd --list-extensions",
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
    if opts.jemalloc {
        run("cargo install --path crates/ra_lsp_server --locked --force --features jemalloc", ".")
    } else {
        run("cargo install --path crates/ra_lsp_server --locked --force", ".")
    }
}
