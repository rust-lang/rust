use clap::{App, Arg, SubCommand};
use core::str;
use ra_tools::{
    gen_tests, generate, install_format_hook, run, run_clippy, run_fuzzer, run_rustfmt, Cmd,
    Overwrite, Result,
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
    let matches = App::new("tasks")
        .setting(clap::AppSettings::SubcommandRequiredElseHelp)
        .subcommand(SubCommand::with_name("gen-syntax"))
        .subcommand(SubCommand::with_name("gen-tests"))
        .subcommand(
            SubCommand::with_name("install-ra")
                .arg(Arg::with_name("server").long("--server"))
                .arg(Arg::with_name("jemalloc").long("jemalloc"))
                .arg(Arg::with_name("client-code").long("client-code").conflicts_with("server")),
        )
        .alias("install-code")
        .subcommand(SubCommand::with_name("format"))
        .subcommand(SubCommand::with_name("format-hook"))
        .subcommand(SubCommand::with_name("fuzz-tests"))
        .subcommand(SubCommand::with_name("lint"))
        .get_matches();
    match matches.subcommand() {
        ("install-ra", Some(matches)) => {
            let opts = InstallOpt {
                client: if matches.is_present("server") { None } else { Some(ClientOpt::VsCode) },
                server: if matches.is_present("client-code") {
                    None
                } else {
                    Some(ServerOpt { jemalloc: matches.is_present("jemalloc") })
                },
            };
            install(opts)?
        }
        ("gen-tests", _) => gen_tests(Overwrite)?,
        ("gen-syntax", _) => generate(Overwrite)?,
        ("format", _) => run_rustfmt(Overwrite)?,
        ("format-hook", _) => install_format_hook()?,
        ("lint", _) => run_clippy()?,
        ("fuzz-tests", _) => run_fuzzer()?,
        _ => unreachable!(),
    }
    Ok(())
}

fn install(opts: InstallOpt) -> Result<()> {
    if cfg!(target_os = "macos") {
        fix_path_for_mac()?
    }
    if let Some(client) = opts.client {
        install_client(client)?;
    }
    if let Some(server) = opts.server {
        install_server(server)?;
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
        run("cargo install --path crates/ra_lsp_server --force --features jemalloc", ".")
    } else {
        run("cargo install --path crates/ra_lsp_server --force", ".")
    }
}
