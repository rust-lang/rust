//! Installs rust-analyzer language server and/or editor plugin.

use std::{env, path::PathBuf, str};

use anyhow::{Context, Result};

use crate::cmd::{run, run_with_output, Cmd};

// Latest stable, feel free to send a PR if this lags behind.
const REQUIRED_RUST_VERSION: u32 = 40;

pub struct InstallCmd {
    pub client: Option<ClientOpt>,
    pub server: Option<ServerOpt>,
}

pub enum ClientOpt {
    VsCode,
}

pub struct ServerOpt {
    pub jemalloc: bool,
}

impl InstallCmd {
    pub fn run(self) -> Result<()> {
        if cfg!(target_os = "macos") {
            fix_path_for_mac().context("Fix path for mac")?
        }
        if let Some(server) = self.server {
            install_server(server).context("install server")?;
        }
        if let Some(client) = self.client {
            install_client(client).context("install client")?;
        }
        Ok(())
    }
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
        windows: r"cmd.exe /c npm --version",
        work_dir: "./editors/code",
    }
    .run();

    if npm_version.is_err() {
        eprintln!("\nERROR: `npm --version` failed, `npm` is required to build the VS Code plugin")
    }

    Cmd { unix: r"npm install", windows: r"cmd.exe /c npm install", work_dir: "./editors/code" }
        .run()?;
    Cmd {
        unix: r"npm run package --scripts-prepend-node-path",
        windows: r"cmd.exe /c npm run package",
        work_dir: "./editors/code",
    }
    .run()?;

    let code_binary = ["code", "code-insiders", "codium", "code-oss"].iter().find(|bin| {
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
             Please make sure you have at least NodeJS 10.x together with the latest version of VS Code installed and try again."
        );
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
            REQUIRED_RUST_VERSION,
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
            REQUIRED_RUST_VERSION,
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
