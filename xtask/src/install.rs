//! Installs rust-analyzer language server and/or editor plugin.

use std::{env, path::PathBuf, str};

use anyhow::{bail, format_err, Context, Result};

use crate::not_bash::{pushd, run};

// Latest stable, feel free to send a PR if this lags behind.
const REQUIRED_RUST_VERSION: u32 = 43;

pub struct InstallCmd {
    pub client: Option<ClientOpt>,
    pub server: Option<ServerOpt>,
}

pub enum ClientOpt {
    VsCode,
}

pub struct ServerOpt {
    pub malloc: Malloc,
}

pub enum Malloc {
    System,
    Mimalloc,
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
            Err(e) => bail!("Failed getting HOME from environment with error: {}.", e),
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
            None => bail!("Could not get PATH variable from env."),
        };

        let mut paths = env::split_paths(&vars).collect::<Vec<_>>();
        paths.append(&mut vscode_path);
        let new_paths = env::join_paths(paths).context("build env PATH")?;
        env::set_var("PATH", &new_paths);
    }

    Ok(())
}

fn install_client(ClientOpt::VsCode: ClientOpt) -> Result<()> {
    let _dir = pushd("./editors/code");

    let find_code = |f: fn(&str) -> bool| -> Result<&'static str> {
        ["code", "code-insiders", "codium", "code-oss"]
            .iter()
            .copied()
            .find(|bin| f(bin))
            .ok_or_else(|| {
                format_err!("Can't execute `code --version`. Perhaps it is not in $PATH?")
            })
    };

    let installed_extensions = if cfg!(unix) {
        run!("npm --version").context("`npm` is required to build the VS Code plugin")?;
        run!("npm install")?;

        run!("npm run package --scripts-prepend-node-path")?;

        let code = find_code(|bin| run!("{} --version", bin).is_ok())?;
        run!("{} --install-extension rust-analyzer.vsix --force", code)?;
        run!("{} --list-extensions", code; echo = false)?
    } else {
        run!("cmd.exe /c npm --version")
            .context("`npm` is required to build the VS Code plugin")?;
        run!("cmd.exe /c npm install")?;

        run!("cmd.exe /c npm run package")?;

        let code = find_code(|bin| run!("cmd.exe /c {}.cmd --version", bin).is_ok())?;
        run!(r"cmd.exe /c {}.cmd --install-extension rust-analyzer.vsix --force", code)?;
        run!("cmd.exe /c {}.cmd --list-extensions", code; echo = false)?
    };

    if !installed_extensions.contains("rust-analyzer") {
        bail!(
            "Could not install the Visual Studio Code extension. \
            Please make sure you have at least NodeJS 12.x together with the latest version of VS Code installed and try again. \
            Note that installing via xtask install does not work for VS Code Remote, instead you’ll need to install the .vsix manually."
        );
    }

    Ok(())
}

fn install_server(opts: ServerOpt) -> Result<()> {
    let mut old_rust = false;
    if let Ok(stdout) = run!("cargo --version") {
        if !check_version(&stdout, REQUIRED_RUST_VERSION) {
            old_rust = true;
        }
    }

    if old_rust {
        eprintln!(
            "\nWARNING: at least rust 1.{}.0 is required to compile rust-analyzer\n",
            REQUIRED_RUST_VERSION,
        )
    }

    let malloc_feature = match opts.malloc {
        Malloc::System => "",
        Malloc::Mimalloc => "--features mimalloc",
    };
    let res = run!("cargo install --path crates/rust-analyzer --locked --force {}", malloc_feature);

    if res.is_err() && old_rust {
        eprintln!(
            "\nWARNING: at least rust 1.{}.0 is required to compile rust-analyzer\n",
            REQUIRED_RUST_VERSION,
        );
    }

    res.map(drop)
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
