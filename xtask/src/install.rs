//! Installs rust-analyzer language server and/or editor plugin.

use std::{env, path::PathBuf, str};

use anyhow::{bail, format_err, Context, Result};
use xshell::{cmd, pushd};

use crate::flags;

// Latest stable, feel free to send a PR if this lags behind.
const REQUIRED_RUST_VERSION: u32 = 50;

impl flags::Install {
    pub(crate) fn run(self) -> Result<()> {
        if cfg!(target_os = "macos") {
            fix_path_for_mac().context("Fix path for mac")?
        }
        if let Some(server) = self.server() {
            install_server(server).context("install server")?;
        }
        if let Some(client) = self.client() {
            install_client(client).context("install client")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ClientOpt {
    VsCode,
    VsCodeExploration,
    VsCodeInsiders,
    VsCodium,
    VsCodeOss,
    Any,
}

impl ClientOpt {
    pub(crate) const fn as_cmds(&self) -> &'static [&'static str] {
        match self {
            ClientOpt::VsCode => &["code"],
            ClientOpt::VsCodeExploration => &["code-exploration"],
            ClientOpt::VsCodeInsiders => &["code-insiders"],
            ClientOpt::VsCodium => &["codium"],
            ClientOpt::VsCodeOss => &["code-oss"],
            ClientOpt::Any => &["code", "code-exploration", "code-insiders", "codium", "code-oss"],
        }
    }
}

impl Default for ClientOpt {
    fn default() -> Self {
        ClientOpt::Any
    }
}

impl std::str::FromStr for ClientOpt {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        [
            ClientOpt::VsCode,
            ClientOpt::VsCodeExploration,
            ClientOpt::VsCodeInsiders,
            ClientOpt::VsCodium,
            ClientOpt::VsCodeOss,
        ]
        .iter()
        .copied()
        .find(|c| [s] == c.as_cmds())
        .ok_or_else(|| anyhow::format_err!("no such client"))
    }
}

pub(crate) struct ServerOpt {
    pub(crate) malloc: Malloc,
}

pub(crate) enum Malloc {
    System,
    Mimalloc,
    Jemalloc,
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

fn install_client(client_opt: ClientOpt) -> Result<()> {
    let _dir = pushd("./editors/code");

    let find_code = |f: fn(&str) -> bool| -> Result<&'static str> {
        client_opt.as_cmds().iter().copied().find(|bin| f(bin)).ok_or_else(|| {
            format_err!("Can't execute `code --version`. Perhaps it is not in $PATH?")
        })
    };

    let installed_extensions = if cfg!(unix) {
        cmd!("npm --version").run().context("`npm` is required to build the VS Code plugin")?;
        cmd!("npm ci").run()?;

        cmd!("npm run package --scripts-prepend-node-path").run()?;

        let code = find_code(|bin| cmd!("{bin} --version").read().is_ok())?;
        cmd!("{code} --install-extension rust-analyzer.vsix --force").run()?;
        cmd!("{code} --list-extensions").read()?
    } else {
        cmd!("cmd.exe /c npm --version")
            .run()
            .context("`npm` is required to build the VS Code plugin")?;
        cmd!("cmd.exe /c npm ci").run()?;

        cmd!("cmd.exe /c npm run package").run()?;

        let code = find_code(|bin| cmd!("cmd.exe /c {bin}.cmd --version").read().is_ok())?;
        cmd!("cmd.exe /c {code}.cmd --install-extension rust-analyzer.vsix --force").run()?;
        cmd!("cmd.exe /c {code}.cmd --list-extensions").read()?
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
    if let Ok(stdout) = cmd!("cargo --version").read() {
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
    let features = match opts.malloc {
        Malloc::System => &[][..],
        Malloc::Mimalloc => &["--features", "mimalloc"],
        Malloc::Jemalloc => &["--features", "jemalloc"],
    };

    let cmd = cmd!("cargo install --path crates/rust-analyzer --locked --force --features force-always-assert {features...}");
    let res = cmd.run();

    if res.is_err() && old_rust {
        eprintln!(
            "\nWARNING: at least rust 1.{}.0 is required to compile rust-analyzer\n",
            REQUIRED_RUST_VERSION,
        );
    }

    res?;
    Ok(())
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
