//! Installs rust-analyzer language server and/or editor plugin.

use std::{env, path::PathBuf, str};

use anyhow::{bail, format_err, Context};
use xshell::{cmd, Shell};

use crate::flags;

impl flags::Install {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        if cfg!(target_os = "macos") {
            fix_path_for_mac(sh).context("Fix path for mac")?;
        }
        if let Some(server) = self.server() {
            install_server(sh, server).context("install server")?;
        }
        if let Some(client) = self.client() {
            install_client(sh, client).context("install client")?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct ClientOpt {
    pub(crate) code_bin: Option<String>,
}

const VS_CODES: &[&str] = &["code", "code-exploration", "code-insiders", "codium", "code-oss"];

pub(crate) struct ServerOpt {
    pub(crate) malloc: Malloc,
}

pub(crate) enum Malloc {
    System,
    Mimalloc,
    Jemalloc,
}

fn fix_path_for_mac(sh: &Shell) -> anyhow::Result<()> {
    let mut vscode_path: Vec<PathBuf> = {
        const COMMON_APP_PATH: &str =
            r"/Applications/Visual Studio Code.app/Contents/Resources/app/bin";
        const ROOT_DIR: &str = "";
        let home_dir = sh.var("HOME").map_err(|err| {
            format_err!("Failed getting HOME from environment with error: {}.", err)
        })?;

        [ROOT_DIR, &home_dir]
            .into_iter()
            .map(|dir| dir.to_string() + COMMON_APP_PATH)
            .map(PathBuf::from)
            .filter(|path| path.exists())
            .collect()
    };

    if !vscode_path.is_empty() {
        let vars = sh.var_os("PATH").context("Could not get PATH variable from env.")?;

        let mut paths = env::split_paths(&vars).collect::<Vec<_>>();
        paths.append(&mut vscode_path);
        let new_paths = env::join_paths(paths).context("build env PATH")?;
        sh.set_var("PATH", new_paths);
    }

    Ok(())
}

fn install_client(sh: &Shell, client_opt: ClientOpt) -> anyhow::Result<()> {
    let _dir = sh.push_dir("./editors/code");

    // Package extension.
    if cfg!(unix) {
        cmd!(sh, "npm --version").run().context("`npm` is required to build the VS Code plugin")?;
        cmd!(sh, "npm ci").run()?;

        cmd!(sh, "npm run package --scripts-prepend-node-path").run()?;
    } else {
        cmd!(sh, "cmd.exe /c npm --version")
            .run()
            .context("`npm` is required to build the VS Code plugin")?;
        cmd!(sh, "cmd.exe /c npm ci").run()?;

        cmd!(sh, "cmd.exe /c npm run package").run()?;
    };

    // Find the appropriate VS Code binary.
    let lifetime_extender;
    let candidates: &[&str] = match client_opt.code_bin.as_deref() {
        Some(it) => {
            lifetime_extender = [it];
            &lifetime_extender[..]
        }
        None => VS_CODES,
    };
    let code = candidates
        .iter()
        .copied()
        .find(|&bin| {
            if cfg!(unix) {
                cmd!(sh, "{bin} --version").read().is_ok()
            } else {
                cmd!(sh, "cmd.exe /c {bin}.cmd --version").read().is_ok()
            }
        })
        .ok_or_else(|| {
            format_err!("Can't execute `{} --version`. Perhaps it is not in $PATH?", candidates[0])
        })?;

    // Install & verify.
    let installed_extensions = if cfg!(unix) {
        cmd!(sh, "{code} --install-extension rust-analyzer.vsix --force").run()?;
        cmd!(sh, "{code} --list-extensions").read()?
    } else {
        cmd!(sh, "cmd.exe /c {code}.cmd --install-extension rust-analyzer.vsix --force").run()?;
        cmd!(sh, "cmd.exe /c {code}.cmd --list-extensions").read()?
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

fn install_server(sh: &Shell, opts: ServerOpt) -> anyhow::Result<()> {
    let features = match opts.malloc {
        Malloc::System => &[][..],
        Malloc::Mimalloc => &["--features", "mimalloc"],
        Malloc::Jemalloc => &["--features", "jemalloc"],
    };

    let cmd = cmd!(sh, "cargo install --path crates/rust-analyzer --locked --force --features force-always-assert {features...}");
    cmd.run()?;
    Ok(())
}
