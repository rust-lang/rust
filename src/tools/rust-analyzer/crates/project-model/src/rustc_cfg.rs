//! Runs `rustc --print cfg` to get built-in cfg flags.

use anyhow::Context;
use cfg::CfgAtom;
use intern::Symbol;
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{utf8_stdout, ManifestPath, Sysroot};

/// Determines how `rustc --print cfg` is discovered and invoked.
pub(crate) enum RustcCfgConfig<'a> {
    /// Use `rustc --print cfg`, either from with the binary from the sysroot or by discovering via
    /// [`toolchain::rustc`].
    Rustc(&'a Sysroot),
    /// Use `cargo --print cfg`, either from with the binary from the sysroot or by discovering via
    /// [`toolchain::cargo`].
    Cargo(&'a Sysroot, &'a ManifestPath),
}

pub(crate) fn get(
    target: Option<&str>,
    extra_env: &FxHashMap<String, String>,
    config: RustcCfgConfig<'_>,
) -> Vec<CfgAtom> {
    let _p = tracing::info_span!("rustc_cfg::get").entered();
    let mut res: Vec<_> = Vec::with_capacity(7 * 2 + 1);

    // Some nightly-only cfgs, which are required for stdlib
    res.push(CfgAtom::Flag(Symbol::intern("target_thread_local")));
    for key in ["target_has_atomic", "target_has_atomic_load_store"] {
        for ty in ["8", "16", "32", "64", "cas", "ptr"] {
            res.push(CfgAtom::KeyValue { key: Symbol::intern(key), value: Symbol::intern(ty) });
        }
        res.push(CfgAtom::Flag(Symbol::intern(key)));
    }

    let rustc_cfgs = get_rust_cfgs(target, extra_env, config);

    let rustc_cfgs = match rustc_cfgs {
        Ok(cfgs) => cfgs,
        Err(e) => {
            tracing::error!(?e, "failed to get rustc cfgs");
            return res;
        }
    };

    let rustc_cfgs = rustc_cfgs.lines().map(crate::parse_cfg).collect::<Result<Vec<_>, _>>();

    match rustc_cfgs {
        Ok(rustc_cfgs) => {
            tracing::debug!(?rustc_cfgs, "rustc cfgs found");
            res.extend(rustc_cfgs);
        }
        Err(e) => {
            tracing::error!(?e, "failed to get rustc cfgs")
        }
    }

    res
}

fn get_rust_cfgs(
    target: Option<&str>,
    extra_env: &FxHashMap<String, String>,
    config: RustcCfgConfig<'_>,
) -> anyhow::Result<String> {
    let sysroot = match config {
        RustcCfgConfig::Cargo(sysroot, cargo_toml) => {
            let mut cmd = sysroot.tool(Tool::Cargo);

            cmd.envs(extra_env);
            cmd.current_dir(cargo_toml.parent())
                .args(["rustc", "-Z", "unstable-options", "--print", "cfg"])
                .env("RUSTC_BOOTSTRAP", "1");
            if let Some(target) = target {
                cmd.args(["--target", target]);
            }

            match utf8_stdout(cmd) {
                Ok(it) => return Ok(it),
                Err(e) => {
                    tracing::warn!("failed to run `cargo rustc --print cfg`, falling back to invoking rustc directly: {e}");
                    sysroot
                }
            }
        }
        RustcCfgConfig::Rustc(sysroot) => sysroot,
    };

    let mut cmd = sysroot.tool(Tool::Rustc);
    cmd.envs(extra_env);
    cmd.args(["--print", "cfg", "-O"]);
    if let Some(target) = target {
        cmd.args(["--target", target]);
    }

    utf8_stdout(cmd).context("unable to fetch cfgs via `rustc --print cfg -O`")
}
