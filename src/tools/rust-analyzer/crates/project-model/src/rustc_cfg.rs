//! Runs `rustc --print cfg` to get built-in cfg flags.

use std::process::Command;

use anyhow::Context;
use rustc_hash::FxHashMap;

use crate::{cfg_flag::CfgFlag, utf8_stdout, ManifestPath, Sysroot};

/// Determines how `rustc --print cfg` is discovered and invoked.
pub(crate) enum RustcCfgConfig<'a> {
    /// Use `rustc --print cfg`, either from with the binary from the sysroot or by discovering via
    /// [`toolchain::rustc`].
    Rustc(Option<&'a Sysroot>),
    /// Use `cargo --print cfg`, either from with the binary from the sysroot or by discovering via
    /// [`toolchain::cargo`].
    Cargo(Option<&'a Sysroot>, &'a ManifestPath),
}

pub(crate) fn get(
    target: Option<&str>,
    extra_env: &FxHashMap<String, String>,
    config: RustcCfgConfig<'_>,
) -> Vec<CfgFlag> {
    let _p = tracing::span!(tracing::Level::INFO, "rustc_cfg::get").entered();
    let mut res = Vec::with_capacity(6 * 2 + 1);

    // Some nightly-only cfgs, which are required for stdlib
    res.push(CfgFlag::Atom("target_thread_local".into()));
    for ty in ["8", "16", "32", "64", "cas", "ptr"] {
        for key in ["target_has_atomic", "target_has_atomic_load_store"] {
            res.push(CfgFlag::KeyValue { key: key.to_owned(), value: ty.into() });
        }
    }

    // Add miri cfg, which is useful for mir eval in stdlib
    res.push(CfgFlag::Atom("miri".into()));

    let rustc_cfgs = get_rust_cfgs(target, extra_env, config);

    let rustc_cfgs = match rustc_cfgs {
        Ok(cfgs) => cfgs,
        Err(e) => {
            tracing::error!(?e, "failed to get rustc cfgs");
            return res;
        }
    };

    let rustc_cfgs =
        rustc_cfgs.lines().map(|it| it.parse::<CfgFlag>()).collect::<Result<Vec<_>, _>>();

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
            let mut cmd = Command::new(toolchain::Tool::Cargo.path());
            Sysroot::set_rustup_toolchain_env(&mut cmd, sysroot);
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

    let mut cmd = Command::new(toolchain::Tool::Rustc.path());
    Sysroot::set_rustup_toolchain_env(&mut cmd, sysroot);
    cmd.envs(extra_env);
    cmd.args(["--print", "cfg", "-O"]);
    if let Some(target) = target {
        cmd.args(["--target", target]);
    }

    utf8_stdout(cmd).context("unable to fetch cfgs via `rustc --print cfg -O`")
}
