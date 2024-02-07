//! Runs `rustc --print cfg` to get built-in cfg flags.

use std::process::Command;

use anyhow::Context;
use rustc_hash::FxHashMap;

use crate::{cfg_flag::CfgFlag, utf8_stdout, ManifestPath, Sysroot};

/// Determines how `rustc --print cfg` is discovered and invoked.
///
/// There options are supported:
/// - [`RustcCfgConfig::Cargo`], which relies on `cargo rustc --print cfg`
///   and `RUSTC_BOOTSTRAP`.
/// - [`RustcCfgConfig::Explicit`], which uses an explicit path to the `rustc`
///   binary in the sysroot.
/// - [`RustcCfgConfig::Discover`], which uses [`toolchain::rustc`].
pub(crate) enum RustcCfgConfig<'a> {
    Cargo(&'a ManifestPath),
    Explicit(&'a Sysroot),
    Discover,
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
            res.push(CfgFlag::KeyValue { key: key.to_string(), value: ty.into() });
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
    let mut cmd = match config {
        RustcCfgConfig::Cargo(cargo_toml) => {
            let mut cmd = Command::new(toolchain::cargo());
            cmd.envs(extra_env);
            cmd.current_dir(cargo_toml.parent())
                .args(["rustc", "-Z", "unstable-options", "--print", "cfg"])
                .env("RUSTC_BOOTSTRAP", "1");
            if let Some(target) = target {
                cmd.args(["--target", target]);
            }

            return utf8_stdout(cmd).context("Unable to run `cargo rustc`");
        }
        RustcCfgConfig::Explicit(sysroot) => {
            let rustc: std::path::PathBuf = sysroot.discover_rustc()?.into();
            tracing::debug!(?rustc, "using explicit rustc from sysroot");
            Command::new(rustc)
        }
        RustcCfgConfig::Discover => {
            let rustc = toolchain::rustc();
            tracing::debug!(?rustc, "using rustc from env");
            Command::new(rustc)
        }
    };

    cmd.envs(extra_env);
    cmd.args(["--print", "cfg", "-O"]);
    if let Some(target) = target {
        cmd.args(["--target", target]);
    }

    utf8_stdout(cmd).context("Unable to run `rustc`")
}
