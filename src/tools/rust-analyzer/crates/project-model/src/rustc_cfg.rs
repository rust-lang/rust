//! Runs `rustc --print cfg` to get built-in cfg flags.

use std::process::Command;

use anyhow::Result;

use crate::{cfg_flag::CfgFlag, utf8_stdout, CargoConfig, ManifestPath};

pub(crate) fn get(
    cargo_toml: Option<&ManifestPath>,
    target: Option<&str>,
    config: &CargoConfig,
) -> Vec<CfgFlag> {
    let _p = profile::span("rustc_cfg::get");
    let mut res = Vec::with_capacity(6 * 2 + 1);

    // Some nightly-only cfgs, which are required for stdlib
    res.push(CfgFlag::Atom("target_thread_local".into()));
    for ty in ["8", "16", "32", "64", "cas", "ptr"] {
        for key in ["target_has_atomic", "target_has_atomic_load_store"] {
            res.push(CfgFlag::KeyValue { key: key.to_string(), value: ty.into() });
        }
    }

    match get_rust_cfgs(cargo_toml, target, config) {
        Ok(rustc_cfgs) => {
            tracing::debug!(
                "rustc cfgs found: {:?}",
                rustc_cfgs
                    .lines()
                    .map(|it| it.parse::<CfgFlag>().map(|it| it.to_string()))
                    .collect::<Vec<_>>()
            );
            res.extend(rustc_cfgs.lines().filter_map(|it| it.parse().ok()));
        }
        Err(e) => tracing::error!("failed to get rustc cfgs: {e:?}"),
    }

    res
}

fn get_rust_cfgs(
    cargo_toml: Option<&ManifestPath>,
    target: Option<&str>,
    config: &CargoConfig,
) -> Result<String> {
    if let Some(cargo_toml) = cargo_toml {
        let mut cargo_config = Command::new(toolchain::cargo());
        cargo_config.envs(&config.extra_env);
        cargo_config
            .current_dir(cargo_toml.parent())
            .args(&["-Z", "unstable-options", "rustc", "--print", "cfg"])
            .env("RUSTC_BOOTSTRAP", "1");
        if let Some(target) = target {
            cargo_config.args(&["--target", target]);
        }
        match utf8_stdout(cargo_config) {
            Ok(it) => return Ok(it),
            Err(e) => tracing::debug!("{e:?}: falling back to querying rustc for cfgs"),
        }
    }
    // using unstable cargo features failed, fall back to using plain rustc
    let mut cmd = Command::new(toolchain::rustc());
    cmd.envs(&config.extra_env);
    cmd.args(&["--print", "cfg", "-O"]);
    if let Some(target) = target {
        cmd.args(&["--target", target]);
    }
    utf8_stdout(cmd)
}
