//! Runs `rustc --print -vV` to get the host target.
use anyhow::Context;
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{utf8_stdout, ManifestPath, Sysroot};

pub(super) enum TargetTipleConfig<'a> {
    #[expect(dead_code)]
    Rustc(&'a Sysroot),
    Cargo(&'a Sysroot, &'a ManifestPath),
}

pub(super) fn get(
    config: TargetTipleConfig<'_>,
    target: Option<&str>,
    extra_env: &FxHashMap<String, String>,
) -> anyhow::Result<Vec<String>> {
    if let Some(target) = target {
        return Ok(vec![target.to_owned()]);
    }

    let sysroot = match config {
        TargetTipleConfig::Cargo(sysroot, cargo_toml) => {
            match cargo_config_build_target(cargo_toml, extra_env, sysroot) {
                Ok(it) => return Ok(it),
                Err(e) => {
                    tracing::warn!("failed to run `cargo rustc --print cfg`, falling back to invoking rustc directly: {e}");
                    sysroot
                }
            }
        }
        TargetTipleConfig::Rustc(sysroot) => sysroot,
    };
    rustc_discover_host_triple(extra_env, sysroot).map(|it| vec![it])
}

fn rustc_discover_host_triple(
    extra_env: &FxHashMap<String, String>,
    sysroot: &Sysroot,
) -> anyhow::Result<String> {
    let mut rustc = sysroot.tool(Tool::Rustc);
    rustc.envs(extra_env);
    rustc.arg("-vV");
    tracing::debug!("Discovering host platform by {:?}", rustc);
    let stdout = utf8_stdout(rustc).context("Failed to discover host platform")?;
    let field = "host: ";
    let target = stdout.lines().find_map(|l| l.strip_prefix(field));
    if let Some(target) = target {
        Ok(target.to_owned())
    } else {
        // If we fail to resolve the host platform, it's not the end of the world.
        Err(anyhow::format_err!("rustc -vV did not report host platform, got:\n{}", stdout))
    }
}

fn cargo_config_build_target(
    cargo_toml: &ManifestPath,
    extra_env: &FxHashMap<String, String>,
    sysroot: &Sysroot,
) -> anyhow::Result<Vec<String>> {
    let mut cargo_config = sysroot.tool(Tool::Cargo);
    cargo_config.envs(extra_env);
    cargo_config
        .current_dir(cargo_toml.parent())
        .args(["-Z", "unstable-options", "config", "get", "build.target"])
        .env("RUSTC_BOOTSTRAP", "1");
    // if successful we receive `build.target = "target-triple"`
    // or `build.target = ["<target 1>", ..]`
    tracing::debug!("Discovering cargo config target by {:?}", cargo_config);
    utf8_stdout(cargo_config).and_then(parse_output_cargo_config_build_target)
}

fn parse_output_cargo_config_build_target(stdout: String) -> anyhow::Result<Vec<String>> {
    let trimmed = stdout.trim_start_matches("build.target = ").trim_matches('"');

    if !trimmed.starts_with('[') {
        return Ok([trimmed.to_owned()].to_vec());
    }

    serde_json::from_str(trimmed).context("Failed to parse `build.target` as an array of target")
}
