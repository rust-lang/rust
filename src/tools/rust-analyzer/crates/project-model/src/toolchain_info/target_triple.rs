use anyhow::Context;
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{toolchain_info::QueryConfig, utf8_stdout, ManifestPath, Sysroot};

/// For cargo, runs `cargo -Zunstable-options config get build.target` to get the configured project target(s).
/// For rustc, runs `rustc --print -vV` to get the host target.
pub fn get(
    config: QueryConfig<'_>,
    target: Option<&str>,
    extra_env: &FxHashMap<String, String>,
) -> anyhow::Result<Vec<String>> {
    let _p = tracing::info_span!("target_triple::get").entered();
    if let Some(target) = target {
        return Ok(vec![target.to_owned()]);
    }

    let sysroot = match config {
        QueryConfig::Cargo(sysroot, cargo_toml) => {
            match cargo_config_build_target(cargo_toml, extra_env, sysroot) {
                Some(it) => return Ok(it),
                None => sysroot,
            }
        }
        QueryConfig::Rustc(sysroot) => sysroot,
    };
    rustc_discover_host_triple(extra_env, sysroot).map(|it| vec![it])
}

fn rustc_discover_host_triple(
    extra_env: &FxHashMap<String, String>,
    sysroot: &Sysroot,
) -> anyhow::Result<String> {
    let mut cmd = sysroot.tool(Tool::Rustc, &std::env::current_dir()?);
    cmd.envs(extra_env);
    cmd.arg("-vV");
    let stdout = utf8_stdout(&mut cmd)
        .with_context(|| format!("unable to discover host platform via `{cmd:?}`"))?;
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
) -> Option<Vec<String>> {
    let mut cmd = sysroot.tool(Tool::Cargo, cargo_toml.parent());
    cmd.envs(extra_env);
    cmd.current_dir(cargo_toml.parent()).env("RUSTC_BOOTSTRAP", "1");
    cmd.args(["-Z", "unstable-options", "config", "get", "build.target"]);
    // if successful we receive `build.target = "target-triple"`
    // or `build.target = ["<target 1>", ..]`
    // this might be `error: config value `build.target` is not set` in which case we
    // don't wanna log the error
    utf8_stdout(&mut cmd).and_then(parse_output_cargo_config_build_target).ok()
}

// Parses `"build.target = [target-triple, target-triple, ...]"` or `"build.target = "target-triple"`
fn parse_output_cargo_config_build_target(stdout: String) -> anyhow::Result<Vec<String>> {
    let trimmed = stdout.trim_start_matches("build.target = ").trim_matches('"');

    if !trimmed.starts_with('[') {
        return Ok([trimmed.to_owned()].to_vec());
    }

    serde_json::from_str(trimmed).context("Failed to parse `build.target` as an array of target")
}
