//! Functionality to discover the current build target(s).
use std::path::Path;

use anyhow::Context;
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{
    Sysroot, cargo_config_file::CargoConfigFile, toolchain_info::QueryConfig, utf8_stdout,
};

/// For cargo, runs `cargo -Zunstable-options config get build.target` to get the configured project target(s).
/// For rustc, runs `rustc --print -vV` to get the host target.
pub fn get(
    config: QueryConfig<'_>,
    target: Option<&str>,
    extra_env: &FxHashMap<String, Option<String>>,
) -> anyhow::Result<Vec<String>> {
    let _p = tracing::info_span!("target_tuple::get").entered();
    if let Some(target) = target {
        return Ok(vec![target.to_owned()]);
    }

    let (sysroot, current_dir) = match config {
        QueryConfig::Cargo(sysroot, cargo_toml, config_file) => {
            match config_file.as_ref().and_then(cargo_config_build_target) {
                Some(it) => return Ok(it),
                None => (sysroot, cargo_toml.parent().as_ref()),
            }
        }
        QueryConfig::Rustc(sysroot, current_dir) => (sysroot, current_dir),
    };
    rustc_discover_host_tuple(extra_env, sysroot, current_dir).map(|it| vec![it])
}

fn rustc_discover_host_tuple(
    extra_env: &FxHashMap<String, Option<String>>,
    sysroot: &Sysroot,
    current_dir: &Path,
) -> anyhow::Result<String> {
    let mut cmd = sysroot.tool(Tool::Rustc, current_dir, extra_env);
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

fn cargo_config_build_target(config: &CargoConfigFile) -> Option<Vec<String>> {
    match parse_json_cargo_config_build_target(config) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("Failed to discover cargo config build target {e:?}");
            None
        }
    }
}

// Parses `"build.target = [target-tuple, target-tuple, ...]"` or `"build.target = "target-tuple"`
fn parse_json_cargo_config_build_target(
    config: &CargoConfigFile,
) -> anyhow::Result<Option<Vec<String>>> {
    let target = config.get("build").and_then(|v| v.as_object()).and_then(|m| m.get("target"));
    match target {
        Some(serde_json::Value::String(s)) => Ok(Some(vec![s.to_owned()])),
        Some(v) => serde_json::from_value(v.clone())
            .map(Option::Some)
            .context("Failed to parse `build.target` as an array of target"),
        // t`error: config value `build.target` is not set`, in which case we
        // don't wanna log the error
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use paths::{AbsPathBuf, Utf8PathBuf};

    use crate::{ManifestPath, Sysroot};

    use super::*;

    #[test]
    fn cargo() {
        let manifest_path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let sysroot = Sysroot::empty();
        let manifest_path =
            ManifestPath::try_from(AbsPathBuf::assert(Utf8PathBuf::from(manifest_path))).unwrap();
        let cfg = QueryConfig::Cargo(&sysroot, &manifest_path, &None);
        assert!(get(cfg, None, &FxHashMap::default()).is_ok());
    }

    #[test]
    fn rustc() {
        let sysroot = Sysroot::empty();
        let cfg = QueryConfig::Rustc(&sysroot, env!("CARGO_MANIFEST_DIR").as_ref());
        assert!(get(cfg, None, &FxHashMap::default()).is_ok());
    }
}
