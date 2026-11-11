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
    match parse_toml_cargo_config_build_target(config) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("Failed to discover cargo config build target {e:?}");
            None
        }
    }
}

// Parses `"build.target = [target-tuple, target-tuple, ...]"` or `"build.target = "target-tuple"`
fn parse_toml_cargo_config_build_target(
    config: &CargoConfigFile,
) -> anyhow::Result<Option<Vec<String>>> {
    let Some(config_reader) = config.read() else {
        return Ok(None);
    };
    let Some(target) = config_reader.get_spanned(["build", "target"]) else {
        return Ok(None);
    };

    // if the target ends with `.json`, join it to the config file's parent dir.
    // See https://github.com/rust-lang/cargo/blob/f7acf448fc127df9a77c52cc2bba027790ac4931/src/cargo/core/compiler/compile_kind.rs#L171-L192
    let join_to_origin_if_json_path = |s: &str, spanned: &toml::Spanned<toml::de::DeValue<'_>>| {
        if s.ends_with(".json") {
            config_reader
                .get_origin_root(spanned)
                .map(|p| p.join(s).to_string())
                .unwrap_or_else(|| s.to_owned())
        } else {
            s.to_owned()
        }
    };

    let parse_err = "Failed to parse `build.target` as an array of target";

    match target.as_ref() {
        toml::de::DeValue::String(s) => {
            Ok(Some(vec![join_to_origin_if_json_path(s.as_ref(), target)]))
        }
        toml::de::DeValue::Array(arr) => arr
            .iter()
            .map(|v| {
                let s = v.as_ref().as_str().context(parse_err)?;
                Ok(join_to_origin_if_json_path(s, v))
            })
            .collect::<anyhow::Result<_>>()
            .map(Option::Some),
        _ => Err(anyhow::anyhow!(parse_err)),
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
