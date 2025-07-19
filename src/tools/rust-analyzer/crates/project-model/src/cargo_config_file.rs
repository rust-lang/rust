//! Read `.cargo/config.toml` as a JSON object
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{ManifestPath, Sysroot, utf8_stdout};

pub(crate) type CargoConfigFile = serde_json::Map<String, serde_json::Value>;

pub(crate) fn read(
    manifest: &ManifestPath,
    extra_env: &FxHashMap<String, Option<String>>,
    sysroot: &Sysroot,
) -> Option<CargoConfigFile> {
    let mut cargo_config = sysroot.tool(Tool::Cargo, manifest.parent(), extra_env);
    cargo_config
        .args(["-Z", "unstable-options", "config", "get", "--format", "json"])
        .env("RUSTC_BOOTSTRAP", "1");
    if manifest.is_rust_manifest() {
        cargo_config.arg("-Zscript");
    }

    tracing::debug!("Discovering cargo config by {:?}", cargo_config);
    let json: serde_json::Map<String, serde_json::Value> = utf8_stdout(&mut cargo_config)
        .inspect(|json| {
            tracing::debug!("Discovered cargo config: {:?}", json);
        })
        .inspect_err(|err| {
            tracing::debug!("Failed to discover cargo config: {:?}", err);
        })
        .ok()
        .and_then(|stdout| serde_json::from_str(&stdout).ok())?;

    Some(json)
}
