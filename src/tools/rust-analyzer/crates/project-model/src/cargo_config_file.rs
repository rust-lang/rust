//! Read `.cargo/config.toml` as a JSON object
use paths::{Utf8Path, Utf8PathBuf};
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

pub(crate) fn make_lockfile_copy(
    lockfile_path: &Utf8Path,
) -> Option<(temp_dir::TempDir, Utf8PathBuf)> {
    let temp_dir = temp_dir::TempDir::with_prefix("rust-analyzer").ok()?;
    let target_lockfile = temp_dir.path().join("Cargo.lock").try_into().ok()?;
    match std::fs::copy(lockfile_path, &target_lockfile) {
        Ok(_) => {
            tracing::debug!("Copied lock file from `{}` to `{}`", lockfile_path, target_lockfile);
            Some((temp_dir, target_lockfile))
        }
        // lockfile does not yet exist, so we can just create a new one in the temp dir
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Some((temp_dir, target_lockfile)),
        Err(e) => {
            tracing::warn!(
                "Failed to copy lock file from `{lockfile_path}` to `{target_lockfile}`: {e}",
            );
            None
        }
    }
}
