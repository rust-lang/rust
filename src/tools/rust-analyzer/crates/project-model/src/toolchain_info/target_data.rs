//! Runs `rustc --print target-spec-json` to get the target_data_layout.

use anyhow::Context;
use base_db::target;
use rustc_hash::FxHashMap;
use serde_derive::Deserialize;
use toolchain::Tool;

use crate::{Sysroot, toolchain_info::QueryConfig, utf8_stdout};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arch {
    Wasm32,
    Wasm64,
    #[serde(other)]
    Other,
}

impl From<Arch> for target::Arch {
    fn from(value: Arch) -> Self {
        match value {
            Arch::Wasm32 => target::Arch::Wasm32,
            Arch::Wasm64 => target::Arch::Wasm64,
            Arch::Other => target::Arch::Other,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TargetSpec {
    pub data_layout: String,
    pub arch: Arch,
}

/// Uses `rustc --print target-spec-json`.
pub fn get(
    config: QueryConfig<'_>,
    target: Option<&str>,
    extra_env: &FxHashMap<String, Option<String>>,
) -> anyhow::Result<target::TargetData> {
    const RUSTC_ARGS: [&str; 2] = ["--print", "target-spec-json"];
    let process = |output: String| {
        let target_spec = serde_json::from_str::<TargetSpec>(&output).map_err(|_| {
            anyhow::format_err!("could not parse target-spec-json from command output")
        })?;
        Ok(target::TargetData {
            arch: target_spec.arch.into(),
            data_layout: target_spec.data_layout.into_boxed_str(),
        })
    };
    let (sysroot, current_dir) = match config {
        QueryConfig::Cargo(sysroot, cargo_toml, _) => {
            let mut cmd = sysroot.tool(Tool::Cargo, cargo_toml.parent(), extra_env);
            cmd.env("RUSTC_BOOTSTRAP", "1");
            cmd.args(["rustc", "-Z", "unstable-options"]).args(RUSTC_ARGS);
            if let Some(target) = target {
                cmd.args(["--target", target]);
            }
            cmd.args(["--", "-Z", "unstable-options"]);
            match utf8_stdout(&mut cmd) {
                Ok(output) => return process(output),
                Err(e) => {
                    tracing::warn!(%e, "failed to run `{cmd:?}`, falling back to invoking rustc directly");
                    (sysroot, cargo_toml.parent().as_ref())
                }
            }
        }
        QueryConfig::Rustc(sysroot, current_dir) => (sysroot, current_dir),
    };

    let mut cmd = Sysroot::tool(sysroot, Tool::Rustc, current_dir, extra_env);
    cmd.env("RUSTC_BOOTSTRAP", "1").args(["-Z", "unstable-options"]).args(RUSTC_ARGS);
    if let Some(target) = target {
        cmd.args(["--target", target]);
    }
    utf8_stdout(&mut cmd)
        .with_context(|| format!("unable to fetch target-data-layout via `{cmd:?}`"))
        .and_then(process)
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
