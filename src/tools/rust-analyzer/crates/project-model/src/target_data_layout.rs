//! Runs `rustc --print target-spec-json` to get the target_data_layout.
use std::process::Command;

use rustc_hash::FxHashMap;

use crate::{utf8_stdout, ManifestPath};

pub fn get(
    cargo_toml: Option<&ManifestPath>,
    target: Option<&str>,
    extra_env: &FxHashMap<String, String>,
) -> anyhow::Result<String> {
    let output = (|| {
        if let Some(cargo_toml) = cargo_toml {
            let mut cmd = Command::new(toolchain::rustc());
            cmd.envs(extra_env);
            cmd.current_dir(cargo_toml.parent())
                .args(["-Z", "unstable-options", "--print", "target-spec-json"])
                .env("RUSTC_BOOTSTRAP", "1");
            if let Some(target) = target {
                cmd.args(["--target", target]);
            }
            match utf8_stdout(cmd) {
                Ok(it) => return Ok(it),
                Err(e) => tracing::debug!("{e:?}: falling back to querying rustc for cfgs"),
            }
        }
        // using unstable cargo features failed, fall back to using plain rustc
        let mut cmd = Command::new(toolchain::rustc());
        cmd.envs(extra_env)
            .args(["-Z", "unstable-options", "--print", "target-spec-json"])
            .env("RUSTC_BOOTSTRAP", "1");
        if let Some(target) = target {
            cmd.args(["--target", target]);
        }
        utf8_stdout(cmd)
    })()?;
    (|| Some(output.split_once(r#""data-layout": ""#)?.1.split_once('"')?.0.to_owned()))()
        .ok_or_else(|| anyhow::format_err!("could not fetch target-spec-json from command output"))
}
