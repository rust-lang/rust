//! Runs `rustc --print target-spec-json` to get the target_data_layout.

use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{utf8_stdout, ManifestPath, Sysroot};

/// Determines how `rustc --print target-spec-json` is discovered and invoked.
pub enum RustcDataLayoutConfig<'a> {
    /// Use `rustc --print target-spec-json`, either from with the binary from the sysroot or by discovering via
    /// [`toolchain::rustc`].
    Rustc(Option<&'a Sysroot>),
    /// Use `cargo --print target-spec-json`, either from with the binary from the sysroot or by discovering via
    /// [`toolchain::cargo`].
    Cargo(Option<&'a Sysroot>, &'a ManifestPath),
}

pub fn get(
    config: RustcDataLayoutConfig<'_>,
    target: Option<&str>,
    extra_env: &FxHashMap<String, String>,
) -> anyhow::Result<String> {
    let process = |output: String| {
        (|| Some(output.split_once(r#""data-layout": ""#)?.1.split_once('"')?.0.to_owned()))()
            .ok_or_else(|| {
                anyhow::format_err!("could not fetch target-spec-json from command output")
            })
    };
    let sysroot = match config {
        RustcDataLayoutConfig::Cargo(sysroot, cargo_toml) => {
            let mut cmd = Sysroot::tool(sysroot, Tool::Cargo);
            cmd.envs(extra_env);
            cmd.current_dir(cargo_toml.parent())
                .args([
                    "rustc",
                    "-Z",
                    "unstable-options",
                    "--print",
                    "target-spec-json",
                    "--",
                    "-Z",
                    "unstable-options",
                ])
                .env("RUSTC_BOOTSTRAP", "1");
            if let Some(target) = target {
                cmd.args(["--target", target]);
            }
            match utf8_stdout(cmd) {
                Ok(output) => return process(output),
                Err(e) => {
                    tracing::warn!("failed to run `cargo rustc --print target-spec-json`, falling back to invoking rustc directly: {e}");
                    sysroot
                }
            }
        }
        RustcDataLayoutConfig::Rustc(sysroot) => sysroot,
    };

    let mut cmd = Sysroot::tool(sysroot, Tool::Rustc);
    cmd.envs(extra_env)
        .args(["-Z", "unstable-options", "--print", "target-spec-json"])
        .env("RUSTC_BOOTSTRAP", "1");
    if let Some(target) = target {
        cmd.args(["--target", target]);
    }
    process(utf8_stdout(cmd)?)
}
