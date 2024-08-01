//! Cargo-like environment variables injection.
use base_db::Env;
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{utf8_stdout, ManifestPath, PackageData, Sysroot, TargetKind};

/// Recreates the compile-time environment variables that Cargo sets.
///
/// Should be synced with
/// <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates>
///
/// FIXME: ask Cargo to provide this data instead of re-deriving.
pub(crate) fn inject_cargo_package_env(env: &mut Env, package: &PackageData) {
    // FIXME: Missing variables:
    // CARGO_BIN_NAME, CARGO_BIN_EXE_<name>

    let manifest_dir = package.manifest.parent();
    env.set("CARGO_MANIFEST_DIR", manifest_dir.as_str());

    env.set("CARGO_PKG_VERSION", package.version.to_string());
    env.set("CARGO_PKG_VERSION_MAJOR", package.version.major.to_string());
    env.set("CARGO_PKG_VERSION_MINOR", package.version.minor.to_string());
    env.set("CARGO_PKG_VERSION_PATCH", package.version.patch.to_string());
    env.set("CARGO_PKG_VERSION_PRE", package.version.pre.to_string());

    env.set("CARGO_PKG_AUTHORS", package.authors.join(":").clone());

    env.set("CARGO_PKG_NAME", package.name.clone());
    env.set("CARGO_PKG_DESCRIPTION", package.description.as_deref().unwrap_or_default());
    env.set("CARGO_PKG_HOMEPAGE", package.homepage.as_deref().unwrap_or_default());
    env.set("CARGO_PKG_REPOSITORY", package.repository.as_deref().unwrap_or_default());
    env.set("CARGO_PKG_LICENSE", package.license.as_deref().unwrap_or_default());
    env.set(
        "CARGO_PKG_LICENSE_FILE",
        package.license_file.as_ref().map(ToString::to_string).unwrap_or_default(),
    );
    env.set(
        "CARGO_PKG_README",
        package.readme.as_ref().map(ToString::to_string).unwrap_or_default(),
    );

    env.set(
        "CARGO_PKG_RUST_VERSION",
        package.rust_version.as_ref().map(ToString::to_string).unwrap_or_default(),
    );
}

pub(crate) fn inject_cargo_env(env: &mut Env) {
    env.set("CARGO", Tool::Cargo.path().to_string());
}

pub(crate) fn inject_rustc_tool_env(env: &mut Env, cargo_name: &str, kind: TargetKind) {
    _ = kind;
    // FIXME
    // if kind.is_executable() {
    //     env.set("CARGO_BIN_NAME", cargo_name);
    // }
    env.set("CARGO_CRATE_NAME", cargo_name.replace('-', "_"));
}

pub(crate) fn cargo_config_env(
    manifest: &ManifestPath,
    extra_env: &FxHashMap<String, String>,
    sysroot: &Sysroot,
) -> FxHashMap<String, String> {
    let mut cargo_config = sysroot.tool(Tool::Cargo);
    cargo_config.envs(extra_env);
    cargo_config
        .current_dir(manifest.parent())
        .args(["-Z", "unstable-options", "config", "get", "env"])
        .env("RUSTC_BOOTSTRAP", "1");
    if manifest.is_rust_manifest() {
        cargo_config.arg("-Zscript");
    }
    // if successful we receive `env.key.value = "value" per entry
    tracing::debug!("Discovering cargo config env by {:?}", cargo_config);
    utf8_stdout(cargo_config).map(parse_output_cargo_config_env).unwrap_or_default()
}

fn parse_output_cargo_config_env(stdout: String) -> FxHashMap<String, String> {
    stdout
        .lines()
        .filter_map(|l| l.strip_prefix("env."))
        .filter_map(|l| l.split_once(".value = "))
        .map(|(key, value)| (key.to_owned(), value.trim_matches('"').to_owned()))
        .collect()
}
