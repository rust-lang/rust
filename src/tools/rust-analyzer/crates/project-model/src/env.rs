//! Cargo-like environment variables injection.
use base_db::Env;
use paths::Utf8Path;
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{ManifestPath, PackageData, Sysroot, TargetKind, utf8_stdout};

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
    env.set("CARGO_MANIFEST_PATH", package.manifest.as_str());

    env.set("CARGO_PKG_VERSION", package.version.to_string());
    env.set("CARGO_PKG_VERSION_MAJOR", package.version.major.to_string());
    env.set("CARGO_PKG_VERSION_MINOR", package.version.minor.to_string());
    env.set("CARGO_PKG_VERSION_PATCH", package.version.patch.to_string());
    env.set("CARGO_PKG_VERSION_PRE", package.version.pre.to_string());

    env.set("CARGO_PKG_AUTHORS", package.authors.join(":"));

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
    extra_env: &FxHashMap<String, Option<String>>,
    sysroot: &Sysroot,
) -> Env {
    let mut cargo_config = sysroot.tool(Tool::Cargo, manifest.parent(), extra_env);
    cargo_config
        .args(["-Z", "unstable-options", "config", "get", "env"])
        .env("RUSTC_BOOTSTRAP", "1");
    if manifest.is_rust_manifest() {
        cargo_config.arg("-Zscript");
    }
    // if successful we receive `env.key.value = "value" per entry
    tracing::debug!("Discovering cargo config env by {:?}", cargo_config);
    utf8_stdout(&mut cargo_config)
        .map(|stdout| parse_output_cargo_config_env(manifest, &stdout))
        .inspect(|env| {
            tracing::debug!("Discovered cargo config env: {:?}", env);
        })
        .inspect_err(|err| {
            tracing::debug!("Failed to discover cargo config env: {:?}", err);
        })
        .unwrap_or_default()
}

fn parse_output_cargo_config_env(manifest: &ManifestPath, stdout: &str) -> Env {
    let mut env = Env::default();
    let mut relatives = vec![];
    for (key, val) in
        stdout.lines().filter_map(|l| l.strip_prefix("env.")).filter_map(|l| l.split_once(" = "))
    {
        let val = val.trim_matches('"').to_owned();
        if let Some((key, modifier)) = key.split_once('.') {
            match modifier {
                "relative" => relatives.push((key, val)),
                "value" => _ = env.insert(key, val),
                _ => {
                    tracing::warn!(
                        "Unknown modifier in cargo config env: {}, expected `relative` or `value`",
                        modifier
                    );
                    continue;
                }
            }
        } else {
            env.insert(key, val);
        }
    }
    // FIXME: The base here should be the parent of the `.cargo/config` file, not the manifest.
    // But cargo does not provide this information.
    let base = <_ as AsRef<Utf8Path>>::as_ref(manifest.parent());
    for (key, relative) in relatives {
        if relative != "true" {
            continue;
        }
        if let Some(suffix) = env.get(key) {
            env.insert(key, base.join(suffix).to_string());
        }
    }
    env
}

#[test]
fn parse_output_cargo_config_env_works() {
    let stdout = r#"
env.CARGO_WORKSPACE_DIR.relative = true
env.CARGO_WORKSPACE_DIR.value = ""
env.RELATIVE.relative = true
env.RELATIVE.value = "../relative"
env.INVALID.relative = invalidbool
env.INVALID.value = "../relative"
env.TEST.value = "test"
"#
    .trim();
    let cwd = paths::Utf8PathBuf::try_from(std::env::current_dir().unwrap()).unwrap();
    let manifest = paths::AbsPathBuf::assert(cwd.join("Cargo.toml"));
    let manifest = ManifestPath::try_from(manifest).unwrap();
    let env = parse_output_cargo_config_env(&manifest, stdout);
    assert_eq!(env.get("CARGO_WORKSPACE_DIR").as_deref(), Some(cwd.join("").as_str()));
    assert_eq!(env.get("RELATIVE").as_deref(), Some(cwd.join("../relative").as_str()));
    assert_eq!(env.get("INVALID").as_deref(), Some("../relative"));
    assert_eq!(env.get("TEST").as_deref(), Some("test"));
}
