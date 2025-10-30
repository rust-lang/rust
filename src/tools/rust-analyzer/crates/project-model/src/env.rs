//! Cargo-like environment variables injection.
use base_db::Env;
use paths::Utf8Path;
use rustc_hash::FxHashMap;
use toolchain::Tool;

use crate::{ManifestPath, PackageData, TargetKind, cargo_config_file::CargoConfigFile};

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
    config: &Option<CargoConfigFile>,
    extra_env: &FxHashMap<String, Option<String>>,
) -> Env {
    let mut env = Env::default();
    env.extend(extra_env.iter().filter_map(|(k, v)| v.as_ref().map(|v| (k.clone(), v.clone()))));

    let Some(serde_json::Value::Object(env_json)) = config.as_ref().and_then(|c| c.get("env"))
    else {
        return env;
    };

    // FIXME: The base here should be the parent of the `.cargo/config` file, not the manifest.
    // But cargo does not provide this information.
    let base = <_ as AsRef<Utf8Path>>::as_ref(manifest.parent());

    for (key, entry) in env_json {
        let value = match entry {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(entry) => {
                // Each entry MUST have a `value` key.
                let Some(value) = entry.get("value").and_then(|v| v.as_str()) else {
                    continue;
                };
                // If the entry already exists in the environment AND the `force` key is not set to
                // true, then don't overwrite the value.
                if extra_env.get(key).is_some_and(Option::is_some)
                    && !entry.get("force").and_then(|v| v.as_bool()).unwrap_or(false)
                {
                    continue;
                }

                if entry
                    .get("relative")
                    .and_then(|v| v.as_bool())
                    .is_some_and(std::convert::identity)
                {
                    base.join(value).to_string()
                } else {
                    value.to_owned()
                }
            }
            _ => continue,
        };

        env.insert(key, value);
    }

    env
}

#[test]
fn parse_output_cargo_config_env_works() {
    let raw = r#"
{
  "env": {
    "CARGO_WORKSPACE_DIR": {
      "relative": true,
      "value": ""
    },
    "INVALID": {
      "relative": "invalidbool",
      "value": "../relative"
    },
    "RELATIVE": {
      "relative": true,
      "value": "../relative"
    },
    "TEST": {
      "value": "test"
    },
    "FORCED": {
      "value": "test",
      "force": true
    },
    "UNFORCED": {
      "value": "test",
      "force": false
    },
    "OVERWRITTEN": {
      "value": "test"
    },
    "NOT_AN_OBJECT": "value"
  }
}
"#;
    let config: CargoConfigFile = serde_json::from_str(raw).unwrap();
    let cwd = paths::Utf8PathBuf::try_from(std::env::current_dir().unwrap()).unwrap();
    let manifest = paths::AbsPathBuf::assert(cwd.join("Cargo.toml"));
    let manifest = ManifestPath::try_from(manifest).unwrap();
    let extra_env = [
        ("FORCED", Some("ignored")),
        ("UNFORCED", Some("newvalue")),
        ("OVERWRITTEN", Some("newvalue")),
        ("TEST", None),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.map(ToString::to_string)))
    .collect();
    let env = cargo_config_env(&manifest, &Some(config), &extra_env);
    assert_eq!(env.get("CARGO_WORKSPACE_DIR").as_deref(), Some(cwd.join("").as_str()));
    assert_eq!(env.get("RELATIVE").as_deref(), Some(cwd.join("../relative").as_str()));
    assert_eq!(env.get("INVALID").as_deref(), Some("../relative"));
    assert_eq!(env.get("TEST").as_deref(), Some("test"));
    assert_eq!(env.get("FORCED").as_deref(), Some("test"));
    assert_eq!(env.get("UNFORCED").as_deref(), Some("newvalue"));
    assert_eq!(env.get("OVERWRITTEN").as_deref(), Some("newvalue"));
    assert_eq!(env.get("NOT_AN_OBJECT").as_deref(), Some("value"));
}
