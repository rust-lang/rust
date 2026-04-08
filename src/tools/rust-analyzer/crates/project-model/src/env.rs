//! Cargo-like environment variables injection.
use base_db::Env;
use paths::Utf8Path;
use rustc_hash::FxHashMap;

use crate::{PackageData, TargetKind, cargo_config_file::CargoConfigFile};

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

pub(crate) fn inject_cargo_env(env: &mut Env, cargo_path: &Utf8Path) {
    env.set("CARGO", cargo_path.as_str());
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
    config: &Option<CargoConfigFile>,
    extra_env: &FxHashMap<String, Option<String>>,
) -> Env {
    use toml::de::*;

    let mut env = Env::default();
    env.extend(extra_env.iter().filter_map(|(k, v)| v.as_ref().map(|v| (k.clone(), v.clone()))));

    let Some(config_reader) = config.as_ref().and_then(|c| c.read()) else {
        return env;
    };
    let Some(env_toml) = config_reader.get(["env"]).and_then(|it| it.as_table()) else {
        return env;
    };

    for (key, entry) in env_toml {
        let key = key.as_ref().as_ref();
        let value = match entry.as_ref() {
            DeValue::String(s) => {
                // Plain string entries have no `force` option, so they should not
                // override existing environment variables (matching Cargo behavior).
                if extra_env.get(key).is_some_and(Option::is_some) {
                    continue;
                }
                if let Ok(val) = std::env::var(key) { val } else { String::from(s.clone()) }
            }
            DeValue::Table(entry) => {
                // Each entry MUST have a `value` key.
                let Some(map) = entry.get("value").and_then(|v| v.as_ref().as_str()) else {
                    continue;
                };
                let is_forced =
                    entry.get("force").and_then(|v| v.as_ref().as_bool()).unwrap_or(false);
                // If the entry already exists in the environment AND the `force` key is not set
                // to true, use the existing value instead of the config value.
                if !is_forced {
                    if extra_env.get(key).is_some_and(Option::is_some) {
                        continue;
                    }
                    if let Ok(val) = std::env::var(key) {
                        env.insert(key, val);
                        continue;
                    }
                }

                if let Some(base) = entry.get("relative").and_then(|v| {
                    if v.as_ref().as_bool().is_some_and(std::convert::identity) {
                        config_reader.get_origin_root(v)
                    } else {
                        None
                    }
                }) {
                    base.join(map).to_string()
                } else {
                    map.to_owned()
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
    use itertools::Itertools;

    let cwd = paths::AbsPathBuf::try_from(
        paths::Utf8PathBuf::try_from(std::env::current_dir().unwrap()).unwrap(),
    )
    .unwrap();
    let config_path = cwd.join(".cargo").join("config.toml");
    let raw = r#"
env.RA_TEST_WORKSPACE_DIR.relative = true
env.RA_TEST_WORKSPACE_DIR.value = ""
env.RA_TEST_INVALID.relative = "invalidbool"
env.RA_TEST_INVALID.value = "../relative"
env.RA_TEST_RELATIVE.relative = true
env.RA_TEST_RELATIVE.value = "../relative"
env.RA_TEST_UNSET.value = "test"
env.RA_TEST_FORCED.value = "test"
env.RA_TEST_FORCED.force = true
env.RA_TEST_UNFORCED.value = "test"
env.RA_TEST_UNFORCED.forced = false
env.RA_TEST_OVERWRITTEN.value = "test"
env.RA_TEST_NOT_AN_OBJECT = "value"
"#;
    let raw = raw.lines().map(|l| format!("{l} # {config_path}")).join("\n");
    let config = CargoConfigFile::from_string_for_test(raw);
    let extra_env = [
        ("RA_TEST_FORCED", Some("ignored")),
        ("RA_TEST_UNFORCED", Some("newvalue")),
        ("RA_TEST_OVERWRITTEN", Some("newvalue")),
        ("RA_TEST_UNSET", None),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.map(ToString::to_string)))
    .collect();
    let env = cargo_config_env(&Some(config), &extra_env);
    assert_eq!(env.get("RA_TEST_WORKSPACE_DIR").as_deref(), Some(cwd.join("").as_str()));
    assert_eq!(env.get("RA_TEST_RELATIVE").as_deref(), Some(cwd.join("../relative").as_str()));
    assert_eq!(env.get("RA_TEST_INVALID").as_deref(), Some("../relative"));
    assert_eq!(env.get("RA_TEST_UNSET").as_deref(), Some("test"));
    assert_eq!(env.get("RA_TEST_FORCED").as_deref(), Some("test"));
    assert_eq!(env.get("RA_TEST_UNFORCED").as_deref(), Some("newvalue"));
    assert_eq!(env.get("RA_TEST_OVERWRITTEN").as_deref(), Some("newvalue"));
    assert_eq!(env.get("RA_TEST_NOT_AN_OBJECT").as_deref(), Some("value"));
}
