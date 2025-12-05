//! Read `.cargo/config.toml` as a TOML table
use paths::{AbsPath, Utf8Path, Utf8PathBuf};
use rustc_hash::FxHashMap;
use toml::{
    Spanned,
    de::{DeTable, DeValue},
};
use toolchain::Tool;

use crate::{ManifestPath, Sysroot, utf8_stdout};

#[derive(Clone)]
pub struct CargoConfigFile(String);

impl CargoConfigFile {
    pub(crate) fn load(
        manifest: &ManifestPath,
        extra_env: &FxHashMap<String, Option<String>>,
        sysroot: &Sysroot,
    ) -> Option<Self> {
        let mut cargo_config = sysroot.tool(Tool::Cargo, manifest.parent(), extra_env);
        cargo_config
            .args(["-Z", "unstable-options", "config", "get", "--format", "toml", "--show-origin"])
            .env("RUSTC_BOOTSTRAP", "1");
        if manifest.is_rust_manifest() {
            cargo_config.arg("-Zscript");
        }

        tracing::debug!("Discovering cargo config by {cargo_config:?}");
        utf8_stdout(&mut cargo_config)
            .inspect(|toml| {
                tracing::debug!("Discovered cargo config: {toml:?}");
            })
            .inspect_err(|err| {
                tracing::debug!("Failed to discover cargo config: {err:?}");
            })
            .ok()
            .map(CargoConfigFile)
    }

    pub(crate) fn read<'a>(&'a self) -> Option<CargoConfigFileReader<'a>> {
        CargoConfigFileReader::new(&self.0)
    }

    #[cfg(test)]
    pub(crate) fn from_string_for_test(s: String) -> Self {
        CargoConfigFile(s)
    }
}

pub(crate) struct CargoConfigFileReader<'a> {
    toml_str: &'a str,
    line_ends: Vec<usize>,
    table: Spanned<DeTable<'a>>,
}

impl<'a> CargoConfigFileReader<'a> {
    fn new(toml_str: &'a str) -> Option<Self> {
        let toml = DeTable::parse(toml_str)
            .inspect_err(|err| tracing::debug!("Failed to parse cargo config into toml: {err:?}"))
            .ok()?;
        let mut last_line_end = 0;
        let line_ends = toml_str
            .lines()
            .map(|l| {
                last_line_end += l.len() + 1;
                last_line_end
            })
            .collect();

        Some(CargoConfigFileReader { toml_str, table: toml, line_ends })
    }

    pub(crate) fn get_spanned(
        &self,
        accessor: impl IntoIterator<Item = &'a str>,
    ) -> Option<&Spanned<DeValue<'a>>> {
        let mut keys = accessor.into_iter();
        let mut val = self.table.get_ref().get(keys.next()?)?;
        for key in keys {
            let DeValue::Table(map) = val.get_ref() else { return None };
            val = map.get(key)?;
        }
        Some(val)
    }

    pub(crate) fn get(&self, accessor: impl IntoIterator<Item = &'a str>) -> Option<&DeValue<'a>> {
        self.get_spanned(accessor).map(|it| it.as_ref())
    }

    pub(crate) fn get_origin_root(&self, spanned: &Spanned<DeValue<'a>>) -> Option<&AbsPath> {
        let span = spanned.span();

        for &line_end in &self.line_ends {
            if line_end < span.end {
                continue;
            }

            let after_span = &self.toml_str[span.end..line_end];

            // table.key = "value" # /parent/.cargo/config.toml
            //                   |                            |
            //                   span.end                     line_end
            let origin_path = after_span
                .strip_prefix([',']) // strip trailing comma
                .unwrap_or(after_span)
                .trim_start()
                .strip_prefix(['#'])
                .and_then(|path| {
                    let path = path.trim();
                    if path.starts_with("environment variable")
                        || path.starts_with("--config cli option")
                    {
                        None
                    } else {
                        Some(path)
                    }
                });

            return origin_path.and_then(|path| {
                <&Utf8Path>::from(path)
                    .try_into()
                    .ok()
                    // Two levels up to the config file.
                    // See https://doc.rust-lang.org/cargo/reference/config.html#config-relative-paths
                    .and_then(AbsPath::parent)
                    .and_then(AbsPath::parent)
            });
        }

        None
    }
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

#[test]
fn cargo_config_file_reader_works() {
    #[cfg(target_os = "windows")]
    let root = "C://ROOT";

    #[cfg(not(target_os = "windows"))]
    let root = "/ROOT";

    let toml = format!(
        r##"
alias.foo = "abc"
alias.bar = "🙂" # {root}/home/.cargo/config.toml
alias.sub-example = [
    "sub", # {root}/foo/.cargo/config.toml
    "example", # {root}/❤️💛💙/💝/.cargo/config.toml
]
build.rustflags = [
    "--flag", # {root}/home/.cargo/config.toml
    "env", # environment variable `CARGO_BUILD_RUSTFLAGS`
    "cli", # --config cli option
]
env.CARGO_WORKSPACE_DIR.relative = true # {root}/home/.cargo/config.toml
env.CARGO_WORKSPACE_DIR.value = "" # {root}/home/.cargo/config.toml
"##
    );

    let reader = CargoConfigFileReader::new(&toml).unwrap();

    let alias_foo = reader.get_spanned(["alias", "foo"]).unwrap();
    assert_eq!(alias_foo.as_ref().as_str().unwrap(), "abc");
    assert!(reader.get_origin_root(alias_foo).is_none());

    let alias_bar = reader.get_spanned(["alias", "bar"]).unwrap();
    assert_eq!(alias_bar.as_ref().as_str().unwrap(), "🙂");
    assert_eq!(reader.get_origin_root(alias_bar).unwrap().as_str(), format!("{root}/home"));

    let alias_sub_example = reader.get_spanned(["alias", "sub-example"]).unwrap();
    assert!(reader.get_origin_root(alias_sub_example).is_none());
    let alias_sub_example = alias_sub_example.as_ref().as_array().unwrap();

    assert_eq!(alias_sub_example[0].get_ref().as_str().unwrap(), "sub");
    assert_eq!(
        reader.get_origin_root(&alias_sub_example[0]).unwrap().as_str(),
        format!("{root}/foo")
    );

    assert_eq!(alias_sub_example[1].get_ref().as_str().unwrap(), "example");
    assert_eq!(
        reader.get_origin_root(&alias_sub_example[1]).unwrap().as_str(),
        format!("{root}/❤️💛💙/💝")
    );

    let build_rustflags = reader.get(["build", "rustflags"]).unwrap().as_array().unwrap();
    assert_eq!(
        reader.get_origin_root(&build_rustflags[0]).unwrap().as_str(),
        format!("{root}/home")
    );
    assert!(reader.get_origin_root(&build_rustflags[1]).is_none());
    assert!(reader.get_origin_root(&build_rustflags[2]).is_none());

    let env_cargo_workspace_dir =
        reader.get(["env", "CARGO_WORKSPACE_DIR"]).unwrap().as_table().unwrap();
    let env_relative = &env_cargo_workspace_dir["relative"];
    assert!(env_relative.as_ref().as_bool().unwrap());
    assert_eq!(reader.get_origin_root(env_relative).unwrap().as_str(), format!("{root}/home"));

    let env_val = &env_cargo_workspace_dir["value"];
    assert_eq!(env_val.as_ref().as_str().unwrap(), "");
    assert_eq!(reader.get_origin_root(env_val).unwrap().as_str(), format!("{root}/home"));
}
