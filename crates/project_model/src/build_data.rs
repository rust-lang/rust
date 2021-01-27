//! Handles build script specific information

use std::{
    ffi::OsStr,
    io::BufReader,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::Result;
use cargo_metadata::{BuildScript, Message, Package, PackageId};
use itertools::Itertools;
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use stdx::JodChild;

use crate::{cfg_flag::CfgFlag, CargoConfig};

#[derive(Debug, Clone, Default)]
pub(crate) struct BuildDataMap {
    data: FxHashMap<PackageId, BuildData>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BuildData {
    /// List of config flags defined by this package's build script
    pub cfgs: Vec<CfgFlag>,
    /// List of cargo-related environment variables with their value
    ///
    /// If the package has a build script which defines environment variables,
    /// they can also be found here.
    pub envs: Vec<(String, String)>,
    /// Directory where a build script might place its output
    pub out_dir: Option<AbsPathBuf>,
    /// Path to the proc-macro library file if this package exposes proc-macros
    pub proc_macro_dylib_path: Option<AbsPathBuf>,
}

impl BuildDataMap {
    pub(crate) fn new(
        cargo_toml: &AbsPath,
        cargo_features: &CargoConfig,
        packages: &Vec<Package>,
        progress: &dyn Fn(String),
    ) -> Result<BuildDataMap> {
        let mut cmd = Command::new(toolchain::cargo());
        cmd.args(&["check", "--workspace", "--message-format=json", "--manifest-path"])
            .arg(cargo_toml.as_ref());

        // --all-targets includes tests, benches and examples in addition to the
        // default lib and bins. This is an independent concept from the --targets
        // flag below.
        cmd.arg("--all-targets");

        if let Some(target) = &cargo_features.target {
            cmd.args(&["--target", target]);
        }

        if cargo_features.all_features {
            cmd.arg("--all-features");
        } else {
            if cargo_features.no_default_features {
                // FIXME: `NoDefaultFeatures` is mutual exclusive with `SomeFeatures`
                // https://github.com/oli-obk/cargo_metadata/issues/79
                cmd.arg("--no-default-features");
            }
            if !cargo_features.features.is_empty() {
                cmd.arg("--features");
                cmd.arg(cargo_features.features.join(" "));
            }
        }

        cmd.stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null());

        let mut child = cmd.spawn().map(JodChild)?;
        let child_stdout = child.stdout.take().unwrap();
        let stdout = BufReader::new(child_stdout);

        let mut res = BuildDataMap::default();
        for message in cargo_metadata::Message::parse_stream(stdout) {
            if let Ok(message) = message {
                match message {
                    Message::BuildScriptExecuted(BuildScript {
                        package_id,
                        out_dir,
                        cfgs,
                        env,
                        ..
                    }) => {
                        let cfgs = {
                            let mut acc = Vec::new();
                            for cfg in cfgs {
                                match cfg.parse::<CfgFlag>() {
                                    Ok(it) => acc.push(it),
                                    Err(err) => {
                                        anyhow::bail!("invalid cfg from cargo-metadata: {}", err)
                                    }
                                };
                            }
                            acc
                        };
                        let res = res.data.entry(package_id.clone()).or_default();
                        // cargo_metadata crate returns default (empty) path for
                        // older cargos, which is not absolute, so work around that.
                        if out_dir != PathBuf::default() {
                            let out_dir = AbsPathBuf::assert(out_dir);
                            res.out_dir = Some(out_dir);
                            res.cfgs = cfgs;
                        }

                        res.envs = env;
                    }
                    Message::CompilerArtifact(message) => {
                        progress(format!("metadata {}", message.target.name));

                        if message.target.kind.contains(&"proc-macro".to_string()) {
                            let package_id = message.package_id;
                            // Skip rmeta file
                            if let Some(filename) =
                                message.filenames.iter().find(|name| is_dylib(name))
                            {
                                let filename = AbsPathBuf::assert(filename.clone());
                                let res = res.data.entry(package_id.clone()).or_default();
                                res.proc_macro_dylib_path = Some(filename);
                            }
                        }
                    }
                    Message::CompilerMessage(message) => {
                        progress(message.target.name.clone());
                    }
                    Message::Unknown => (),
                    Message::BuildFinished(_) => {}
                    Message::TextLine(_) => {}
                }
            }
        }
        res.inject_cargo_env(packages);
        Ok(res)
    }

    pub(crate) fn with_cargo_env(packages: &Vec<Package>) -> Self {
        let mut res = Self::default();
        res.inject_cargo_env(packages);
        res
    }

    pub(crate) fn get(&self, id: &PackageId) -> Option<&BuildData> {
        self.data.get(id)
    }

    fn inject_cargo_env(&mut self, packages: &Vec<Package>) {
        for meta_pkg in packages {
            let resource = self.data.entry(meta_pkg.id.clone()).or_default();
            inject_cargo_env(meta_pkg, &mut resource.envs);

            if let Some(out_dir) = &resource.out_dir {
                // NOTE: cargo and rustc seem to hide non-UTF-8 strings from env! and option_env!()
                if let Some(out_dir) = out_dir.to_str().map(|s| s.to_owned()) {
                    resource.envs.push(("OUT_DIR".to_string(), out_dir));
                }
            }
        }
    }
}

// FIXME: File a better way to know if it is a dylib
fn is_dylib(path: &Path) -> bool {
    match path.extension().and_then(OsStr::to_str).map(|it| it.to_string().to_lowercase()) {
        None => false,
        Some(ext) => matches!(ext.as_str(), "dll" | "dylib" | "so"),
    }
}

/// Recreates the compile-time environment variables that Cargo sets.
///
/// Should be synced with <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates>
fn inject_cargo_env(package: &cargo_metadata::Package, env: &mut Vec<(String, String)>) {
    // FIXME: Missing variables:
    // CARGO_PKG_HOMEPAGE, CARGO_CRATE_NAME, CARGO_BIN_NAME, CARGO_BIN_EXE_<name>

    let mut manifest_dir = package.manifest_path.clone();
    manifest_dir.pop();
    if let Some(cargo_manifest_dir) = manifest_dir.to_str() {
        env.push(("CARGO_MANIFEST_DIR".into(), cargo_manifest_dir.into()));
    }

    // Not always right, but works for common cases.
    env.push(("CARGO".into(), "cargo".into()));

    env.push(("CARGO_PKG_VERSION".into(), package.version.to_string()));
    env.push(("CARGO_PKG_VERSION_MAJOR".into(), package.version.major.to_string()));
    env.push(("CARGO_PKG_VERSION_MINOR".into(), package.version.minor.to_string()));
    env.push(("CARGO_PKG_VERSION_PATCH".into(), package.version.patch.to_string()));

    let pre = package.version.pre.iter().map(|id| id.to_string()).format(".");
    env.push(("CARGO_PKG_VERSION_PRE".into(), pre.to_string()));

    let authors = package.authors.join(";");
    env.push(("CARGO_PKG_AUTHORS".into(), authors));

    env.push(("CARGO_PKG_NAME".into(), package.name.clone()));
    env.push(("CARGO_PKG_DESCRIPTION".into(), package.description.clone().unwrap_or_default()));
    //env.push(("CARGO_PKG_HOMEPAGE".into(), package.homepage.clone().unwrap_or_default()));
    env.push(("CARGO_PKG_REPOSITORY".into(), package.repository.clone().unwrap_or_default()));
    env.push(("CARGO_PKG_LICENSE".into(), package.license.clone().unwrap_or_default()));

    let license_file =
        package.license_file.as_ref().map(|buf| buf.display().to_string()).unwrap_or_default();
    env.push(("CARGO_PKG_LICENSE_FILE".into(), license_file));
}
