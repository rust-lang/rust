//! Handles build script specific information

use std::{
    io::BufReader,
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
};

use anyhow::Result;
use cargo_metadata::camino::Utf8Path;
use cargo_metadata::{BuildScript, Message};
use itertools::Itertools;
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use stdx::{format_to, JodChild};

use crate::{cfg_flag::CfgFlag, CargoConfig};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PackageBuildData {
    /// List of config flags defined by this package's build script
    pub(crate) cfgs: Vec<CfgFlag>,
    /// List of cargo-related environment variables with their value
    ///
    /// If the package has a build script which defines environment variables,
    /// they can also be found here.
    pub(crate) envs: Vec<(String, String)>,
    /// Directory where a build script might place its output
    pub(crate) out_dir: Option<AbsPathBuf>,
    /// Path to the proc-macro library file if this package exposes proc-macros
    pub(crate) proc_macro_dylib_path: Option<AbsPathBuf>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub(crate) struct WorkspaceBuildData {
    per_package: FxHashMap<String, PackageBuildData>,
    error: Option<String>,
}

#[derive(Debug, Default, PartialEq, Eq, Clone)]
pub struct BuildDataResult {
    per_workspace: FxHashMap<AbsPathBuf, WorkspaceBuildData>,
}

#[derive(Clone, Debug)]
pub(crate) struct BuildDataConfig {
    cargo_toml: AbsPathBuf,
    cargo_features: CargoConfig,
    packages: Arc<Vec<cargo_metadata::Package>>,
}

impl PartialEq for BuildDataConfig {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.packages, &other.packages)
    }
}

impl Eq for BuildDataConfig {}

#[derive(Debug)]
pub struct BuildDataCollector {
    wrap_rustc: bool,
    configs: FxHashMap<AbsPathBuf, BuildDataConfig>,
}

impl BuildDataCollector {
    pub fn new(wrap_rustc: bool) -> Self {
        Self { wrap_rustc, configs: FxHashMap::default() }
    }

    pub(crate) fn add_config(&mut self, workspace_root: &AbsPath, config: BuildDataConfig) {
        self.configs.insert(workspace_root.to_path_buf(), config);
    }

    pub fn collect(&mut self, progress: &dyn Fn(String)) -> Result<BuildDataResult> {
        let mut res = BuildDataResult::default();
        for (path, config) in self.configs.iter() {
            let workspace_build_data = WorkspaceBuildData::collect(
                &config.cargo_toml,
                &config.cargo_features,
                &config.packages,
                self.wrap_rustc,
                progress,
            )?;
            res.per_workspace.insert(path.clone(), workspace_build_data);
        }
        Ok(res)
    }
}

impl WorkspaceBuildData {
    pub(crate) fn get(&self, package_id: &str) -> Option<&PackageBuildData> {
        self.per_package.get(package_id)
    }
}

impl BuildDataResult {
    pub(crate) fn get(&self, workspace_root: &AbsPath) -> Option<&WorkspaceBuildData> {
        self.per_workspace.get(workspace_root)
    }
    pub fn error(&self) -> Option<String> {
        let mut buf = String::new();
        for (_workspace_root, build_data) in &self.per_workspace {
            if let Some(err) = &build_data.error {
                format_to!(buf, "cargo check failed:\n{}", err);
            }
        }
        if buf.is_empty() {
            return None;
        }

        Some(buf)
    }
}

impl BuildDataConfig {
    pub(crate) fn new(
        cargo_toml: AbsPathBuf,
        cargo_features: CargoConfig,
        packages: Arc<Vec<cargo_metadata::Package>>,
    ) -> Self {
        Self { cargo_toml, cargo_features, packages }
    }
}

impl WorkspaceBuildData {
    fn collect(
        cargo_toml: &AbsPath,
        cargo_features: &CargoConfig,
        packages: &Vec<cargo_metadata::Package>,
        wrap_rustc: bool,
        progress: &dyn Fn(String),
    ) -> Result<WorkspaceBuildData> {
        let mut cmd = Command::new(toolchain::cargo());

        if wrap_rustc {
            // Setup RUSTC_WRAPPER to point to `rust-analyzer` binary itself. We use
            // that to compile only proc macros and build scripts during the initial
            // `cargo check`.
            let myself = std::env::current_exe()?;
            cmd.env("RUSTC_WRAPPER", myself);
            cmd.env("RA_RUSTC_WRAPPER", "1");
        }

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

        cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());

        let mut child = cmd.spawn().map(JodChild)?;
        let child_stdout = child.stdout.take().unwrap();
        let stdout = BufReader::new(child_stdout);

        let mut res = WorkspaceBuildData::default();
        for message in cargo_metadata::Message::parse_stream(stdout).flatten() {
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
                    let package_build_data =
                        res.per_package.entry(package_id.repr.clone()).or_default();
                    // cargo_metadata crate returns default (empty) path for
                    // older cargos, which is not absolute, so work around that.
                    if !out_dir.as_str().is_empty() {
                        let out_dir = AbsPathBuf::assert(PathBuf::from(out_dir.into_os_string()));
                        package_build_data.out_dir = Some(out_dir);
                        package_build_data.cfgs = cfgs;
                    }

                    package_build_data.envs = env;
                }
                Message::CompilerArtifact(message) => {
                    progress(format!("metadata {}", message.target.name));

                    if message.target.kind.contains(&"proc-macro".to_string()) {
                        let package_id = message.package_id;
                        // Skip rmeta file
                        if let Some(filename) = message.filenames.iter().find(|name| is_dylib(name))
                        {
                            let filename = AbsPathBuf::assert(PathBuf::from(&filename));
                            let package_build_data =
                                res.per_package.entry(package_id.repr.clone()).or_default();
                            package_build_data.proc_macro_dylib_path = Some(filename);
                        }
                    }
                }
                Message::CompilerMessage(message) => {
                    progress(message.target.name.clone());
                }
                Message::BuildFinished(_) => {}
                Message::TextLine(_) => {}
                _ => {}
            }
        }

        for package in packages {
            let package_build_data = res.per_package.entry(package.id.repr.clone()).or_default();
            inject_cargo_env(package, package_build_data);
            if let Some(out_dir) = &package_build_data.out_dir {
                // NOTE: cargo and rustc seem to hide non-UTF-8 strings from env! and option_env!()
                if let Some(out_dir) = out_dir.to_str().map(|s| s.to_owned()) {
                    package_build_data.envs.push(("OUT_DIR".to_string(), out_dir));
                }
            }
        }

        let output = child.into_inner().wait_with_output()?;
        if !output.status.success() {
            let mut stderr = String::from_utf8(output.stderr).unwrap_or_default();
            if stderr.is_empty() {
                stderr = "cargo check failed".to_string();
            }
            res.error = Some(stderr)
        }

        Ok(res)
    }
}

// FIXME: File a better way to know if it is a dylib
fn is_dylib(path: &Utf8Path) -> bool {
    match path.extension().map(|e| e.to_string().to_lowercase()) {
        None => false,
        Some(ext) => matches!(ext.as_str(), "dll" | "dylib" | "so"),
    }
}

/// Recreates the compile-time environment variables that Cargo sets.
///
/// Should be synced with <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates>
fn inject_cargo_env(package: &cargo_metadata::Package, build_data: &mut PackageBuildData) {
    let env = &mut build_data.envs;

    // FIXME: Missing variables:
    // CARGO_PKG_HOMEPAGE, CARGO_CRATE_NAME, CARGO_BIN_NAME, CARGO_BIN_EXE_<name>

    let mut manifest_dir = package.manifest_path.clone();
    manifest_dir.pop();
    env.push(("CARGO_MANIFEST_DIR".into(), manifest_dir.into_string()));

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

    let license_file = package.license_file.as_ref().map(|buf| buf.to_string()).unwrap_or_default();
    env.push(("CARGO_PKG_LICENSE_FILE".into(), license_file));
}
