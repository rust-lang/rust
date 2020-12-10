//! FIXME: write short doc here

use std::{
    ffi::OsStr,
    ops,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result};
use arena::{Arena, Idx};
use base_db::Edition;
use cargo_metadata::{BuildScript, CargoOpt, Message, MetadataCommand, PackageId};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;

use crate::cfg_flag::CfgFlag;

/// `CargoWorkspace` represents the logical structure of, well, a Cargo
/// workspace. It pretty closely mirrors `cargo metadata` output.
///
/// Note that internally, rust analyzer uses a different structure:
/// `CrateGraph`. `CrateGraph` is lower-level: it knows only about the crates,
/// while this knows about `Packages` & `Targets`: purely cargo-related
/// concepts.
///
/// We use absolute paths here, `cargo metadata` guarantees to always produce
/// abs paths.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CargoWorkspace {
    packages: Arena<PackageData>,
    targets: Arena<TargetData>,
    workspace_root: AbsPathBuf,
}

impl ops::Index<Package> for CargoWorkspace {
    type Output = PackageData;
    fn index(&self, index: Package) -> &PackageData {
        &self.packages[index]
    }
}

impl ops::Index<Target> for CargoWorkspace {
    type Output = TargetData;
    fn index(&self, index: Target) -> &TargetData {
        &self.targets[index]
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct CargoConfig {
    /// Do not activate the `default` feature.
    pub no_default_features: bool,

    /// Activate all available features
    pub all_features: bool,

    /// List of features to activate.
    /// This will be ignored if `cargo_all_features` is true.
    pub features: Vec<String>,

    /// Runs cargo check on launch to figure out the correct values of OUT_DIR
    pub load_out_dirs_from_check: bool,

    /// rustc target
    pub target: Option<String>,

    /// Don't load sysroot crates (`std`, `core` & friends). Might be useful
    /// when debugging isolated issues.
    pub no_sysroot: bool,

    /// rustc private crate source
    pub rustc_source: Option<AbsPathBuf>,
}

pub type Package = Idx<PackageData>;

pub type Target = Idx<TargetData>;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageData {
    pub version: String,
    pub name: String,
    pub manifest: AbsPathBuf,
    pub targets: Vec<Target>,
    pub is_member: bool,
    pub dependencies: Vec<PackageDependency>,
    pub edition: Edition,
    pub features: Vec<String>,
    pub cfgs: Vec<CfgFlag>,
    pub envs: Vec<(String, String)>,
    pub out_dir: Option<AbsPathBuf>,
    pub proc_macro_dylib_path: Option<AbsPathBuf>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageDependency {
    pub pkg: Package,
    pub name: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TargetData {
    pub package: Package,
    pub name: String,
    pub root: AbsPathBuf,
    pub kind: TargetKind,
    pub is_proc_macro: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    Bin,
    /// Any kind of Cargo lib crate-type (dylib, rlib, proc-macro, ...).
    Lib,
    Example,
    Test,
    Bench,
    Other,
}

impl TargetKind {
    fn new(kinds: &[String]) -> TargetKind {
        for kind in kinds {
            return match kind.as_str() {
                "bin" => TargetKind::Bin,
                "test" => TargetKind::Test,
                "bench" => TargetKind::Bench,
                "example" => TargetKind::Example,
                "proc-macro" => TargetKind::Lib,
                _ if kind.contains("lib") => TargetKind::Lib,
                _ => continue,
            };
        }
        TargetKind::Other
    }
}

impl PackageData {
    pub fn root(&self) -> &AbsPath {
        self.manifest.parent().unwrap()
    }
}

impl CargoWorkspace {
    pub fn from_cargo_metadata(
        cargo_toml: &AbsPath,
        config: &CargoConfig,
    ) -> Result<CargoWorkspace> {
        let mut meta = MetadataCommand::new();
        meta.cargo_path(toolchain::cargo());
        meta.manifest_path(cargo_toml.to_path_buf());
        if config.all_features {
            meta.features(CargoOpt::AllFeatures);
        } else {
            if config.no_default_features {
                // FIXME: `NoDefaultFeatures` is mutual exclusive with `SomeFeatures`
                // https://github.com/oli-obk/cargo_metadata/issues/79
                meta.features(CargoOpt::NoDefaultFeatures);
            }
            if !config.features.is_empty() {
                meta.features(CargoOpt::SomeFeatures(config.features.clone()));
            }
        }
        if let Some(parent) = cargo_toml.parent() {
            meta.current_dir(parent.to_path_buf());
        }
        if let Some(target) = config.target.as_ref() {
            meta.other_options(vec![String::from("--filter-platform"), target.clone()]);
        }
        let mut meta = meta.exec().with_context(|| {
            format!("Failed to run `cargo metadata --manifest-path {}`", cargo_toml.display())
        })?;

        let mut out_dir_by_id = FxHashMap::default();
        let mut cfgs = FxHashMap::default();
        let mut envs = FxHashMap::default();
        let mut proc_macro_dylib_paths = FxHashMap::default();
        if config.load_out_dirs_from_check {
            let resources = load_extern_resources(cargo_toml, config)?;
            out_dir_by_id = resources.out_dirs;
            cfgs = resources.cfgs;
            envs = resources.env;
            proc_macro_dylib_paths = resources.proc_dylib_paths;
        }

        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        meta.packages.sort_by(|a, b| a.id.cmp(&b.id));
        for meta_pkg in meta.packages {
            let id = meta_pkg.id.clone();
            inject_cargo_env(&meta_pkg, envs.entry(id).or_default());

            let cargo_metadata::Package { id, edition, name, manifest_path, version, .. } =
                meta_pkg;
            let is_member = ws_members.contains(&id);
            let edition = edition
                .parse::<Edition>()
                .with_context(|| format!("Failed to parse edition {}", edition))?;
            let pkg = packages.alloc(PackageData {
                name,
                version: version.to_string(),
                manifest: AbsPathBuf::assert(manifest_path),
                targets: Vec::new(),
                is_member,
                edition,
                dependencies: Vec::new(),
                features: Vec::new(),
                cfgs: cfgs.get(&id).cloned().unwrap_or_default(),
                envs: envs.get(&id).cloned().unwrap_or_default(),
                out_dir: out_dir_by_id.get(&id).cloned(),
                proc_macro_dylib_path: proc_macro_dylib_paths.get(&id).cloned(),
            });
            let pkg_data = &mut packages[pkg];
            pkg_by_id.insert(id, pkg);
            for meta_tgt in meta_pkg.targets {
                let is_proc_macro = meta_tgt.kind.as_slice() == ["proc-macro"];
                let tgt = targets.alloc(TargetData {
                    package: pkg,
                    name: meta_tgt.name,
                    root: AbsPathBuf::assert(meta_tgt.src_path.clone()),
                    kind: TargetKind::new(meta_tgt.kind.as_slice()),
                    is_proc_macro,
                });
                pkg_data.targets.push(tgt);
            }
        }
        let resolve = meta.resolve.expect("metadata executed with deps");
        for mut node in resolve.nodes {
            let source = match pkg_by_id.get(&node.id) {
                Some(&src) => src,
                // FIXME: replace this and a similar branch below with `.unwrap`, once
                // https://github.com/rust-lang/cargo/issues/7841
                // is fixed and hits stable (around 1.43-is probably?).
                None => {
                    log::error!("Node id do not match in cargo metadata, ignoring {}", node.id);
                    continue;
                }
            };
            node.deps.sort_by(|a, b| a.pkg.cmp(&b.pkg));
            for dep_node in node.deps {
                let pkg = match pkg_by_id.get(&dep_node.pkg) {
                    Some(&pkg) => pkg,
                    None => {
                        log::error!(
                            "Dep node id do not match in cargo metadata, ignoring {}",
                            dep_node.pkg
                        );
                        continue;
                    }
                };
                let dep = PackageDependency { name: dep_node.name, pkg };
                packages[source].dependencies.push(dep);
            }
            packages[source].features.extend(node.features);
        }

        let workspace_root = AbsPathBuf::assert(meta.workspace_root);
        Ok(CargoWorkspace { packages, targets, workspace_root: workspace_root })
    }

    pub fn packages<'a>(&'a self) -> impl Iterator<Item = Package> + ExactSizeIterator + 'a {
        self.packages.iter().map(|(id, _pkg)| id)
    }

    pub fn target_by_root(&self, root: &AbsPath) -> Option<Target> {
        self.packages()
            .filter_map(|pkg| self[pkg].targets.iter().find(|&&it| &self[it].root == root))
            .next()
            .copied()
    }

    pub fn workspace_root(&self) -> &AbsPath {
        &self.workspace_root
    }

    pub fn package_flag(&self, package: &PackageData) -> String {
        if self.is_unique(&*package.name) {
            package.name.clone()
        } else {
            format!("{}:{}", package.name, package.version)
        }
    }

    fn is_unique(&self, name: &str) -> bool {
        self.packages.iter().filter(|(_, v)| v.name == name).count() == 1
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExternResources {
    out_dirs: FxHashMap<PackageId, AbsPathBuf>,
    proc_dylib_paths: FxHashMap<PackageId, AbsPathBuf>,
    cfgs: FxHashMap<PackageId, Vec<CfgFlag>>,
    env: FxHashMap<PackageId, Vec<(String, String)>>,
}

pub(crate) fn load_extern_resources(
    cargo_toml: &Path,
    cargo_features: &CargoConfig,
) -> Result<ExternResources> {
    let mut cmd = Command::new(toolchain::cargo());
    cmd.args(&["check", "--message-format=json", "--manifest-path"]).arg(cargo_toml);

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

    let output = cmd.output()?;

    let mut res = ExternResources::default();

    for message in cargo_metadata::Message::parse_stream(output.stdout.as_slice()) {
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
                    // cargo_metadata crate returns default (empty) path for
                    // older cargos, which is not absolute, so work around that.
                    if out_dir != PathBuf::default() {
                        let out_dir = AbsPathBuf::assert(out_dir);
                        res.out_dirs.insert(package_id.clone(), out_dir);
                        res.cfgs.insert(package_id.clone(), cfgs);
                    }

                    res.env.insert(package_id, env);
                }
                Message::CompilerArtifact(message) => {
                    if message.target.kind.contains(&"proc-macro".to_string()) {
                        let package_id = message.package_id;
                        // Skip rmeta file
                        if let Some(filename) = message.filenames.iter().find(|name| is_dylib(name))
                        {
                            let filename = AbsPathBuf::assert(filename.clone());
                            res.proc_dylib_paths.insert(package_id, filename);
                        }
                    }
                }
                Message::CompilerMessage(_) => (),
                Message::Unknown => (),
                Message::BuildFinished(_) => {}
                Message::TextLine(_) => {}
            }
        }
    }
    Ok(res)
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
    // CARGO, CARGO_PKG_HOMEPAGE, CARGO_CRATE_NAME, CARGO_BIN_NAME, CARGO_BIN_EXE_<name>

    let mut manifest_dir = package.manifest_path.clone();
    manifest_dir.pop();
    if let Some(cargo_manifest_dir) = manifest_dir.to_str() {
        env.push(("CARGO_MANIFEST_DIR".into(), cargo_manifest_dir.into()));
    }

    env.push(("CARGO_PKG_VERSION".into(), package.version.to_string()));
    env.push(("CARGO_PKG_VERSION_MAJOR".into(), package.version.major.to_string()));
    env.push(("CARGO_PKG_VERSION_MINOR".into(), package.version.minor.to_string()));
    env.push(("CARGO_PKG_VERSION_PATCH".into(), package.version.patch.to_string()));

    let pre = package.version.pre.iter().map(|id| id.to_string()).collect::<Vec<_>>();
    let pre = pre.join(".");
    env.push(("CARGO_PKG_VERSION_PRE".into(), pre));

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
