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
        cargo_features: &CargoConfig,
    ) -> Result<CargoWorkspace> {
        let mut meta = MetadataCommand::new();
        meta.cargo_path(toolchain::cargo());
        meta.manifest_path(cargo_toml.to_path_buf());
        if cargo_features.all_features {
            meta.features(CargoOpt::AllFeatures);
        } else {
            if cargo_features.no_default_features {
                // FIXME: `NoDefaultFeatures` is mutual exclusive with `SomeFeatures`
                // https://github.com/oli-obk/cargo_metadata/issues/79
                meta.features(CargoOpt::NoDefaultFeatures);
            }
            if !cargo_features.features.is_empty() {
                meta.features(CargoOpt::SomeFeatures(cargo_features.features.clone()));
            }
        }
        if let Some(parent) = cargo_toml.parent() {
            meta.current_dir(parent.to_path_buf());
        }
        if let Some(target) = cargo_features.target.as_ref() {
            meta.other_options(vec![String::from("--filter-platform"), target.clone()]);
        }
        let mut meta = meta.exec().with_context(|| {
            format!("Failed to run `cargo metadata --manifest-path {}`", cargo_toml.display())
        })?;

        let mut out_dir_by_id = FxHashMap::default();
        let mut cfgs = FxHashMap::default();
        let mut proc_macro_dylib_paths = FxHashMap::default();
        if cargo_features.load_out_dirs_from_check {
            let resources = load_extern_resources(cargo_toml, cargo_features)?;
            out_dir_by_id = resources.out_dirs;
            cfgs = resources.cfgs;
            proc_macro_dylib_paths = resources.proc_dylib_paths;
        }

        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        meta.packages.sort_by(|a, b| a.id.cmp(&b.id));
        for meta_pkg in meta.packages {
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
pub struct ExternResources {
    out_dirs: FxHashMap<PackageId, AbsPathBuf>,
    proc_dylib_paths: FxHashMap<PackageId, AbsPathBuf>,
    cfgs: FxHashMap<PackageId, Vec<CfgFlag>>,
}

pub fn load_extern_resources(
    cargo_toml: &Path,
    cargo_features: &CargoConfig,
) -> Result<ExternResources> {
    let mut cmd = Command::new(toolchain::cargo());
    cmd.args(&["check", "--message-format=json", "--manifest-path"]).arg(cargo_toml);
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
                Message::BuildScriptExecuted(BuildScript { package_id, out_dir, cfgs, .. }) => {
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
                        res.cfgs.insert(package_id, cfgs);
                    }
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
