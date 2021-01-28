//! FIXME: write short doc here

use std::{convert::TryInto, ops, process::Command, sync::Arc};

use anyhow::{Context, Result};
use base_db::Edition;
use cargo_metadata::{CargoOpt, MetadataCommand};
use la_arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;

use crate::build_data::BuildDataConfig;
use crate::utf8_stdout;

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
    build_data_config: BuildDataConfig,
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

/// Information associated with a cargo crate
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageData {
    /// Version given in the `Cargo.toml`
    pub version: String,
    /// Name as given in the `Cargo.toml`
    pub name: String,
    /// Path containing the `Cargo.toml`
    pub manifest: AbsPathBuf,
    /// Targets provided by the crate (lib, bin, example, test, ...)
    pub targets: Vec<Target>,
    /// Is this package a member of the current workspace
    pub is_member: bool,
    /// List of packages this package depends on
    pub dependencies: Vec<PackageDependency>,
    /// Rust edition for this package
    pub edition: Edition,
    /// Features provided by the crate, mapped to the features required by that feature.
    pub features: FxHashMap<String, Vec<String>>,
    /// List of features enabled on this package
    pub active_features: Vec<String>,
    // String representation of package id
    pub id: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageDependency {
    pub pkg: Package,
    pub name: String,
}

/// Information associated with a package's target
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TargetData {
    /// Package that provided this target
    pub package: Package,
    /// Name as given in the `Cargo.toml` or generated from the file name
    pub name: String,
    /// Path to the main source file of the target
    pub root: AbsPathBuf,
    /// Kind of target
    pub kind: TargetKind,
    /// Is this target a proc-macro
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
        progress: &dyn Fn(String),
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
        let target = if let Some(target) = config.target.as_ref() {
            Some(target.clone())
        } else {
            // cargo metadata defaults to giving information for _all_ targets.
            // In the absence of a preference from the user, we use the host platform.
            let mut rustc = Command::new(toolchain::rustc());
            rustc.current_dir(cargo_toml.parent().unwrap()).arg("-vV");
            log::debug!("Discovering host platform by {:?}", rustc);
            match utf8_stdout(rustc) {
                Ok(stdout) => {
                    let field = "host: ";
                    let target = stdout.lines().find_map(|l| l.strip_prefix(field));
                    if let Some(target) = target {
                        Some(target.to_string())
                    } else {
                        // If we fail to resolve the host platform, it's not the end of the world.
                        log::info!("rustc -vV did not report host platform, got:\n{}", stdout);
                        None
                    }
                }
                Err(e) => {
                    log::warn!("Failed to discover host platform: {}", e);
                    None
                }
            }
        };
        if let Some(target) = target {
            meta.other_options(vec![String::from("--filter-platform"), target]);
        }

        // FIXME: Currently MetadataCommand is not based on parse_stream,
        // So we just report it as a whole
        progress("metadata".to_string());
        let mut meta = meta.exec().with_context(|| {
            let cwd: Option<AbsPathBuf> =
                std::env::current_dir().ok().and_then(|p| p.try_into().ok());

            let workdir = cargo_toml
                .parent()
                .map(|p| p.to_path_buf())
                .or(cwd)
                .map(|dir| dir.to_string_lossy().to_string())
                .unwrap_or_else(|| "<failed to get path>".into());

            format!(
                "Failed to run `cargo metadata --manifest-path {}` in `{}`",
                cargo_toml.display(),
                workdir
            )
        })?;

        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        meta.packages.sort_by(|a, b| a.id.cmp(&b.id));
        for meta_pkg in &meta.packages {
            let cargo_metadata::Package { id, edition, name, manifest_path, version, .. } =
                meta_pkg;
            let is_member = ws_members.contains(&id);
            let edition = edition
                .parse::<Edition>()
                .with_context(|| format!("Failed to parse edition {}", edition))?;
            let pkg = packages.alloc(PackageData {
                id: id.repr.clone(),
                name: name.clone(),
                version: version.to_string(),
                manifest: AbsPathBuf::assert(manifest_path.clone()),
                targets: Vec::new(),
                is_member,
                edition,
                dependencies: Vec::new(),
                features: meta_pkg.features.clone().into_iter().collect(),
                active_features: Vec::new(),
            });
            let pkg_data = &mut packages[pkg];
            pkg_by_id.insert(id, pkg);
            for meta_tgt in &meta_pkg.targets {
                let is_proc_macro = meta_tgt.kind.as_slice() == ["proc-macro"];
                let tgt = targets.alloc(TargetData {
                    package: pkg,
                    name: meta_tgt.name.clone(),
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
            packages[source].active_features.extend(node.features);
        }

        let workspace_root = AbsPathBuf::assert(meta.workspace_root);
        let build_data_config = BuildDataConfig::new(
            cargo_toml.to_path_buf(),
            config.clone(),
            Arc::new(meta.packages.clone()),
        );

        Ok(CargoWorkspace { packages, targets, workspace_root, build_data_config })
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

    pub(crate) fn build_data_config(&self) -> &BuildDataConfig {
        &self.build_data_config
    }

    fn is_unique(&self, name: &str) -> bool {
        self.packages.iter().filter(|(_, v)| v.name == name).count() == 1
    }
}
