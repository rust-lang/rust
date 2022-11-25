//! See [`CargoWorkspace`].

use std::iter;
use std::path::PathBuf;
use std::str::from_utf8;
use std::{ops, process::Command};

use anyhow::{Context, Result};
use base_db::Edition;
use cargo_metadata::{CargoOpt, MetadataCommand};
use la_arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf};
use rustc_hash::FxHashMap;
use serde::Deserialize;
use serde_json::from_value;

use crate::CfgOverrides;
use crate::{utf8_stdout, ManifestPath};

/// [`CargoWorkspace`] represents the logical structure of, well, a Cargo
/// workspace. It pretty closely mirrors `cargo metadata` output.
///
/// Note that internally, rust-analyzer uses a different structure:
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

/// Describes how to set the rustc source directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RustcSource {
    /// Explicit path for the rustc source directory.
    Path(AbsPathBuf),
    /// Try to automatically detect where the rustc source directory is.
    Discover,
}

/// Crates to disable `#[cfg(test)]` on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnsetTestCrates {
    None,
    Only(Vec<String>),
    All,
}

impl Default for UnsetTestCrates {
    fn default() -> Self {
        Self::None
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
    pub rustc_source: Option<RustcSource>,

    /// crates to disable `#[cfg(test)]` on
    pub unset_test_crates: UnsetTestCrates,

    pub wrap_rustc_in_build_scripts: bool,

    pub run_build_script_command: Option<Vec<String>>,

    pub extra_env: FxHashMap<String, String>,
}

impl CargoConfig {
    pub fn cfg_overrides(&self) -> CfgOverrides {
        match &self.unset_test_crates {
            UnsetTestCrates::None => CfgOverrides::Selective(iter::empty().collect()),
            UnsetTestCrates::Only(unset_test_crates) => CfgOverrides::Selective(
                unset_test_crates
                    .iter()
                    .cloned()
                    .zip(iter::repeat_with(|| {
                        cfg::CfgDiff::new(Vec::new(), vec![cfg::CfgAtom::Flag("test".into())])
                            .unwrap()
                    }))
                    .collect(),
            ),
            UnsetTestCrates::All => CfgOverrides::Wildcard(
                cfg::CfgDiff::new(Vec::new(), vec![cfg::CfgAtom::Flag("test".into())]).unwrap(),
            ),
        }
    }
}

pub type Package = Idx<PackageData>;

pub type Target = Idx<TargetData>;

/// Information associated with a cargo crate
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageData {
    /// Version given in the `Cargo.toml`
    pub version: semver::Version,
    /// Name as given in the `Cargo.toml`
    pub name: String,
    /// Repository as given in the `Cargo.toml`
    pub repository: Option<String>,
    /// Path containing the `Cargo.toml`
    pub manifest: ManifestPath,
    /// Targets provided by the crate (lib, bin, example, test, ...)
    pub targets: Vec<Target>,
    /// Does this package come from the local filesystem (and is editable)?
    pub is_local: bool,
    // Whether this package is a member of the workspace
    pub is_member: bool,
    /// List of packages this package depends on
    pub dependencies: Vec<PackageDependency>,
    /// Rust edition for this package
    pub edition: Edition,
    /// Features provided by the crate, mapped to the features required by that feature.
    pub features: FxHashMap<String, Vec<String>>,
    /// List of features enabled on this package
    pub active_features: Vec<String>,
    /// String representation of package id
    pub id: String,
    /// The contents of [package.metadata.rust-analyzer]
    pub metadata: RustAnalyzerPackageMetaData,
}

#[derive(Deserialize, Default, Debug, Clone, Eq, PartialEq)]
pub struct RustAnalyzerPackageMetaData {
    pub rustc_private: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PackageDependency {
    pub pkg: Package,
    pub name: String,
    pub kind: DepKind,
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub enum DepKind {
    /// Available to the library, binary, and dev targets in the package (but not the build script).
    Normal,
    /// Available only to test and bench targets (and the library target, when built with `cfg(test)`).
    Dev,
    /// Available only to the build script target.
    Build,
}

impl DepKind {
    fn iter(list: &[cargo_metadata::DepKindInfo]) -> impl Iterator<Item = Self> + '_ {
        let mut dep_kinds = Vec::new();
        if list.is_empty() {
            dep_kinds.push(Self::Normal);
        }
        for info in list {
            let kind = match info.kind {
                cargo_metadata::DependencyKind::Normal => Self::Normal,
                cargo_metadata::DependencyKind::Development => Self::Dev,
                cargo_metadata::DependencyKind::Build => Self::Build,
                cargo_metadata::DependencyKind::Unknown => continue,
            };
            dep_kinds.push(kind);
        }
        dep_kinds.sort_unstable();
        dep_kinds.dedup();
        dep_kinds.into_iter()
    }
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
    /// Required features of the target without which it won't build
    pub required_features: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    Bin,
    /// Any kind of Cargo lib crate-type (dylib, rlib, proc-macro, ...).
    Lib,
    Example,
    Test,
    Bench,
    BuildScript,
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
                "custom-build" => TargetKind::BuildScript,
                "proc-macro" => TargetKind::Lib,
                _ if kind.contains("lib") => TargetKind::Lib,
                _ => continue,
            };
        }
        TargetKind::Other
    }
}

#[derive(Deserialize, Default)]
// Deserialise helper for the cargo metadata
struct PackageMetadata {
    #[serde(rename = "rust-analyzer")]
    rust_analyzer: Option<RustAnalyzerPackageMetaData>,
}

impl CargoWorkspace {
    pub fn fetch_metadata(
        cargo_toml: &ManifestPath,
        current_dir: &AbsPath,
        config: &CargoConfig,
        progress: &dyn Fn(String),
    ) -> Result<cargo_metadata::Metadata> {
        let target = config
            .target
            .clone()
            .or_else(|| cargo_config_build_target(cargo_toml, config))
            .or_else(|| rustc_discover_host_triple(cargo_toml, config));

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
        meta.current_dir(current_dir.as_os_str());

        if let Some(target) = target {
            meta.other_options(vec![String::from("--filter-platform"), target]);
        }

        // FIXME: Fetching metadata is a slow process, as it might require
        // calling crates.io. We should be reporting progress here, but it's
        // unclear whether cargo itself supports it.
        progress("metadata".to_string());

        fn exec_with_env(
            command: &cargo_metadata::MetadataCommand,
            extra_env: &FxHashMap<String, String>,
        ) -> Result<cargo_metadata::Metadata, cargo_metadata::Error> {
            let mut command = command.cargo_command();
            command.envs(extra_env);
            let output = command.output()?;
            if !output.status.success() {
                return Err(cargo_metadata::Error::CargoMetadata {
                    stderr: String::from_utf8(output.stderr)?,
                });
            }
            let stdout = from_utf8(&output.stdout)?
                .lines()
                .find(|line| line.starts_with('{'))
                .ok_or(cargo_metadata::Error::NoJson)?;
            cargo_metadata::MetadataCommand::parse(stdout)
        }

        let meta = exec_with_env(&meta, &config.extra_env)
            .with_context(|| format!("Failed to run `{:?}`", meta.cargo_command()))?;

        Ok(meta)
    }

    pub fn new(mut meta: cargo_metadata::Metadata) -> CargoWorkspace {
        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        meta.packages.sort_by(|a, b| a.id.cmp(&b.id));
        for meta_pkg in &meta.packages {
            let cargo_metadata::Package {
                id,
                edition,
                name,
                manifest_path,
                version,
                metadata,
                repository,
                ..
            } = meta_pkg;
            let meta = from_value::<PackageMetadata>(metadata.clone()).unwrap_or_default();
            let edition = match edition {
                cargo_metadata::Edition::E2015 => Edition::Edition2015,
                cargo_metadata::Edition::E2018 => Edition::Edition2018,
                cargo_metadata::Edition::E2021 => Edition::Edition2021,
                _ => {
                    tracing::error!("Unsupported edition `{:?}`", edition);
                    Edition::CURRENT
                }
            };
            // We treat packages without source as "local" packages. That includes all members of
            // the current workspace, as well as any path dependency outside the workspace.
            let is_local = meta_pkg.source.is_none();
            let is_member = ws_members.contains(id);

            let pkg = packages.alloc(PackageData {
                id: id.repr.clone(),
                name: name.clone(),
                version: version.clone(),
                manifest: AbsPathBuf::assert(PathBuf::from(&manifest_path)).try_into().unwrap(),
                targets: Vec::new(),
                is_local,
                is_member,
                edition,
                repository: repository.clone(),
                dependencies: Vec::new(),
                features: meta_pkg.features.clone().into_iter().collect(),
                active_features: Vec::new(),
                metadata: meta.rust_analyzer.unwrap_or_default(),
            });
            let pkg_data = &mut packages[pkg];
            pkg_by_id.insert(id, pkg);
            for meta_tgt in &meta_pkg.targets {
                let is_proc_macro = meta_tgt.kind.as_slice() == ["proc-macro"];
                let tgt = targets.alloc(TargetData {
                    package: pkg,
                    name: meta_tgt.name.clone(),
                    root: AbsPathBuf::assert(PathBuf::from(&meta_tgt.src_path)),
                    kind: TargetKind::new(meta_tgt.kind.as_slice()),
                    is_proc_macro,
                    required_features: meta_tgt.required_features.clone(),
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
                    tracing::error!("Node id do not match in cargo metadata, ignoring {}", node.id);
                    continue;
                }
            };
            node.deps.sort_by(|a, b| a.pkg.cmp(&b.pkg));
            for (dep_node, kind) in node
                .deps
                .iter()
                .flat_map(|dep| DepKind::iter(&dep.dep_kinds).map(move |kind| (dep, kind)))
            {
                let pkg = match pkg_by_id.get(&dep_node.pkg) {
                    Some(&pkg) => pkg,
                    None => {
                        tracing::error!(
                            "Dep node id do not match in cargo metadata, ignoring {}",
                            dep_node.pkg
                        );
                        continue;
                    }
                };
                let dep = PackageDependency { name: dep_node.name.clone(), pkg, kind };
                packages[source].dependencies.push(dep);
            }
            packages[source].active_features.extend(node.features);
        }

        let workspace_root =
            AbsPathBuf::assert(PathBuf::from(meta.workspace_root.into_os_string()));

        CargoWorkspace { packages, targets, workspace_root }
    }

    pub fn packages<'a>(&'a self) -> impl Iterator<Item = Package> + ExactSizeIterator + 'a {
        self.packages.iter().map(|(id, _pkg)| id)
    }

    pub fn target_by_root(&self, root: &AbsPath) -> Option<Target> {
        self.packages()
            .filter(|&pkg| self[pkg].is_member)
            .find_map(|pkg| self[pkg].targets.iter().find(|&&it| &self[it].root == root))
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

    pub fn parent_manifests(&self, manifest_path: &ManifestPath) -> Option<Vec<ManifestPath>> {
        let mut found = false;
        let parent_manifests = self
            .packages()
            .filter_map(|pkg| {
                if !found && &self[pkg].manifest == manifest_path {
                    found = true
                }
                self[pkg].dependencies.iter().find_map(|dep| {
                    if &self[dep.pkg].manifest == manifest_path {
                        return Some(self[pkg].manifest.clone());
                    }
                    None
                })
            })
            .collect::<Vec<ManifestPath>>();

        // some packages has this pkg as dep. return their manifests
        if parent_manifests.len() > 0 {
            return Some(parent_manifests);
        }

        // this pkg is inside this cargo workspace, fallback to workspace root
        if found {
            return Some(vec![
                ManifestPath::try_from(self.workspace_root().join("Cargo.toml")).ok()?
            ]);
        }

        // not in this workspace
        None
    }

    fn is_unique(&self, name: &str) -> bool {
        self.packages.iter().filter(|(_, v)| v.name == name).count() == 1
    }
}

fn rustc_discover_host_triple(cargo_toml: &ManifestPath, config: &CargoConfig) -> Option<String> {
    let mut rustc = Command::new(toolchain::rustc());
    rustc.envs(&config.extra_env);
    rustc.current_dir(cargo_toml.parent()).arg("-vV");
    tracing::debug!("Discovering host platform by {:?}", rustc);
    match utf8_stdout(rustc) {
        Ok(stdout) => {
            let field = "host: ";
            let target = stdout.lines().find_map(|l| l.strip_prefix(field));
            if let Some(target) = target {
                Some(target.to_string())
            } else {
                // If we fail to resolve the host platform, it's not the end of the world.
                tracing::info!("rustc -vV did not report host platform, got:\n{}", stdout);
                None
            }
        }
        Err(e) => {
            tracing::warn!("Failed to discover host platform: {}", e);
            None
        }
    }
}

fn cargo_config_build_target(cargo_toml: &ManifestPath, config: &CargoConfig) -> Option<String> {
    let mut cargo_config = Command::new(toolchain::cargo());
    cargo_config.envs(&config.extra_env);
    cargo_config
        .current_dir(cargo_toml.parent())
        .args(&["-Z", "unstable-options", "config", "get", "build.target"])
        .env("RUSTC_BOOTSTRAP", "1");
    // if successful we receive `build.target = "target-triple"`
    tracing::debug!("Discovering cargo config target by {:?}", cargo_config);
    match utf8_stdout(cargo_config) {
        Ok(stdout) => stdout
            .strip_prefix("build.target = \"")
            .and_then(|stdout| stdout.strip_suffix('"'))
            .map(ToOwned::to_owned),
        Err(_) => None,
    }
}
