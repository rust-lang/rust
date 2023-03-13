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

use crate::{utf8_stdout, InvocationLocation, ManifestPath};
use crate::{CfgOverrides, InvocationStrategy};

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CargoFeatures {
    All,
    Selected {
        /// List of features to activate.
        features: Vec<String>,
        /// Do not activate the `default` feature.
        no_default_features: bool,
    },
}

impl Default for CargoFeatures {
    fn default() -> Self {
        CargoFeatures::Selected { features: vec![], no_default_features: false }
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct CargoConfig {
    /// List of features to activate.
    pub features: CargoFeatures,
    /// rustc target
    pub target: Option<String>,
    /// Sysroot loading behavior
    pub sysroot: Option<RustcSource>,
    pub sysroot_src: Option<AbsPathBuf>,
    /// rustc private crate source
    pub rustc_source: Option<RustcSource>,
    /// crates to disable `#[cfg(test)]` on
    pub unset_test_crates: UnsetTestCrates,
    /// Invoke `cargo check` through the RUSTC_WRAPPER.
    pub wrap_rustc_in_build_scripts: bool,
    /// The command to run instead of `cargo check` for building build scripts.
    pub run_build_script_command: Option<Vec<String>>,
    /// Extra args to pass to the cargo command.
    pub extra_args: Vec<String>,
    /// Extra env vars to set when invoking the cargo command
    pub extra_env: FxHashMap<String, String>,
    pub invocation_strategy: InvocationStrategy,
    pub invocation_location: InvocationLocation,
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
    /// Whether this package is a member of the workspace
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

// Deserialize helper for the cargo metadata
#[derive(Deserialize, Default)]
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
        let targets = find_list_of_build_targets(config, cargo_toml);

        let mut meta = MetadataCommand::new();
        meta.cargo_path(toolchain::cargo());
        meta.manifest_path(cargo_toml.to_path_buf());
        match &config.features {
            CargoFeatures::All => {
                meta.features(CargoOpt::AllFeatures);
            }
            CargoFeatures::Selected { features, no_default_features } => {
                if *no_default_features {
                    meta.features(CargoOpt::NoDefaultFeatures);
                }
                if !features.is_empty() {
                    meta.features(CargoOpt::SomeFeatures(features.clone()));
                }
            }
        }
        meta.current_dir(current_dir.as_os_str());

        if !targets.is_empty() {
            let other_options: Vec<_> = targets
                .into_iter()
                .flat_map(|target| ["--filter-platform".to_string(), target])
                .collect();
            meta.other_options(other_options);
        }

        // FIXME: Fetching metadata is a slow process, as it might require
        // calling crates.io. We should be reporting progress here, but it's
        // unclear whether cargo itself supports it.
        progress("metadata".to_string());

        (|| -> Result<cargo_metadata::Metadata, cargo_metadata::Error> {
            let mut command = meta.cargo_command();
            command.envs(&config.extra_env);
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
        })()
        .with_context(|| format!("Failed to run `{:?}`", meta.cargo_command()))
    }

    pub fn new(mut meta: cargo_metadata::Metadata) -> CargoWorkspace {
        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        meta.packages.sort_by(|a, b| a.id.cmp(&b.id));
        for meta_pkg in meta.packages {
            let cargo_metadata::Package {
                name,
                version,
                id,
                source,
                targets: meta_targets,
                features,
                manifest_path,
                repository,
                edition,
                metadata,
                ..
            } = meta_pkg;
            let meta = from_value::<PackageMetadata>(metadata).unwrap_or_default();
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
            let is_local = source.is_none();
            let is_member = ws_members.contains(&id);

            let pkg = packages.alloc(PackageData {
                id: id.repr.clone(),
                name,
                version,
                manifest: AbsPathBuf::assert(manifest_path.into()).try_into().unwrap(),
                targets: Vec::new(),
                is_local,
                is_member,
                edition,
                repository,
                dependencies: Vec::new(),
                features: features.into_iter().collect(),
                active_features: Vec::new(),
                metadata: meta.rust_analyzer.unwrap_or_default(),
            });
            let pkg_data = &mut packages[pkg];
            pkg_by_id.insert(id, pkg);
            for meta_tgt in meta_targets {
                let cargo_metadata::Target { name, kind, required_features, src_path, .. } =
                    meta_tgt;
                let tgt = targets.alloc(TargetData {
                    package: pkg,
                    name,
                    root: AbsPathBuf::assert(src_path.into()),
                    kind: TargetKind::new(&kind),
                    is_proc_macro: &*kind == ["proc-macro"],
                    required_features,
                });
                pkg_data.targets.push(tgt);
            }
        }
        let resolve = meta.resolve.expect("metadata executed with deps");
        for mut node in resolve.nodes {
            let &source = pkg_by_id.get(&node.id).unwrap();
            node.deps.sort_by(|a, b| a.pkg.cmp(&b.pkg));
            let dependencies = node
                .deps
                .iter()
                .flat_map(|dep| DepKind::iter(&dep.dep_kinds).map(move |kind| (dep, kind)));
            for (dep_node, kind) in dependencies {
                let &pkg = pkg_by_id.get(&dep_node.pkg).unwrap();
                let dep = PackageDependency { name: dep_node.name.clone(), pkg, kind };
                packages[source].dependencies.push(dep);
            }
            packages[source].active_features.extend(node.features);
        }

        let workspace_root =
            AbsPathBuf::assert(PathBuf::from(meta.workspace_root.into_os_string()));

        CargoWorkspace { packages, targets, workspace_root }
    }

    pub fn packages(&self) -> impl Iterator<Item = Package> + ExactSizeIterator + '_ {
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
        if self.is_unique(&package.name) {
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
                    (&self[dep.pkg].manifest == manifest_path).then(|| self[pkg].manifest.clone())
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

fn find_list_of_build_targets(config: &CargoConfig, cargo_toml: &ManifestPath) -> Vec<String> {
    if let Some(target) = &config.target {
        return [target.into()].to_vec();
    }

    let build_targets = cargo_config_build_target(cargo_toml, &config.extra_env);
    if !build_targets.is_empty() {
        return build_targets;
    }

    rustc_discover_host_triple(cargo_toml, &config.extra_env).into_iter().collect()
}

fn rustc_discover_host_triple(
    cargo_toml: &ManifestPath,
    extra_env: &FxHashMap<String, String>,
) -> Option<String> {
    let mut rustc = Command::new(toolchain::rustc());
    rustc.envs(extra_env);
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

fn cargo_config_build_target(
    cargo_toml: &ManifestPath,
    extra_env: &FxHashMap<String, String>,
) -> Vec<String> {
    let mut cargo_config = Command::new(toolchain::cargo());
    cargo_config.envs(extra_env);
    cargo_config
        .current_dir(cargo_toml.parent())
        .args(["-Z", "unstable-options", "config", "get", "build.target"])
        .env("RUSTC_BOOTSTRAP", "1");
    // if successful we receive `build.target = "target-triple"`
    // or `build.target = ["<target 1>", ..]`
    tracing::debug!("Discovering cargo config target by {:?}", cargo_config);
    utf8_stdout(cargo_config).map(parse_output_cargo_config_build_target).unwrap_or_default()
}

fn parse_output_cargo_config_build_target(stdout: String) -> Vec<String> {
    let trimmed = stdout.trim_start_matches("build.target = ").trim_matches('"');

    if !trimmed.starts_with('[') {
        return [trimmed.to_string()].to_vec();
    }

    let res = serde_json::from_str(trimmed);
    if let Err(e) = &res {
        tracing::warn!("Failed to parse `build.target` as an array of target: {}`", e);
    }
    res.unwrap_or_default()
}
