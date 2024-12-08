//! See [`CargoWorkspace`].

use std::ops;
use std::str::from_utf8;

use anyhow::Context;
use cargo_metadata::{CargoOpt, MetadataCommand};
use la_arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf, Utf8PathBuf};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;
use serde_json::from_value;
use span::Edition;
use toolchain::Tool;

use crate::{utf8_stdout, ManifestPath, Sysroot, SysrootQueryMetadata};
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
    target_directory: AbsPathBuf,
    manifest_path: ManifestPath,
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
pub enum RustLibSource {
    /// Explicit path for the rustc source directory.
    Path(AbsPathBuf),
    /// Try to automatically detect where the rustc source directory is.
    Discover,
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
    /// Whether to pass `--all-targets` to cargo invocations.
    pub all_targets: bool,
    /// List of features to activate.
    pub features: CargoFeatures,
    /// rustc target
    pub target: Option<String>,
    /// Sysroot loading behavior
    pub sysroot: Option<RustLibSource>,
    /// How to query metadata for the sysroot crate.
    pub sysroot_query_metadata: SysrootQueryMetadata,
    pub sysroot_src: Option<AbsPathBuf>,
    /// rustc private crate source
    pub rustc_source: Option<RustLibSource>,
    pub cfg_overrides: CfgOverrides,
    /// Invoke `cargo check` through the RUSTC_WRAPPER.
    pub wrap_rustc_in_build_scripts: bool,
    /// The command to run instead of `cargo check` for building build scripts.
    pub run_build_script_command: Option<Vec<String>>,
    /// Extra args to pass to the cargo command.
    pub extra_args: Vec<String>,
    /// Extra env vars to set when invoking the cargo command
    pub extra_env: FxHashMap<String, String>,
    pub invocation_strategy: InvocationStrategy,
    /// Optional path to use instead of `target` when building
    pub target_dir: Option<Utf8PathBuf>,
    pub set_test: bool,
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
    /// Authors as given in the `Cargo.toml`
    pub authors: Vec<String>,
    /// Description as given in the `Cargo.toml`
    pub description: Option<String>,
    /// Homepage as given in the `Cargo.toml`
    pub homepage: Option<String>,
    /// License as given in the `Cargo.toml`
    pub license: Option<String>,
    /// License file as given in the `Cargo.toml`
    pub license_file: Option<Utf8PathBuf>,
    /// Readme file as given in the `Cargo.toml`
    pub readme: Option<Utf8PathBuf>,
    /// Rust version as given in the `Cargo.toml`
    pub rust_version: Option<semver::Version>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    /// Available to the library, binary, and dev targets in the package (but not the build script).
    Normal,
    /// Available only to test and bench targets (and the library target, when built with `cfg(test)`).
    Dev,
    /// Available only to the build script target.
    Build,
}

impl DepKind {
    fn iter(list: &[cargo_metadata::DepKindInfo]) -> impl Iterator<Item = Self> {
        let mut dep_kinds = [None; 3];
        if list.is_empty() {
            dep_kinds[0] = Some(Self::Normal);
        }
        for info in list {
            match info.kind {
                cargo_metadata::DependencyKind::Normal => dep_kinds[0] = Some(Self::Normal),
                cargo_metadata::DependencyKind::Development => dep_kinds[1] = Some(Self::Dev),
                cargo_metadata::DependencyKind::Build => dep_kinds[2] = Some(Self::Build),
                cargo_metadata::DependencyKind::Unknown => continue,
            }
        }
        dep_kinds.into_iter().flatten()
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
    /// Required features of the target without which it won't build
    pub required_features: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    Bin,
    /// Any kind of Cargo lib crate-type (dylib, rlib, proc-macro, ...).
    Lib {
        /// Is this target a proc-macro
        is_proc_macro: bool,
    },
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
                "proc-macro" => TargetKind::Lib { is_proc_macro: true },
                _ if kind.contains("lib") => TargetKind::Lib { is_proc_macro: false },
                _ => continue,
            };
        }
        TargetKind::Other
    }

    pub fn is_executable(self) -> bool {
        matches!(self, TargetKind::Bin | TargetKind::Example)
    }

    pub fn is_proc_macro(self) -> bool {
        matches!(self, TargetKind::Lib { is_proc_macro: true })
    }
}

// Deserialize helper for the cargo metadata
#[derive(Deserialize, Default)]
struct PackageMetadata {
    #[serde(rename = "rust-analyzer")]
    rust_analyzer: Option<RustAnalyzerPackageMetaData>,
}

impl CargoWorkspace {
    /// Fetches the metadata for the given `cargo_toml` manifest.
    /// A successful result may contain another metadata error if the initial fetching failed but
    /// the `--no-deps` retry succeeded.
    pub fn fetch_metadata(
        cargo_toml: &ManifestPath,
        current_dir: &AbsPath,
        config: &CargoConfig,
        sysroot: &Sysroot,
        locked: bool,
        progress: &dyn Fn(String),
    ) -> anyhow::Result<(cargo_metadata::Metadata, Option<anyhow::Error>)> {
        Self::fetch_metadata_(cargo_toml, current_dir, config, sysroot, locked, false, progress)
    }

    fn fetch_metadata_(
        cargo_toml: &ManifestPath,
        current_dir: &AbsPath,
        config: &CargoConfig,
        sysroot: &Sysroot,
        locked: bool,
        no_deps: bool,
        progress: &dyn Fn(String),
    ) -> anyhow::Result<(cargo_metadata::Metadata, Option<anyhow::Error>)> {
        let targets = find_list_of_build_targets(config, cargo_toml, sysroot);

        let cargo = sysroot.tool(Tool::Cargo);
        let mut meta = MetadataCommand::new();
        meta.cargo_path(cargo.get_program());
        cargo.get_envs().for_each(|(var, val)| _ = meta.env(var, val.unwrap_or_default()));
        config.extra_env.iter().for_each(|(var, val)| _ = meta.env(var, val));
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
        meta.current_dir(current_dir);

        let mut other_options = vec![];
        // cargo metadata only supports a subset of flags of what cargo usually accepts, and usually
        // the only relevant flags for metadata here are unstable ones, so we pass those along
        // but nothing else
        let mut extra_args = config.extra_args.iter();
        while let Some(arg) = extra_args.next() {
            if arg == "-Z" {
                if let Some(arg) = extra_args.next() {
                    other_options.push("-Z".to_owned());
                    other_options.push(arg.to_owned());
                }
            }
        }

        if !targets.is_empty() {
            other_options.append(
                &mut targets
                    .into_iter()
                    .flat_map(|target| ["--filter-platform".to_owned(), target])
                    .collect(),
            );
        }
        // The manifest is a rust file, so this means its a script manifest
        if cargo_toml.is_rust_manifest() {
            // Deliberately don't set up RUSTC_BOOTSTRAP or a nightly override here, the user should
            // opt into it themselves.
            other_options.push("-Zscript".to_owned());
        }
        if locked {
            other_options.push("--locked".to_owned());
        }
        if no_deps {
            other_options.push("--no-deps".to_owned());
        }
        meta.other_options(other_options);

        // FIXME: Fetching metadata is a slow process, as it might require
        // calling crates.io. We should be reporting progress here, but it's
        // unclear whether cargo itself supports it.
        progress("metadata".to_owned());

        (|| -> anyhow::Result<(_, _)> {
            let output = meta.cargo_command().output()?;
            if !output.status.success() {
                let error = cargo_metadata::Error::CargoMetadata {
                    stderr: String::from_utf8(output.stderr)?,
                }
                .into();
                if !no_deps {
                    // If we failed to fetch metadata with deps, try again without them.
                    // This makes r-a still work partially when offline.
                    if let Ok((metadata, _)) = Self::fetch_metadata_(
                        cargo_toml,
                        current_dir,
                        config,
                        sysroot,
                        locked,
                        true,
                        progress,
                    ) {
                        return Ok((metadata, Some(error)));
                    }
                }
                return Err(error);
            }
            let stdout = from_utf8(&output.stdout)?
                .lines()
                .find(|line| line.starts_with('{'))
                .ok_or(cargo_metadata::Error::NoJson)?;
            Ok((cargo_metadata::MetadataCommand::parse(stdout)?, None))
        })()
        .map(|(metadata, error)| {
            (
                metadata,
                error.map(|e| e.context(format!("Failed to run `{:?}`", meta.cargo_command()))),
            )
        })
        .with_context(|| format!("Failed to run `{:?}`", meta.cargo_command()))
    }

    pub fn new(mut meta: cargo_metadata::Metadata, manifest_path: ManifestPath) -> CargoWorkspace {
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
                authors,
                description,
                homepage,
                license,
                license_file,
                readme,
                rust_version,
                ..
            } = meta_pkg;
            let meta = from_value::<PackageMetadata>(metadata).unwrap_or_default();
            let edition = match edition {
                cargo_metadata::Edition::E2015 => Edition::Edition2015,
                cargo_metadata::Edition::E2018 => Edition::Edition2018,
                cargo_metadata::Edition::E2021 => Edition::Edition2021,
                cargo_metadata::Edition::_E2024 => Edition::Edition2024,
                _ => {
                    tracing::error!("Unsupported edition `{:?}`", edition);
                    Edition::CURRENT
                }
            };
            // We treat packages without source as "local" packages. That includes all members of
            // the current workspace, as well as any path dependency outside the workspace.
            let is_local = source.is_none();
            let is_member = ws_members.contains(&id);

            let manifest = AbsPathBuf::assert(manifest_path);
            let pkg = packages.alloc(PackageData {
                id: id.repr.clone(),
                name,
                version,
                manifest: manifest.clone().try_into().unwrap(),
                targets: Vec::new(),
                is_local,
                is_member,
                edition,
                repository,
                authors,
                description,
                homepage,
                license,
                license_file,
                readme,
                rust_version,
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
                let kind = TargetKind::new(&kind);
                let tgt = targets.alloc(TargetData {
                    package: pkg,
                    name,
                    root: if kind == TargetKind::Bin
                        && manifest.extension().is_some_and(|ext| ext == "rs")
                    {
                        // cargo strips the script part of a cargo script away and places the
                        // modified manifest file into a special target dir which is then used as
                        // the source path. We don't want that, we want the original here so map it
                        // back
                        manifest.clone()
                    } else {
                        AbsPathBuf::assert(src_path)
                    },
                    kind,
                    required_features,
                });
                pkg_data.targets.push(tgt);
            }
        }
        for mut node in meta.resolve.map_or_else(Vec::new, |it| it.nodes) {
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

        let workspace_root = AbsPathBuf::assert(meta.workspace_root);

        let target_directory = AbsPathBuf::assert(meta.target_directory);

        CargoWorkspace { packages, targets, workspace_root, target_directory, manifest_path }
    }

    pub fn packages(&self) -> impl ExactSizeIterator<Item = Package> + '_ {
        self.packages.iter().map(|(id, _pkg)| id)
    }

    pub fn target_by_root(&self, root: &AbsPath) -> Option<Target> {
        self.packages()
            .filter(|&pkg| self[pkg].is_member)
            .find_map(|pkg| self[pkg].targets.iter().find(|&&it| self[it].root == root))
            .copied()
    }

    pub fn workspace_root(&self) -> &AbsPath {
        &self.workspace_root
    }

    pub fn manifest_path(&self) -> &ManifestPath {
        &self.manifest_path
    }

    pub fn target_directory(&self) -> &AbsPath {
        &self.target_directory
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
        if !parent_manifests.is_empty() {
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

    /// Returns the union of the features of all member crates in this workspace.
    pub fn workspace_features(&self) -> FxHashSet<String> {
        self.packages()
            .filter_map(|package| {
                let package = &self[package];
                if package.is_member {
                    Some(package.features.keys().cloned())
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    fn is_unique(&self, name: &str) -> bool {
        self.packages.iter().filter(|(_, v)| v.name == name).count() == 1
    }
}

fn find_list_of_build_targets(
    config: &CargoConfig,
    cargo_toml: &ManifestPath,
    sysroot: &Sysroot,
) -> Vec<String> {
    if let Some(target) = &config.target {
        return [target.into()].to_vec();
    }

    let build_targets = cargo_config_build_target(cargo_toml, &config.extra_env, sysroot);
    if !build_targets.is_empty() {
        return build_targets;
    }

    rustc_discover_host_triple(cargo_toml, &config.extra_env, sysroot).into_iter().collect()
}

fn rustc_discover_host_triple(
    cargo_toml: &ManifestPath,
    extra_env: &FxHashMap<String, String>,
    sysroot: &Sysroot,
) -> Option<String> {
    let mut rustc = sysroot.tool(Tool::Rustc);
    rustc.envs(extra_env);
    rustc.current_dir(cargo_toml.parent()).arg("-vV");
    tracing::debug!("Discovering host platform by {:?}", rustc);
    match utf8_stdout(rustc) {
        Ok(stdout) => {
            let field = "host: ";
            let target = stdout.lines().find_map(|l| l.strip_prefix(field));
            if let Some(target) = target {
                Some(target.to_owned())
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
    sysroot: &Sysroot,
) -> Vec<String> {
    let mut cargo_config = sysroot.tool(Tool::Cargo);
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
        return [trimmed.to_owned()].to_vec();
    }

    let res = serde_json::from_str(trimmed);
    if let Err(e) = &res {
        tracing::warn!("Failed to parse `build.target` as an array of target: {}`", e);
    }
    res.unwrap_or_default()
}
