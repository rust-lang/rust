//! See [`CargoWorkspace`].

use std::ops;
use std::str::from_utf8;

use anyhow::Context;
use base_db::Env;
use cargo_metadata::{CargoOpt, MetadataCommand};
use la_arena::{Arena, Idx};
use paths::{AbsPath, AbsPathBuf, Utf8Path, Utf8PathBuf};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_derive::Deserialize;
use serde_json::from_value;
use span::Edition;
use stdx::process::spawn_with_streaming_output;
use toolchain::Tool;

use crate::{CfgOverrides, InvocationStrategy};
use crate::{ManifestPath, Sysroot};

const MINIMUM_TOOLCHAIN_VERSION_SUPPORTING_LOCKFILE_PATH: semver::Version = semver::Version {
    major: 1,
    minor: 82,
    patch: 0,
    pre: semver::Prerelease::EMPTY,
    build: semver::BuildMetadata::EMPTY,
};

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
    is_virtual_workspace: bool,
    /// Whether this workspace represents the sysroot workspace.
    is_sysroot: bool,
    /// Environment variables set in the `.cargo/config` file.
    config_env: Env,
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
    pub sysroot_src: Option<AbsPathBuf>,
    /// rustc private crate source
    pub rustc_source: Option<RustLibSource>,
    /// Extra includes to add to the VFS.
    pub extra_includes: Vec<AbsPathBuf>,
    pub cfg_overrides: CfgOverrides,
    /// Invoke `cargo check` through the RUSTC_WRAPPER.
    pub wrap_rustc_in_build_scripts: bool,
    /// The command to run instead of `cargo check` for building build scripts.
    pub run_build_script_command: Option<Vec<String>>,
    /// Extra args to pass to the cargo command.
    pub extra_args: Vec<String>,
    /// Extra env vars to set when invoking the cargo command
    pub extra_env: FxHashMap<String, Option<String>>,
    pub invocation_strategy: InvocationStrategy,
    /// Optional path to use instead of `target` when building
    pub target_dir: Option<Utf8PathBuf>,
    /// Gate `#[test]` behind `#[cfg(test)]`
    pub set_test: bool,
    /// Load the project without any dependencies
    pub no_deps: bool,
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
    /// Cargo calls this kind `custom-build`
    BuildScript,
    Other,
}

impl TargetKind {
    fn new(kinds: &[cargo_metadata::TargetKind]) -> TargetKind {
        for kind in kinds {
            return match kind {
                cargo_metadata::TargetKind::Bin => TargetKind::Bin,
                cargo_metadata::TargetKind::Test => TargetKind::Test,
                cargo_metadata::TargetKind::Bench => TargetKind::Bench,
                cargo_metadata::TargetKind::Example => TargetKind::Example,
                cargo_metadata::TargetKind::CustomBuild => TargetKind::BuildScript,
                cargo_metadata::TargetKind::ProcMacro => TargetKind::Lib { is_proc_macro: true },
                cargo_metadata::TargetKind::Lib
                | cargo_metadata::TargetKind::DyLib
                | cargo_metadata::TargetKind::CDyLib
                | cargo_metadata::TargetKind::StaticLib
                | cargo_metadata::TargetKind::RLib => TargetKind::Lib { is_proc_macro: false },
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

    /// If this is a valid cargo target, returns the name cargo uses in command line arguments
    /// and output, otherwise None.
    /// https://docs.rs/cargo_metadata/latest/cargo_metadata/enum.TargetKind.html
    pub fn as_cargo_target(self) -> Option<&'static str> {
        match self {
            TargetKind::Bin => Some("bin"),
            TargetKind::Lib { is_proc_macro: true } => Some("proc-macro"),
            TargetKind::Lib { is_proc_macro: false } => Some("lib"),
            TargetKind::Example => Some("example"),
            TargetKind::Test => Some("test"),
            TargetKind::Bench => Some("bench"),
            TargetKind::BuildScript => Some("custom-build"),
            TargetKind::Other => None,
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct CargoMetadataConfig {
    /// List of features to activate.
    pub features: CargoFeatures,
    /// rustc targets
    pub targets: Vec<String>,
    /// Extra args to pass to the cargo command.
    pub extra_args: Vec<String>,
    /// Extra env vars to set when invoking the cargo command
    pub extra_env: FxHashMap<String, Option<String>>,
    /// The target dir for this workspace load.
    pub target_dir: Utf8PathBuf,
    /// What kind of metadata are we fetching: workspace, rustc, or sysroot.
    pub kind: &'static str,
    /// The toolchain version, if known.
    /// Used to conditionally enable unstable cargo features.
    pub toolchain_version: Option<semver::Version>,
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
    ///
    /// The sysroot is used to set the `RUSTUP_TOOLCHAIN` env var when invoking cargo
    /// to ensure that the rustup proxy uses the correct toolchain.
    pub fn fetch_metadata(
        cargo_toml: &ManifestPath,
        current_dir: &AbsPath,
        config: &CargoMetadataConfig,
        sysroot: &Sysroot,
        no_deps: bool,
        locked: bool,
        progress: &dyn Fn(String),
    ) -> anyhow::Result<(cargo_metadata::Metadata, Option<anyhow::Error>)> {
        let res = Self::fetch_metadata_(
            cargo_toml,
            current_dir,
            config,
            sysroot,
            no_deps,
            locked,
            progress,
        );
        if let Ok((_, Some(ref e))) = res {
            tracing::warn!(
                %cargo_toml,
                ?e,
                "`cargo metadata` failed, but retry with `--no-deps` succeeded"
            );
        }
        res
    }

    fn fetch_metadata_(
        cargo_toml: &ManifestPath,
        current_dir: &AbsPath,
        config: &CargoMetadataConfig,
        sysroot: &Sysroot,
        no_deps: bool,
        locked: bool,
        progress: &dyn Fn(String),
    ) -> anyhow::Result<(cargo_metadata::Metadata, Option<anyhow::Error>)> {
        let cargo = sysroot.tool(Tool::Cargo, current_dir, &config.extra_env);
        let mut meta = MetadataCommand::new();
        meta.cargo_path(cargo.get_program());
        cargo.get_envs().for_each(|(var, val)| _ = meta.env(var, val.unwrap_or_default()));
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

        if !config.targets.is_empty() {
            other_options.extend(
                config.targets.iter().flat_map(|it| ["--filter-platform".to_owned(), it.clone()]),
            );
        }
        if no_deps {
            other_options.push("--no-deps".to_owned());
        }

        let mut using_lockfile_copy = false;
        // The manifest is a rust file, so this means its a script manifest
        if cargo_toml.is_rust_manifest() {
            other_options.push("-Zscript".to_owned());
        } else if config
            .toolchain_version
            .as_ref()
            .is_some_and(|v| *v >= MINIMUM_TOOLCHAIN_VERSION_SUPPORTING_LOCKFILE_PATH)
        {
            let lockfile = <_ as AsRef<Utf8Path>>::as_ref(cargo_toml).with_extension("lock");
            let target_lockfile = config
                .target_dir
                .join("rust-analyzer")
                .join("metadata")
                .join(config.kind)
                .join("Cargo.lock");
            match std::fs::copy(&lockfile, &target_lockfile) {
                Ok(_) => {
                    using_lockfile_copy = true;
                    other_options.push("--lockfile-path".to_owned());
                    other_options.push(target_lockfile.to_string());
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // There exists no lockfile yet
                    using_lockfile_copy = true;
                    other_options.push("--lockfile-path".to_owned());
                    other_options.push(target_lockfile.to_string());
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to copy lock file from `{lockfile}` to `{target_lockfile}`: {e}",
                    );
                }
            }
        }
        if using_lockfile_copy {
            other_options.push("-Zunstable-options".to_owned());
            meta.env("RUSTC_BOOTSTRAP", "1");
        }
        // No need to lock it if we copied the lockfile, we won't modify the original after all/
        // This way cargo cannot error out on us if the lockfile requires updating.
        if !using_lockfile_copy && locked {
            other_options.push("--locked".to_owned());
        }
        meta.other_options(other_options);

        // FIXME: Fetching metadata is a slow process, as it might require
        // calling crates.io. We should be reporting progress here, but it's
        // unclear whether cargo itself supports it.
        progress("cargo metadata: started".to_owned());

        let res = (|| -> anyhow::Result<(_, _)> {
            let mut errored = false;
            let output =
                spawn_with_streaming_output(meta.cargo_command(), &mut |_| (), &mut |line| {
                    errored = errored || line.starts_with("error") || line.starts_with("warning");
                    if errored {
                        progress("cargo metadata: ?".to_owned());
                        return;
                    }
                    progress(format!("cargo metadata: {line}"));
                })?;
            if !output.status.success() {
                progress(format!("cargo metadata: failed {}", output.status));
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
                        true,
                        locked,
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
        .with_context(|| format!("Failed to run `{:?}`", meta.cargo_command()));
        progress("cargo metadata: finished".to_owned());
        res
    }

    pub fn new(
        mut meta: cargo_metadata::Metadata,
        ws_manifest_path: ManifestPath,
        cargo_config_env: Env,
        is_sysroot: bool,
    ) -> CargoWorkspace {
        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        let workspace_root = AbsPathBuf::assert(meta.workspace_root);
        let target_directory = AbsPathBuf::assert(meta.target_directory);
        let mut is_virtual_workspace = true;

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
                cargo_metadata::Edition::E2024 => Edition::Edition2024,
                _ => {
                    tracing::error!("Unsupported edition `{:?}`", edition);
                    Edition::CURRENT
                }
            };
            // We treat packages without source as "local" packages. That includes all members of
            // the current workspace, as well as any path dependency outside the workspace.
            let is_local = source.is_none();
            let is_member = ws_members.contains(&id);

            let manifest = ManifestPath::try_from(AbsPathBuf::assert(manifest_path)).unwrap();
            is_virtual_workspace &= manifest != ws_manifest_path;
            let pkg = packages.alloc(PackageData {
                id: id.repr.clone(),
                name: name.to_string(),
                version,
                manifest: manifest.clone(),
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
                        manifest.clone().into()
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
                let dep = PackageDependency { name: dep_node.name.to_string(), pkg, kind };
                packages[source].dependencies.push(dep);
            }
            packages[source]
                .active_features
                .extend(node.features.into_iter().map(|it| it.to_string()));
        }

        CargoWorkspace {
            packages,
            targets,
            workspace_root,
            target_directory,
            manifest_path: ws_manifest_path,
            is_virtual_workspace,
            is_sysroot,
            config_env: cargo_config_env,
        }
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
                ManifestPath::try_from(self.workspace_root().join("Cargo.toml")).ok()?,
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
                    Some(package.features.keys().cloned().chain(
                        package.features.keys().map(|key| format!("{}/{key}", package.name)),
                    ))
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

    pub fn is_virtual_workspace(&self) -> bool {
        self.is_virtual_workspace
    }

    pub fn env(&self) -> &Env {
        &self.config_env
    }

    pub fn is_sysroot(&self) -> bool {
        self.is_sysroot
    }
}
