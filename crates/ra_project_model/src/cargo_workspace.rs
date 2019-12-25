//! FIXME: write short doc here

use std::path::{Path, PathBuf};

use cargo_metadata::{CargoOpt, MetadataCommand};
use ra_arena::{impl_arena_id, Arena, RawId};
use ra_db::Edition;
use rustc_hash::FxHashMap;
use serde::Deserialize;

use crate::Result;

/// `CargoWorkspace` represents the logical structure of, well, a Cargo
/// workspace. It pretty closely mirrors `cargo metadata` output.
///
/// Note that internally, rust analyzer uses a different structure:
/// `CrateGraph`. `CrateGraph` is lower-level: it knows only about the crates,
/// while this knows about `Packages` & `Targets`: purely cargo-related
/// concepts.
#[derive(Debug, Clone)]
pub struct CargoWorkspace {
    packages: Arena<Package, PackageData>,
    targets: Arena<Target, TargetData>,
    pub(crate) workspace_root: PathBuf,
}

#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase", default)]
pub struct CargoFeatures {
    /// Do not activate the `default` feature.
    pub no_default_features: bool,

    /// Activate all available features
    pub all_features: bool,

    /// List of features to activate.
    /// This will be ignored if `cargo_all_features` is true.
    pub features: Vec<String>,
}

impl Default for CargoFeatures {
    fn default() -> Self {
        CargoFeatures { no_default_features: false, all_features: true, features: Vec::new() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Package(RawId);
impl_arena_id!(Package);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target(RawId);
impl_arena_id!(Target);

#[derive(Debug, Clone)]
struct PackageData {
    name: String,
    manifest: PathBuf,
    targets: Vec<Target>,
    is_member: bool,
    dependencies: Vec<PackageDependency>,
    edition: Edition,
    features: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PackageDependency {
    pub pkg: Package,
    pub name: String,
}

#[derive(Debug, Clone)]
struct TargetData {
    pkg: Package,
    name: String,
    root: PathBuf,
    kind: TargetKind,
    is_proc_macro: bool,
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

impl Package {
    pub fn name(self, ws: &CargoWorkspace) -> &str {
        ws.packages[self].name.as_str()
    }
    pub fn root(self, ws: &CargoWorkspace) -> &Path {
        ws.packages[self].manifest.parent().unwrap()
    }
    pub fn edition(self, ws: &CargoWorkspace) -> Edition {
        ws.packages[self].edition
    }
    pub fn features(self, ws: &CargoWorkspace) -> &[String] {
        &ws.packages[self].features
    }
    pub fn targets<'a>(self, ws: &'a CargoWorkspace) -> impl Iterator<Item = Target> + 'a {
        ws.packages[self].targets.iter().cloned()
    }
    #[allow(unused)]
    pub fn is_member(self, ws: &CargoWorkspace) -> bool {
        ws.packages[self].is_member
    }
    pub fn dependencies<'a>(
        self,
        ws: &'a CargoWorkspace,
    ) -> impl Iterator<Item = &'a PackageDependency> + 'a {
        ws.packages[self].dependencies.iter()
    }
}

impl Target {
    pub fn package(self, ws: &CargoWorkspace) -> Package {
        ws.targets[self].pkg
    }
    pub fn name(self, ws: &CargoWorkspace) -> &str {
        ws.targets[self].name.as_str()
    }
    pub fn root(self, ws: &CargoWorkspace) -> &Path {
        ws.targets[self].root.as_path()
    }
    pub fn kind(self, ws: &CargoWorkspace) -> TargetKind {
        ws.targets[self].kind
    }
    pub fn is_proc_macro(self, ws: &CargoWorkspace) -> bool {
        ws.targets[self].is_proc_macro
    }
}

impl CargoWorkspace {
    pub fn from_cargo_metadata(
        cargo_toml: &Path,
        cargo_features: &CargoFeatures,
    ) -> Result<CargoWorkspace> {
        let mut meta = MetadataCommand::new();
        meta.manifest_path(cargo_toml);
        if cargo_features.all_features {
            meta.features(CargoOpt::AllFeatures);
        } else if cargo_features.no_default_features {
            // FIXME: `NoDefaultFeatures` is mutual exclusive with `SomeFeatures`
            // https://github.com/oli-obk/cargo_metadata/issues/79
            meta.features(CargoOpt::NoDefaultFeatures);
        } else if cargo_features.features.len() > 0 {
            meta.features(CargoOpt::SomeFeatures(cargo_features.features.clone()));
        }
        if let Some(parent) = cargo_toml.parent() {
            meta.current_dir(parent);
        }
        let meta = meta.exec().map_err(|e| format!("cargo metadata failed: {}", e))?;
        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        for meta_pkg in meta.packages {
            let cargo_metadata::Package { id, edition, name, manifest_path, .. } = meta_pkg;
            let is_member = ws_members.contains(&id);
            let edition = edition.parse::<Edition>()?;
            let pkg = packages.alloc(PackageData {
                name,
                manifest: manifest_path,
                targets: Vec::new(),
                is_member,
                edition,
                dependencies: Vec::new(),
                features: Vec::new(),
            });
            let pkg_data = &mut packages[pkg];
            pkg_by_id.insert(id, pkg);
            for meta_tgt in meta_pkg.targets {
                let is_proc_macro = meta_tgt.kind.as_slice() == &["proc-macro"];
                let tgt = targets.alloc(TargetData {
                    pkg,
                    name: meta_tgt.name,
                    root: meta_tgt.src_path.clone(),
                    kind: TargetKind::new(meta_tgt.kind.as_slice()),
                    is_proc_macro,
                });
                pkg_data.targets.push(tgt);
            }
        }
        let resolve = meta.resolve.expect("metadata executed with deps");
        for node in resolve.nodes {
            let source = pkg_by_id[&node.id];
            for dep_node in node.deps {
                let dep = PackageDependency { name: dep_node.name, pkg: pkg_by_id[&dep_node.pkg] };
                packages[source].dependencies.push(dep);
            }
            packages[source].features.extend(node.features);
        }

        Ok(CargoWorkspace { packages, targets, workspace_root: meta.workspace_root })
    }

    pub fn packages<'a>(&'a self) -> impl Iterator<Item = Package> + ExactSizeIterator + 'a {
        self.packages.iter().map(|(id, _pkg)| id)
    }

    pub fn target_by_root(&self, root: &Path) -> Option<Target> {
        self.packages().filter_map(|pkg| pkg.targets(self).find(|it| it.root(self) == root)).next()
    }
}
