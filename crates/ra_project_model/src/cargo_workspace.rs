use std::path::{Path, PathBuf};

use cargo_metadata::{CargoOpt, MetadataCommand};
use ra_arena::{impl_arena_id, Arena, RawId};
use ra_db::Edition;
use rustc_hash::FxHashMap;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    Bin,
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
}

impl CargoWorkspace {
    pub fn from_cargo_metadata(cargo_toml: &Path) -> Result<CargoWorkspace> {
        let mut meta = MetadataCommand::new();
        meta.manifest_path(cargo_toml).features(CargoOpt::AllFeatures);
        if let Some(parent) = cargo_toml.parent() {
            meta.current_dir(parent);
        }
        let meta = meta.exec().map_err(|e| format!("cargo metadata failed: {}", e))?;
        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Arena::default();
        let mut targets = Arena::default();

        let ws_members = &meta.workspace_members;

        for meta_pkg in meta.packages {
            let is_member = ws_members.contains(&meta_pkg.id);
            let pkg = packages.alloc(PackageData {
                name: meta_pkg.name,
                manifest: meta_pkg.manifest_path.clone(),
                targets: Vec::new(),
                is_member,
                edition: Edition::from_string(&meta_pkg.edition),
                dependencies: Vec::new(),
            });
            let pkg_data = &mut packages[pkg];
            pkg_by_id.insert(meta_pkg.id.clone(), pkg);
            for meta_tgt in meta_pkg.targets {
                let tgt = targets.alloc(TargetData {
                    pkg,
                    name: meta_tgt.name,
                    root: meta_tgt.src_path.clone(),
                    kind: TargetKind::new(meta_tgt.kind.as_slice()),
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
