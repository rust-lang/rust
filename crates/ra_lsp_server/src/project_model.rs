use std::path::{Path, PathBuf};

use cargo_metadata::{metadata_run, CargoOpt};
use ra_syntax::SmolStr;
use rustc_hash::FxHashMap;
use failure::{format_err, bail};
use thread_worker::{WorkerHandle, Worker};

use crate::Result;

/// `CargoWorksapce` represents the logical structure of, well, a Cargo
/// workspace. It pretty closely mirrors `cargo metadata` output.
///
/// Note that internally, rust analyzer uses a differnet structure:
/// `CrateGraph`. `CrateGraph` is lower-level: it knows only about the crates,
/// while this knows about `Pacakges` & `Targets`: purely cargo-related
/// concepts.
#[derive(Debug, Clone)]
pub struct CargoWorkspace {
    packages: Vec<PackageData>,
    targets: Vec<TargetData>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Package(usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target(usize);

#[derive(Debug, Clone)]
struct PackageData {
    name: SmolStr,
    manifest: PathBuf,
    targets: Vec<Target>,
    is_member: bool,
    dependencies: Vec<PackageDependency>,
}

#[derive(Debug, Clone)]
pub struct PackageDependency {
    pub pkg: Package,
    pub name: SmolStr,
}

#[derive(Debug, Clone)]
struct TargetData {
    pkg: Package,
    name: SmolStr,
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

impl Package {
    pub fn name(self, ws: &CargoWorkspace) -> &str {
        ws.pkg(self).name.as_str()
    }
    pub fn root(self, ws: &CargoWorkspace) -> &Path {
        ws.pkg(self).manifest.parent().unwrap()
    }
    pub fn targets<'a>(self, ws: &'a CargoWorkspace) -> impl Iterator<Item = Target> + 'a {
        ws.pkg(self).targets.iter().cloned()
    }
    #[allow(unused)]
    pub fn is_member(self, ws: &CargoWorkspace) -> bool {
        ws.pkg(self).is_member
    }
    pub fn dependencies<'a>(
        self,
        ws: &'a CargoWorkspace,
    ) -> impl Iterator<Item = &'a PackageDependency> + 'a {
        ws.pkg(self).dependencies.iter()
    }
}

impl Target {
    pub fn package(self, ws: &CargoWorkspace) -> Package {
        ws.tgt(self).pkg
    }
    pub fn name(self, ws: &CargoWorkspace) -> &str {
        ws.tgt(self).name.as_str()
    }
    pub fn root(self, ws: &CargoWorkspace) -> &Path {
        ws.tgt(self).root.as_path()
    }
    pub fn kind(self, ws: &CargoWorkspace) -> TargetKind {
        ws.tgt(self).kind
    }
}

impl CargoWorkspace {
    pub fn from_cargo_metadata(path: &Path) -> Result<CargoWorkspace> {
        let cargo_toml = find_cargo_toml(path)?;
        let meta = metadata_run(
            Some(cargo_toml.as_path()),
            true,
            Some(CargoOpt::AllFeatures),
        )
        .map_err(|e| format_err!("cargo metadata failed: {}", e))?;
        let mut pkg_by_id = FxHashMap::default();
        let mut packages = Vec::new();
        let mut targets = Vec::new();

        let ws_members = &meta.workspace_members;

        for meta_pkg in meta.packages {
            let pkg = Package(packages.len());
            let is_member = ws_members.contains(&meta_pkg.id);
            pkg_by_id.insert(meta_pkg.id.clone(), pkg);
            let mut pkg_data = PackageData {
                name: meta_pkg.name.into(),
                manifest: meta_pkg.manifest_path.clone(),
                targets: Vec::new(),
                is_member,
                dependencies: Vec::new(),
            };
            for meta_tgt in meta_pkg.targets {
                let tgt = Target(targets.len());
                targets.push(TargetData {
                    pkg,
                    name: meta_tgt.name.into(),
                    root: meta_tgt.src_path.clone(),
                    kind: TargetKind::new(meta_tgt.kind.as_slice()),
                });
                pkg_data.targets.push(tgt);
            }
            packages.push(pkg_data)
        }
        let resolve = meta.resolve.expect("metadata executed with deps");
        for node in resolve.nodes {
            let source = pkg_by_id[&node.id];
            for dep_node in node.deps {
                let dep = PackageDependency {
                    name: dep_node.name.into(),
                    pkg: pkg_by_id[&dep_node.pkg],
                };
                packages[source.0].dependencies.push(dep);
            }
        }

        Ok(CargoWorkspace { packages, targets })
    }
    pub fn packages<'a>(&'a self) -> impl Iterator<Item = Package> + 'a {
        (0..self.packages.len()).map(Package)
    }
    pub fn target_by_root(&self, root: &Path) -> Option<Target> {
        self.packages()
            .filter_map(|pkg| pkg.targets(self).find(|it| it.root(self) == root))
            .next()
    }
    fn pkg(&self, pkg: Package) -> &PackageData {
        &self.packages[pkg.0]
    }
    fn tgt(&self, tgt: Target) -> &TargetData {
        &self.targets[tgt.0]
    }
}

fn find_cargo_toml(path: &Path) -> Result<PathBuf> {
    if path.ends_with("Cargo.toml") {
        return Ok(path.to_path_buf());
    }
    let mut curr = Some(path);
    while let Some(path) = curr {
        let candidate = path.join("Cargo.toml");
        if candidate.exists() {
            return Ok(candidate);
        }
        curr = path.parent();
    }
    bail!("can't find Cargo.toml at {}", path.display())
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

pub fn workspace_loader() -> (Worker<PathBuf, Result<CargoWorkspace>>, WorkerHandle) {
    thread_worker::spawn::<PathBuf, Result<CargoWorkspace>, _>(
        "workspace loader",
        1,
        |input_receiver, output_sender| {
            input_receiver
                .into_iter()
                .map(|path| CargoWorkspace::from_cargo_metadata(path.as_path()))
                .try_for_each(|it| output_sender.send(it))
                .unwrap()
        },
    )
}
