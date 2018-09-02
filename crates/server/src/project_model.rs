use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use libsyntax2::SmolStr;
use cargo_metadata::{metadata_run, CargoOpt};
use Result;

#[derive(Debug)]
pub struct CargoWorkspace {
    ws_members: Vec<Package>,
    packages: Vec<PackageData>,
    targets: Vec<TargetData>,
}

#[derive(Clone, Copy, Debug)]
pub struct Package(usize);
#[derive(Clone, Copy, Debug)]
pub struct Target(usize);

#[derive(Debug)]
struct PackageData {
    name: SmolStr,
    manifest: PathBuf,
    targets: Vec<Target>
}

#[derive(Debug)]
struct TargetData {
    pkg: Package,
    name: SmolStr,
    root: PathBuf,
    kind: TargetKind,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TargetKind {
    Bin, Lib, Example, Test, Bench, Other,
}

impl Package {
    pub fn name(self, ws: &CargoWorkspace) -> &str {
        ws.pkg(self).name.as_str()
    }
    pub fn manifest(self, ws: &CargoWorkspace) -> &Path {
        ws.pkg(self).manifest.as_path()
    }
    pub fn targets<'a>(self, ws: &'a CargoWorkspace) -> impl Iterator<Item=Target> + 'a {
        ws.pkg(self).targets.iter().cloned()
    }
}

impl Target {
    pub fn pkg(self, ws: &CargoWorkspace) -> Package {
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
    pub fn from_path(path: &Path) -> Result<CargoWorkspace> {
        let meta = metadata_run(
            Some(path),
            true,
            Some(CargoOpt::AllFeatures)
        ).map_err(|e| format_err!("cargo metadata failed: {}", e))?;
        let mut pkg_by_id = HashMap::new();
        let mut packages = Vec::new();
        let mut targets = Vec::new();
        for meta_pkg in meta.packages {
            let pkg = Package(packages.len());
            pkg_by_id.insert(meta_pkg.id.clone(), pkg);
            let mut pkg_data = PackageData {
                name: meta_pkg.name.into(),
                manifest: PathBuf::from(meta_pkg.manifest_path),
                targets: Vec::new(),
            };
            for meta_tgt in meta_pkg.targets {
                let tgt = Target(targets.len());
                targets.push(TargetData {
                    pkg,
                    name: meta_tgt.name.into(),
                    root: PathBuf::from(meta_tgt.src_path),
                    kind: TargetKind::new(meta_tgt.kind.as_slice()),
                });
                pkg_data.targets.push(tgt);
            }
            packages.push(pkg_data)
        }
        let ws_members = meta.workspace_members
            .iter()
            .map(|it| pkg_by_id[&it.raw])
            .collect();

        Ok(CargoWorkspace { packages, targets, ws_members })
    }
    pub fn packages<'a>(&'a self) -> impl Iterator<Item=Package> + 'a {
        (0..self.packages.len()).map(Package)
    }
    pub fn ws_members<'a>(&'a self) -> impl Iterator<Item=Package> + 'a {
        self.ws_members.iter().cloned()
    }
    pub fn target_by_roo(&self, root: &Path) -> Option<Target> {
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
            }
        }
        TargetKind::Other
    }
}
