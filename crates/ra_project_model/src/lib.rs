mod cargo_workspace;
mod sysroot;

use std::path::{Path, PathBuf};

use failure::bail;
use rustc_hash::FxHashMap;

use ra_db::{CrateGraph, FileId};
use ra_vfs::Vfs;

pub use crate::{
    cargo_workspace::{CargoWorkspace, Package, Target, TargetKind},
    sysroot::Sysroot,
};

// TODO use own error enum?
pub type Result<T> = ::std::result::Result<T, ::failure::Error>;

#[derive(Debug, Clone)]
pub struct ProjectWorkspace {
    pub cargo: CargoWorkspace,
    pub sysroot: Sysroot,
}

impl ProjectWorkspace {
    pub fn discover(path: &Path) -> Result<ProjectWorkspace> {
        let cargo_toml = find_cargo_toml(path)?;
        let cargo = CargoWorkspace::from_cargo_metadata(&cargo_toml)?;
        let sysroot = Sysroot::discover(&cargo_toml)?;
        let res = ProjectWorkspace { cargo, sysroot };
        Ok(res)
    }

    pub fn to_crate_graph(&self, vfs: &mut Vfs) -> CrateGraph {
        let mut crate_graph = CrateGraph::default();
        let mut sysroot_crates = FxHashMap::default();
        for krate in self.sysroot.crates() {
            if let Some(file_id) = vfs.load(krate.root(&self.sysroot)) {
                let file_id = FileId(file_id.0.into());
                sysroot_crates.insert(krate, crate_graph.add_crate_root(file_id));
            }
        }
        for from in self.sysroot.crates() {
            for to in from.deps(&self.sysroot) {
                let name = to.name(&self.sysroot);
                if let (Some(&from), Some(&to)) =
                    (sysroot_crates.get(&from), sysroot_crates.get(&to))
                {
                    if let Err(_) = crate_graph.add_dep(from, name.clone(), to) {
                        log::error!("cyclic dependency between sysroot crates")
                    }
                }
            }
        }

        let libstd = self.sysroot.std().and_then(|it| sysroot_crates.get(&it).map(|&it| it));

        let mut pkg_to_lib_crate = FxHashMap::default();
        let mut pkg_crates = FxHashMap::default();
        // Next, create crates for each package, target pair
        for pkg in self.cargo.packages() {
            let mut lib_tgt = None;
            for tgt in pkg.targets(&self.cargo) {
                let root = tgt.root(&self.cargo);
                if let Some(file_id) = vfs.load(root) {
                    let file_id = FileId(file_id.0.into());
                    let crate_id = crate_graph.add_crate_root(file_id);
                    if tgt.kind(&self.cargo) == TargetKind::Lib {
                        lib_tgt = Some(crate_id);
                        pkg_to_lib_crate.insert(pkg, crate_id);
                    }
                    pkg_crates.entry(pkg).or_insert_with(Vec::new).push(crate_id);
                }
            }

            // Set deps to the std and to the lib target of the current package
            for &from in pkg_crates.get(&pkg).into_iter().flatten() {
                if let Some(to) = lib_tgt {
                    if to != from {
                        if let Err(_) = crate_graph.add_dep(from, pkg.name(&self.cargo).into(), to)
                        {
                            log::error!(
                                "cyclic dependency between targets of {}",
                                pkg.name(&self.cargo)
                            )
                        }
                    }
                }
                if let Some(std) = libstd {
                    if let Err(_) = crate_graph.add_dep(from, "std".into(), std) {
                        log::error!("cyclic dependency on std for {}", pkg.name(&self.cargo))
                    }
                }
            }
        }

        // Now add a dep ednge from all targets of upstream to the lib
        // target of downstream.
        for pkg in self.cargo.packages() {
            for dep in pkg.dependencies(&self.cargo) {
                if let Some(&to) = pkg_to_lib_crate.get(&dep.pkg) {
                    for &from in pkg_crates.get(&pkg).into_iter().flatten() {
                        if let Err(_) = crate_graph.add_dep(from, dep.name.clone(), to) {
                            log::error!(
                                "cyclic dependency {} -> {}",
                                pkg.name(&self.cargo),
                                dep.pkg.name(&self.cargo)
                            )
                        }
                    }
                }
            }
        }

        crate_graph
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
