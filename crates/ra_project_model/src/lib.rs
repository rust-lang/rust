mod cargo_workspace;
mod sysroot;

use std::path::{Path, PathBuf};

use failure::bail;

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
