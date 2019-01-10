mod cargo_workspace;
mod sysroot;

use std::{
    path::{Path, PathBuf},
};

use failure::bail;
use thread_worker::{WorkerHandle, Worker};

use crate::Result;

pub use crate::project_model::{
    cargo_workspace::{CargoWorkspace, Package, Target, TargetKind},
    sysroot::Sysroot,
};

#[derive(Debug, Clone)]
pub struct ProjectWorkspace {
    pub(crate) cargo: CargoWorkspace,
    pub(crate) sysroot: Sysroot,
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

pub fn workspace_loader() -> (Worker<PathBuf, Result<ProjectWorkspace>>, WorkerHandle) {
    thread_worker::spawn::<PathBuf, Result<ProjectWorkspace>, _>(
        "workspace loader",
        1,
        |input_receiver, output_sender| {
            input_receiver
                .into_iter()
                .map(|path| ProjectWorkspace::discover(path.as_path()))
                .try_for_each(|it| output_sender.send(it))
                .unwrap()
        },
    )
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
