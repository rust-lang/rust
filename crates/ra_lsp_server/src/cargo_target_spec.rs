//! FIXME: write short doc here

use ra_ide::{FileId, RunnableKind};
use ra_project_model::{self, ProjectWorkspace, TargetKind};

use crate::{world::WorldSnapshot, Result};

pub(crate) fn runnable_args(
    world: &WorldSnapshot,
    file_id: FileId,
    kind: &RunnableKind,
) -> Result<Vec<String>> {
    let spec = CargoTargetSpec::for_file(world, file_id)?;
    let mut res = Vec::new();
    match kind {
        RunnableKind::Test { name } => {
            res.push("test".to_string());
            if let Some(spec) = spec {
                spec.push_to(&mut res);
            }
            res.push("--".to_string());
            res.push(name.to_string());
            res.push("--nocapture".to_string());
        }
        RunnableKind::TestMod { path } => {
            res.push("test".to_string());
            if let Some(spec) = spec {
                spec.push_to(&mut res);
            }
            res.push("--".to_string());
            res.push(path.to_string());
            res.push("--nocapture".to_string());
        }
        RunnableKind::Bench { name } => {
            res.push("bench".to_string());
            if let Some(spec) = spec {
                spec.push_to(&mut res);
            }
            res.push("--".to_string());
            res.push(name.to_string());
            res.push("--nocapture".to_string());
        }
        RunnableKind::Bin => {
            res.push("run".to_string());
            if let Some(spec) = spec {
                spec.push_to(&mut res);
            }
        }
    }
    Ok(res)
}

pub struct CargoTargetSpec {
    pub package: String,
    pub target: String,
    pub target_kind: TargetKind,
}

impl CargoTargetSpec {
    pub fn for_file(world: &WorldSnapshot, file_id: FileId) -> Result<Option<CargoTargetSpec>> {
        let &crate_id = match world.analysis().crate_for(file_id)?.first() {
            Some(crate_id) => crate_id,
            None => return Ok(None),
        };
        let file_id = world.analysis().crate_root(crate_id)?;
        let path = world.vfs.read().file2path(ra_vfs::VfsFile(file_id.0));
        let res = world.workspaces.iter().find_map(|ws| match ws {
            ProjectWorkspace::Cargo { cargo, .. } => {
                let tgt = cargo.target_by_root(&path)?;
                Some(CargoTargetSpec {
                    package: tgt.package(&cargo).name(&cargo).to_string(),
                    target: tgt.name(&cargo).to_string(),
                    target_kind: tgt.kind(&cargo),
                })
            }
            ProjectWorkspace::Json { .. } => None,
        });
        Ok(res)
    }

    pub fn push_to(self, buf: &mut Vec<String>) {
        buf.push("--package".to_string());
        buf.push(self.package);
        match self.target_kind {
            TargetKind::Bin => {
                buf.push("--bin".to_string());
                buf.push(self.target);
            }
            TargetKind::Test => {
                buf.push("--test".to_string());
                buf.push(self.target);
            }
            TargetKind::Bench => {
                buf.push("--bench".to_string());
                buf.push(self.target);
            }
            TargetKind::Example => {
                buf.push("--example".to_string());
                buf.push(self.target);
            }
            TargetKind::Lib => {
                buf.push("--lib".to_string());
            }
            TargetKind::Other => (),
        }
    }
}
