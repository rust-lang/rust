//! See `CargoTargetSpec`

use ra_ide::{FileId, RunnableKind, TestId};
use ra_project_model::{self, ProjectWorkspace, TargetKind};

use crate::{world::WorldSnapshot, Result};

/// Abstract representation of Cargo target.
///
/// We use it to cook up the set of cli args we need to pass to Cargo to
/// build/test/run the target.
pub(crate) struct CargoTargetSpec {
    pub(crate) package: String,
    pub(crate) target: String,
    pub(crate) target_kind: TargetKind,
}

impl CargoTargetSpec {
    pub(crate) fn runnable_args(
        spec: Option<CargoTargetSpec>,
        kind: &RunnableKind,
    ) -> Result<Vec<String>> {
        let mut res = Vec::new();
        match kind {
            RunnableKind::Test { test_id } => {
                res.push("test".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut res);
                }
                res.push("--".to_string());
                res.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    res.push("--exact".to_string());
                }
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
            RunnableKind::Bench { test_id } => {
                res.push("bench".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut res);
                }
                res.push("--".to_string());
                res.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    res.push("--exact".to_string());
                }
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

    pub(crate) fn for_file(
        world: &WorldSnapshot,
        file_id: FileId,
    ) -> Result<Option<CargoTargetSpec>> {
        let &crate_id = match world.analysis().crate_for(file_id)?.first() {
            Some(crate_id) => crate_id,
            None => return Ok(None),
        };
        let file_id = world.analysis().crate_root(crate_id)?;
        let path = world.file_id_to_path(file_id);
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

    pub(crate) fn push_to(self, buf: &mut Vec<String>) {
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
