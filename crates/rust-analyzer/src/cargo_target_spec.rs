//! See `CargoTargetSpec`

use ra_ide::{FileId, RunnableKind, TestId};
use ra_project_model::{self, ProjectWorkspace, TargetKind};

use crate::{world::WorldSnapshot, Result};

/// Abstract representation of Cargo target.
///
/// We use it to cook up the set of cli args we need to pass to Cargo to
/// build/test/run the target.
#[derive(Clone)]
pub(crate) struct CargoTargetSpec {
    pub(crate) package: String,
    pub(crate) target: String,
    pub(crate) target_kind: TargetKind,
}

impl CargoTargetSpec {
    pub(crate) fn runnable_args(
        spec: Option<CargoTargetSpec>,
        kind: &RunnableKind,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let mut args = Vec::new();
        let mut extra_args = Vec::new();
        match kind {
            RunnableKind::Test { test_id, attr } => {
                args.push("test".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    extra_args.push("--exact".to_string());
                }
                extra_args.push("--nocapture".to_string());
                if attr.ignore {
                    extra_args.push("--ignored".to_string());
                }
            }
            RunnableKind::TestMod { path } => {
                args.push("test".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(path.to_string());
                extra_args.push("--nocapture".to_string());
            }
            RunnableKind::Bench { test_id } => {
                args.push("bench".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(test_id.to_string());
                if let TestId::Path(_) = test_id {
                    extra_args.push("--exact".to_string());
                }
                extra_args.push("--nocapture".to_string());
            }
            RunnableKind::DocTest { test_id } => {
                args.push("test".to_string());
                args.push("--doc".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
                extra_args.push(test_id.to_string());
                extra_args.push("--nocapture".to_string());
            }
            RunnableKind::Bin => {
                args.push("run".to_string());
                if let Some(spec) = spec {
                    spec.push_to(&mut args, kind);
                }
            }
        }
        Ok((args, extra_args))
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
                    package: cargo.package_flag(&cargo[cargo[tgt].package]),
                    target: cargo[tgt].name.clone(),
                    target_kind: cargo[tgt].kind,
                })
            }
            ProjectWorkspace::Json { .. } => None,
        });
        Ok(res)
    }

    pub(crate) fn push_to(self, buf: &mut Vec<String>, kind: &RunnableKind) {
        buf.push("--package".to_string());
        buf.push(self.package);

        // Can't mix --doc with other target flags
        if let RunnableKind::DocTest { .. } = kind {
            return;
        }
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
