use std::path::PathBuf;

use crate::{thread_worker::Worker, Result};

pub use ra_project_model::{
    CargoWorkspace, Package, ProjectWorkspace, Sysroot, Target, TargetKind,
};

pub fn workspace_loader(with_sysroot: bool) -> Worker<PathBuf, Result<ProjectWorkspace>> {
    Worker::<PathBuf, Result<ProjectWorkspace>>::spawn(
        "workspace loader",
        1,
        move |input_receiver, output_sender| {
            input_receiver
                .into_iter()
                .map(|path| ProjectWorkspace::discover_with_sysroot(path.as_path(), with_sysroot))
                .try_for_each(|it| output_sender.send(it))
                .unwrap()
        },
    )
}
