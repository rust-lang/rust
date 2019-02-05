use std::path::PathBuf;

use thread_worker::{WorkerHandle, Worker};

use crate::Result;

pub use ra_project_model::{
    ProjectWorkspace, CargoWorkspace, Package, Target, TargetKind, Sysroot,
};

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
