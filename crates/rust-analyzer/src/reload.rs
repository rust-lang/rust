//! Project loading & configuration updates
use std::{mem, sync::Arc};

use crossbeam_channel::unbounded;
use flycheck::FlycheckHandle;
use ra_db::{CrateGraph, SourceRoot, VfsPath};
use ra_ide::AnalysisChange;
use ra_project_model::{PackageRoot, ProcMacroClient, ProjectWorkspace};
use vfs::{file_set::FileSetConfig, AbsPath};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    global_state::{GlobalState, Handle},
    main_loop::Task,
};

impl GlobalState {
    pub(crate) fn update_configuration(&mut self, config: Config) {
        let old_config = mem::replace(&mut self.config, config);
        if self.config.lru_capacity != old_config.lru_capacity {
            self.analysis_host.update_lru_capacity(old_config.lru_capacity);
        }
        if self.config.linked_projects != old_config.linked_projects {
            self.fetch_workspaces()
        } else if self.config.flycheck != old_config.flycheck {
            self.reload_flycheck();
        }
    }
    pub(crate) fn fetch_workspaces(&mut self) {
        self.task_pool.handle.spawn({
            let linked_projects = self.config.linked_projects.clone();
            let cargo_config = self.config.cargo.clone();
            let with_sysroot = self.config.with_sysroot.clone();
            move || {
                let workspaces = linked_projects
                    .iter()
                    .map(|project| match project {
                        LinkedProject::ProjectManifest(manifest) => {
                            ra_project_model::ProjectWorkspace::load(
                                manifest.clone(),
                                &cargo_config,
                                with_sysroot,
                            )
                        }
                        LinkedProject::InlineJsonProject(it) => {
                            Ok(ra_project_model::ProjectWorkspace::Json { project: it.clone() })
                        }
                    })
                    .collect::<Vec<_>>();
                Task::Workspaces(workspaces)
            }
        });
    }
    pub(crate) fn switch_workspaces(&mut self, workspaces: Vec<anyhow::Result<ProjectWorkspace>>) {
        log::info!("reloading projects: {:?}", self.config.linked_projects);
        let workspaces = workspaces
            .into_iter()
            .filter_map(|res| {
                res.map_err(|err| {
                    log::error!("failed to load workspace: {:#}", err);
                    self.show_message(
                        lsp_types::MessageType::Error,
                        format!("rust-analyzer failed to load workspace: {:#}", err),
                    );
                })
                .ok()
            })
            .collect::<Vec<_>>();

        if let FilesWatcher::Client = self.config.files.watcher {
            let registration_options = lsp_types::DidChangeWatchedFilesRegistrationOptions {
                watchers: workspaces
                    .iter()
                    .flat_map(ProjectWorkspace::to_roots)
                    .filter(PackageRoot::is_member)
                    .map(|root| format!("{}/**/*.rs", root.path().display()))
                    .map(|glob_pattern| lsp_types::FileSystemWatcher { glob_pattern, kind: None })
                    .collect(),
            };
            let registration = lsp_types::Registration {
                id: "workspace/didChangeWatchedFiles".to_string(),
                method: "workspace/didChangeWatchedFiles".to_string(),
                register_options: Some(serde_json::to_value(registration_options).unwrap()),
            };
            self.send_request::<lsp_types::request::RegisterCapability>(
                lsp_types::RegistrationParams { registrations: vec![registration] },
                |_, _| (),
            );
        }

        let mut change = AnalysisChange::new();

        let project_folders = ProjectFolders::new(&workspaces);

        self.proc_macro_client = match &self.config.proc_macro_srv {
            None => ProcMacroClient::dummy(),
            Some((path, args)) => match ProcMacroClient::extern_process(path.into(), args) {
                Ok(it) => it,
                Err(err) => {
                    log::error!(
                        "Failed to run ra_proc_macro_srv from path {}, error: {:?}",
                        path.display(),
                        err
                    );
                    ProcMacroClient::dummy()
                }
            },
        };
        let watch = match self.config.files.watcher {
            FilesWatcher::Client => vec![],
            FilesWatcher::Notify => project_folders.watch,
        };
        self.loader.handle.set_config(vfs::loader::Config { load: project_folders.load, watch });

        // Create crate graph from all the workspaces
        let crate_graph = {
            let mut crate_graph = CrateGraph::default();
            let vfs = &mut self.vfs.write().0;
            let loader = &mut self.loader;
            let mut load = |path: &AbsPath| {
                let contents = loader.handle.load_sync(path);
                let path = vfs::VfsPath::from(path.to_path_buf());
                vfs.set_file_contents(path.clone(), contents);
                vfs.file_id(&path)
            };
            for ws in workspaces.iter() {
                crate_graph.extend(ws.to_crate_graph(
                    self.config.cargo.target.as_deref(),
                    &self.proc_macro_client,
                    &mut load,
                ));
            }

            crate_graph
        };
        change.set_crate_graph(crate_graph);

        self.source_root_config = project_folders.source_root_config;
        self.workspaces = Arc::new(workspaces);

        self.analysis_host.apply_change(change);
        self.process_changes();
        self.reload_flycheck();
    }

    fn reload_flycheck(&mut self) {
        let config = match self.config.flycheck.clone() {
            Some(it) => it,
            None => {
                self.flycheck = None;
                return;
            }
        };

        // FIXME: Figure out the multi-workspace situation
        self.flycheck = self.workspaces.iter().find_map(move |w| match w {
            ProjectWorkspace::Cargo { cargo, .. } => {
                let (sender, receiver) = unbounded();
                let sender = Box::new(move |msg| sender.send(msg).unwrap());
                let cargo_project_root = cargo.workspace_root().to_path_buf();
                let handle =
                    FlycheckHandle::spawn(sender, config.clone(), cargo_project_root.into());
                Some(Handle { handle, receiver })
            }
            ProjectWorkspace::Json { .. } => {
                log::warn!("Cargo check watching only supported for cargo workspaces, disabling");
                None
            }
        })
    }
}

#[derive(Default)]
pub(crate) struct ProjectFolders {
    pub(crate) load: Vec<vfs::loader::Entry>,
    pub(crate) watch: Vec<usize>,
    pub(crate) source_root_config: SourceRootConfig,
}

impl ProjectFolders {
    pub(crate) fn new(workspaces: &[ProjectWorkspace]) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        for root in workspaces.iter().flat_map(|it| it.to_roots()) {
            let path = root.path().to_owned();

            let mut file_set_roots: Vec<VfsPath> = vec![];

            let entry = if root.is_member() {
                vfs::loader::Entry::local_cargo_package(path.to_path_buf())
            } else {
                vfs::loader::Entry::cargo_package_dependency(path.to_path_buf())
            };
            res.load.push(entry);
            if root.is_member() {
                res.watch.push(res.load.len() - 1);
            }

            if let Some(out_dir) = root.out_dir() {
                let out_dir = out_dir.to_path_buf();
                res.load.push(vfs::loader::Entry::rs_files_recursively(out_dir.clone()));
                if root.is_member() {
                    res.watch.push(res.load.len() - 1);
                }
                file_set_roots.push(out_dir.into());
            }
            file_set_roots.push(path.to_path_buf().into());

            if root.is_member() {
                local_filesets.push(fsc.len());
            }
            fsc.add_file_set(file_set_roots)
        }

        let fsc = fsc.build();
        res.source_root_config = SourceRootConfig { fsc, local_filesets };

        res
    }
}

#[derive(Default, Debug)]
pub(crate) struct SourceRootConfig {
    pub(crate) fsc: FileSetConfig,
    pub(crate) local_filesets: Vec<usize>,
}

impl SourceRootConfig {
    pub(crate) fn partition(&self, vfs: &vfs::Vfs) -> Vec<SourceRoot> {
        self.fsc
            .partition(vfs)
            .into_iter()
            .enumerate()
            .map(|(idx, file_set)| {
                let is_local = self.local_filesets.contains(&idx);
                if is_local {
                    SourceRoot::new_local(file_set)
                } else {
                    SourceRoot::new_library(file_set)
                }
            })
            .collect()
    }
}
