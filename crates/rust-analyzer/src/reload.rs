//! Project loading & configuration updates
use std::{mem, sync::Arc};

use flycheck::{FlycheckConfig, FlycheckHandle};
use ide::Change;
use ide_db::base_db::{CrateGraph, SourceRoot, VfsPath};
use project_model::{BuildDataCollector, BuildDataResult, ProcMacroClient, ProjectWorkspace};
use vfs::{file_set::FileSetConfig, AbsPath, AbsPathBuf, ChangeKind};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    global_state::{GlobalState, Status},
    lsp_ext,
    main_loop::Task,
};
use lsp_ext::StatusParams;

#[derive(Debug)]
pub(crate) enum ProjectWorkspaceProgress {
    Begin,
    Report(String),
    End(Vec<anyhow::Result<ProjectWorkspace>>),
}

#[derive(Debug)]
pub(crate) enum BuildDataProgress {
    Begin,
    Report(String),
    End(anyhow::Result<BuildDataResult>),
}

impl GlobalState {
    pub(crate) fn update_configuration(&mut self, config: Config) {
        let _p = profile::span("GlobalState::update_configuration");
        let old_config = mem::replace(&mut self.config, Arc::new(config));
        if self.config.lru_capacity() != old_config.lru_capacity() {
            self.analysis_host.update_lru_capacity(self.config.lru_capacity());
        }
        if self.config.linked_projects() != old_config.linked_projects() {
            self.fetch_workspaces_request()
        } else if self.config.flycheck() != old_config.flycheck() {
            self.reload_flycheck();
        }
    }
    pub(crate) fn maybe_refresh(&mut self, changes: &[(AbsPathBuf, ChangeKind)]) {
        if !changes.iter().any(|(path, kind)| is_interesting(path, *kind)) {
            return;
        }
        match self.status {
            Status::Loading | Status::NeedsReload => return,
            Status::Ready { .. } | Status::Invalid => (),
        }
        log::info!(
            "Reloading workspace because of the following changes: {}",
            itertools::join(
                changes
                    .iter()
                    .filter(|(path, kind)| is_interesting(path, *kind))
                    .map(|(path, kind)| format!("{}/{:?}", path.display(), kind)),
                ", "
            )
        );
        if self.config.cargo_autoreload() {
            self.fetch_workspaces_request();
        } else {
            self.transition(Status::NeedsReload);
        }

        fn is_interesting(path: &AbsPath, change_kind: ChangeKind) -> bool {
            const IMPLICIT_TARGET_FILES: &[&str] = &["build.rs", "src/main.rs", "src/lib.rs"];
            const IMPLICIT_TARGET_DIRS: &[&str] = &["src/bin", "examples", "tests", "benches"];

            if path.ends_with("Cargo.toml") || path.ends_with("Cargo.lock") {
                return true;
            }
            if change_kind == ChangeKind::Modify {
                return false;
            }
            if path.extension().unwrap_or_default() != "rs" {
                return false;
            }
            if IMPLICIT_TARGET_FILES.iter().any(|it| path.ends_with(it)) {
                return true;
            }
            let parent = match path.parent() {
                Some(it) => it,
                None => return false,
            };
            if IMPLICIT_TARGET_DIRS.iter().any(|it| parent.ends_with(it)) {
                return true;
            }
            if path.ends_with("main.rs") {
                let grand_parent = match parent.parent() {
                    Some(it) => it,
                    None => return false,
                };
                if IMPLICIT_TARGET_DIRS.iter().any(|it| grand_parent.ends_with(it)) {
                    return true;
                }
            }
            false
        }
    }
    pub(crate) fn transition(&mut self, new_status: Status) {
        self.status = new_status;
        if self.config.status_notification() {
            let lsp_status = match new_status {
                Status::Loading => lsp_ext::Status::Loading,
                Status::Ready { partial: true } => lsp_ext::Status::ReadyPartial,
                Status::Ready { partial: false } => lsp_ext::Status::Ready,
                Status::Invalid => lsp_ext::Status::Invalid,
                Status::NeedsReload => lsp_ext::Status::NeedsReload,
            };
            self.send_notification::<lsp_ext::StatusNotification>(StatusParams {
                status: lsp_status,
            });
        }
    }

    pub(crate) fn fetch_build_data_request(&mut self, build_data_collector: BuildDataCollector) {
        self.fetch_build_data_queue.request_op(build_data_collector);
    }

    pub(crate) fn fetch_build_data_if_needed(&mut self) {
        let mut build_data_collector = match self.fetch_build_data_queue.should_start_op() {
            Some(it) => it,
            None => return,
        };
        self.task_pool.handle.spawn_with_sender(move |sender| {
            sender.send(Task::FetchBuildData(BuildDataProgress::Begin)).unwrap();

            let progress = {
                let sender = sender.clone();
                move |msg| {
                    sender.send(Task::FetchBuildData(BuildDataProgress::Report(msg))).unwrap()
                }
            };
            let res = build_data_collector.collect(&progress);
            sender.send(Task::FetchBuildData(BuildDataProgress::End(res))).unwrap();
        });
    }
    pub(crate) fn fetch_build_data_completed(&mut self) {
        self.fetch_build_data_queue.op_completed()
    }

    pub(crate) fn fetch_workspaces_request(&mut self) {
        self.fetch_workspaces_queue.request_op(())
    }
    pub(crate) fn fetch_workspaces_if_needed(&mut self) {
        if self.fetch_workspaces_queue.should_start_op().is_none() {
            return;
        }
        log::info!("will fetch workspaces");

        self.task_pool.handle.spawn_with_sender({
            let linked_projects = self.config.linked_projects();
            let cargo_config = self.config.cargo();

            move |sender| {
                let progress = {
                    let sender = sender.clone();
                    move |msg| {
                        sender
                            .send(Task::FetchWorkspace(ProjectWorkspaceProgress::Report(msg)))
                            .unwrap()
                    }
                };

                sender.send(Task::FetchWorkspace(ProjectWorkspaceProgress::Begin)).unwrap();

                let workspaces = linked_projects
                    .iter()
                    .map(|project| match project {
                        LinkedProject::ProjectManifest(manifest) => {
                            project_model::ProjectWorkspace::load(
                                manifest.clone(),
                                &cargo_config,
                                &progress,
                            )
                        }
                        LinkedProject::InlineJsonProject(it) => {
                            project_model::ProjectWorkspace::load_inline(
                                it.clone(),
                                cargo_config.target.as_deref(),
                            )
                        }
                    })
                    .collect::<Vec<_>>();

                log::info!("did fetch workspaces {:?}", workspaces);
                sender
                    .send(Task::FetchWorkspace(ProjectWorkspaceProgress::End(workspaces)))
                    .unwrap();
            }
        });
    }
    pub(crate) fn fetch_workspaces_completed(&mut self) {
        self.fetch_workspaces_queue.op_completed()
    }

    pub(crate) fn switch_workspaces(
        &mut self,
        workspaces: Vec<anyhow::Result<ProjectWorkspace>>,
        workspace_build_data: Option<anyhow::Result<BuildDataResult>>,
    ) {
        let _p = profile::span("GlobalState::switch_workspaces");
        log::info!("will switch workspaces: {:?}", workspaces);

        let mut has_errors = false;
        let workspaces = workspaces
            .into_iter()
            .filter_map(|res| {
                res.map_err(|err| {
                    has_errors = true;
                    log::error!("failed to load workspace: {:#}", err);
                    if self.workspaces.is_empty() {
                        self.show_message(
                            lsp_types::MessageType::Error,
                            format!("rust-analyzer failed to load workspace: {:#}", err),
                        );
                    }
                })
                .ok()
            })
            .collect::<Vec<_>>();

        let workspace_build_data = match workspace_build_data {
            Some(Ok(it)) => Some(it),
            Some(Err(err)) => {
                log::error!("failed to fetch build data: {:#}", err);
                self.show_message(
                    lsp_types::MessageType::Error,
                    format!("rust-analyzer failed to fetch build data: {:#}", err),
                );
                return;
            }
            None => None,
        };

        if *self.workspaces == workspaces && self.workspace_build_data == workspace_build_data {
            return;
        }

        if !self.workspaces.is_empty() && has_errors {
            return;
        }

        if let FilesWatcher::Client = self.config.files().watcher {
            if self.config.did_change_watched_files_dynamic_registration() {
                let registration_options = lsp_types::DidChangeWatchedFilesRegistrationOptions {
                    watchers: workspaces
                        .iter()
                        .flat_map(|it| it.to_roots(workspace_build_data.as_ref()))
                        .filter(|it| it.is_member)
                        .flat_map(|root| {
                            root.include.into_iter().map(|it| format!("{}/**/*.rs", it.display()))
                        })
                        .map(|glob_pattern| lsp_types::FileSystemWatcher {
                            glob_pattern,
                            kind: None,
                        })
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
        }

        let mut change = Change::new();

        let files_config = self.config.files();
        let project_folders =
            ProjectFolders::new(&workspaces, &files_config.exclude, workspace_build_data.as_ref());

        if self.proc_macro_client.is_none() {
            self.proc_macro_client = match self.config.proc_macro_srv() {
                None => None,
                Some((path, args)) => match ProcMacroClient::extern_process(path.clone(), args) {
                    Ok(it) => Some(it),
                    Err(err) => {
                        log::error!(
                            "Failed to run proc_macro_srv from path {}, error: {:?}",
                            path.display(),
                            err
                        );
                        None
                    }
                },
            };
        }

        let watch = match files_config.watcher {
            FilesWatcher::Client => vec![],
            FilesWatcher::Notify => project_folders.watch,
        };
        self.vfs_config_version += 1;
        self.loader.handle.set_config(vfs::loader::Config {
            load: project_folders.load,
            watch,
            version: self.vfs_config_version,
        });

        // Create crate graph from all the workspaces
        let crate_graph = {
            let mut crate_graph = CrateGraph::default();
            let vfs = &mut self.vfs.write().0;
            let loader = &mut self.loader;
            let mem_docs = &self.mem_docs;
            let mut load = |path: &AbsPath| {
                let vfs_path = vfs::VfsPath::from(path.to_path_buf());
                if !mem_docs.contains_key(&vfs_path) {
                    let contents = loader.handle.load_sync(path);
                    vfs.set_file_contents(vfs_path.clone(), contents);
                }
                let res = vfs.file_id(&vfs_path);
                if res.is_none() {
                    log::warn!("failed to load {}", path.display())
                }
                res
            };
            for ws in workspaces.iter() {
                crate_graph.extend(ws.to_crate_graph(
                    workspace_build_data.as_ref(),
                    self.proc_macro_client.as_ref(),
                    &mut load,
                ));
            }

            crate_graph
        };
        change.set_crate_graph(crate_graph);

        if self.config.run_build_scripts() && workspace_build_data.is_none() {
            let mut collector = BuildDataCollector::default();
            for ws in &workspaces {
                ws.collect_build_data_configs(&mut collector);
            }
            self.fetch_build_data_request(collector)
        }

        self.source_root_config = project_folders.source_root_config;
        self.workspaces = Arc::new(workspaces);
        self.workspace_build_data = workspace_build_data;

        self.analysis_host.apply_change(change);
        self.process_changes();
        self.reload_flycheck();
        log::info!("did switch workspaces");
    }

    fn reload_flycheck(&mut self) {
        let _p = profile::span("GlobalState::reload_flycheck");
        let config = match self.config.flycheck() {
            Some(it) => it,
            None => {
                self.flycheck = Vec::new();
                return;
            }
        };

        let sender = self.flycheck_sender.clone();
        self.flycheck = self
            .workspaces
            .iter()
            .enumerate()
            .filter_map(|(id, w)| match w {
                ProjectWorkspace::Cargo { cargo, .. } => Some((id, cargo.workspace_root())),
                ProjectWorkspace::Json { project, .. } => {
                    // Enable flychecks for json projects if a custom flycheck command was supplied
                    // in the workspace configuration.
                    match config {
                        FlycheckConfig::CustomCommand { .. } => Some((id, project.path())),
                        _ => None,
                    }
                }
            })
            .map(|(id, root)| {
                let sender = sender.clone();
                FlycheckHandle::spawn(
                    id,
                    Box::new(move |msg| sender.send(msg).unwrap()),
                    config.clone(),
                    root.to_path_buf().into(),
                )
            })
            .collect();
    }
}

#[derive(Default)]
pub(crate) struct ProjectFolders {
    pub(crate) load: Vec<vfs::loader::Entry>,
    pub(crate) watch: Vec<usize>,
    pub(crate) source_root_config: SourceRootConfig,
}

impl ProjectFolders {
    pub(crate) fn new(
        workspaces: &[ProjectWorkspace],
        global_excludes: &[AbsPathBuf],
        build_data: Option<&BuildDataResult>,
    ) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        for root in workspaces.iter().flat_map(|it| it.to_roots(build_data)) {
            let file_set_roots: Vec<VfsPath> =
                root.include.iter().cloned().map(VfsPath::from).collect();

            let entry = {
                let mut dirs = vfs::loader::Directories::default();
                dirs.extensions.push("rs".into());
                dirs.include.extend(root.include);
                dirs.exclude.extend(root.exclude);
                for excl in global_excludes {
                    if dirs.include.iter().any(|incl| incl.starts_with(excl)) {
                        dirs.exclude.push(excl.clone());
                    }
                }

                vfs::loader::Entry::Directories(dirs)
            };

            if root.is_member {
                res.watch.push(res.load.len());
            }
            res.load.push(entry);

            if root.is_member {
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
        let _p = profile::span("SourceRootConfig::partition");
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
