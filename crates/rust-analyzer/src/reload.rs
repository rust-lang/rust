//! Project loading & configuration updates
use std::{mem, sync::Arc};

use base_db::{CrateGraph, SourceRoot, VfsPath};
use flycheck::FlycheckHandle;
use ide::AnalysisChange;
use project_model::{ProcMacroClient, ProjectWorkspace};
use vfs::{file_set::FileSetConfig, AbsPath, AbsPathBuf, ChangeKind};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    global_state::{GlobalState, Status},
    lsp_ext,
    main_loop::Task,
};
use lsp_ext::StatusParams;

impl GlobalState {
    pub(crate) fn update_configuration(&mut self, config: Config) {
        let _p = profile::span("GlobalState::update_configuration");
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
    pub(crate) fn maybe_refresh(&mut self, changes: &[(AbsPathBuf, ChangeKind)]) {
        if !changes.iter().any(|(path, kind)| is_interesting(path, *kind)) {
            return;
        }
        match self.status {
            Status::Loading | Status::NeedsReload => return,
            Status::Ready | Status::Invalid => (),
        }
        if self.config.cargo_autoreload {
            self.fetch_workspaces();
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
        if self.config.client_caps.status_notification {
            let lsp_status = match new_status {
                Status::Loading => lsp_ext::Status::Loading,
                Status::Ready => lsp_ext::Status::Ready,
                Status::Invalid => lsp_ext::Status::Invalid,
                Status::NeedsReload => lsp_ext::Status::NeedsReload,
            };
            self.send_notification::<lsp_ext::StatusNotification>(StatusParams {
                status: lsp_status,
            });
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
                            project_model::ProjectWorkspace::load(
                                manifest.clone(),
                                &cargo_config,
                                with_sysroot,
                            )
                        }
                        LinkedProject::InlineJsonProject(it) => {
                            Ok(project_model::ProjectWorkspace::Json { project: it.clone() })
                        }
                    })
                    .collect::<Vec<_>>();
                Task::Workspaces(workspaces)
            }
        });
    }
    pub(crate) fn switch_workspaces(&mut self, workspaces: Vec<anyhow::Result<ProjectWorkspace>>) {
        let _p = profile::span("GlobalState::switch_workspaces");
        log::info!("reloading projects: {:?}", self.config.linked_projects);

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

        if &*self.workspaces == &workspaces {
            return;
        }

        if !self.workspaces.is_empty() && has_errors {
            return;
        }

        if let FilesWatcher::Client = self.config.files.watcher {
            let registration_options = lsp_types::DidChangeWatchedFilesRegistrationOptions {
                watchers: workspaces
                    .iter()
                    .flat_map(ProjectWorkspace::to_roots)
                    .filter(|it| it.is_member)
                    .flat_map(|root| {
                        root.include.into_iter().map(|it| format!("{}/**/*.rs", it.display()))
                    })
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
                        "Failed to run proc_macro_srv from path {}, error: {:?}",
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

        let sender = self.flycheck_sender.clone();
        let sender = Box::new(move |msg| sender.send(msg).unwrap());
        self.flycheck = self
            .workspaces
            .iter()
            // FIXME: Figure out the multi-workspace situation
            .find_map(|w| match w {
                ProjectWorkspace::Cargo { cargo, sysroot: _ } => Some(cargo),
                ProjectWorkspace::Json { .. } => None,
            })
            .map(move |cargo| {
                let cargo_project_root = cargo.workspace_root().to_path_buf();
                FlycheckHandle::spawn(sender, config, cargo_project_root.into())
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
            let file_set_roots: Vec<VfsPath> =
                root.include.iter().cloned().map(VfsPath::from).collect();

            let entry = {
                let mut dirs = vfs::loader::Directories::default();
                dirs.extensions.push("rs".into());
                dirs.include.extend(root.include);
                dirs.exclude.extend(root.exclude);
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
