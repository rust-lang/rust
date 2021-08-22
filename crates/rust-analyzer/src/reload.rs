//! Project loading & configuration updates
use std::{mem, sync::Arc};

use flycheck::{FlycheckConfig, FlycheckHandle};
use hir::db::DefDatabase;
use ide::Change;
use ide_db::base_db::{
    CrateGraph, Env, ProcMacro, ProcMacroExpander, ProcMacroKind, SourceRoot, VfsPath,
};
use proc_macro_api::ProcMacroClient;
use project_model::{ProjectWorkspace, WorkspaceBuildScripts};
use vfs::{file_set::FileSetConfig, AbsPath, AbsPathBuf, ChangeKind};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    global_state::GlobalState,
    lsp_ext,
    main_loop::Task,
};

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
    End((Arc<Vec<ProjectWorkspace>>, Vec<anyhow::Result<WorkspaceBuildScripts>>)),
}

impl GlobalState {
    pub(crate) fn is_quiescent(&self) -> bool {
        !(self.fetch_workspaces_queue.op_in_progress()
            || self.fetch_build_data_queue.op_in_progress()
            || self.vfs_progress_config_version < self.vfs_config_version
            || self.vfs_progress_n_done < self.vfs_progress_n_total)
    }

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

        // Apply experimental feature flags.
        self.analysis_host
            .raw_database_mut()
            .set_enable_proc_attr_macros(self.config.expand_proc_attr_macros());
    }
    pub(crate) fn maybe_refresh(&mut self, changes: &[(AbsPathBuf, ChangeKind)]) {
        if !changes.iter().any(|(path, kind)| is_interesting(path, *kind)) {
            return;
        }
        log::info!(
            "Requesting workspace reload because of the following changes: {}",
            itertools::join(
                changes
                    .iter()
                    .filter(|(path, kind)| is_interesting(path, *kind))
                    .map(|(path, kind)| format!("{}: {:?}", path.display(), kind)),
                ", "
            )
        );
        self.fetch_workspaces_request();

        fn is_interesting(path: &AbsPath, change_kind: ChangeKind) -> bool {
            const IMPLICIT_TARGET_FILES: &[&str] = &["build.rs", "src/main.rs", "src/lib.rs"];
            const IMPLICIT_TARGET_DIRS: &[&str] = &["src/bin", "examples", "tests", "benches"];
            let file_name = path.file_name().unwrap_or_default();

            if file_name == "Cargo.toml" || file_name == "Cargo.lock" {
                return true;
            }
            if change_kind == ChangeKind::Modify {
                return false;
            }
            if path.extension().unwrap_or_default() != "rs" {
                return false;
            }
            if IMPLICIT_TARGET_FILES.iter().any(|it| path.as_ref().ends_with(it)) {
                return true;
            }
            let parent = match path.parent() {
                Some(it) => it,
                None => return false,
            };
            if IMPLICIT_TARGET_DIRS.iter().any(|it| parent.as_ref().ends_with(it)) {
                return true;
            }
            if file_name == "main.rs" {
                let grand_parent = match parent.parent() {
                    Some(it) => it,
                    None => return false,
                };
                if IMPLICIT_TARGET_DIRS.iter().any(|it| grand_parent.as_ref().ends_with(it)) {
                    return true;
                }
            }
            false
        }
    }
    pub(crate) fn report_new_status_if_needed(&mut self) {
        let mut status = lsp_ext::ServerStatusParams {
            health: lsp_ext::Health::Ok,
            quiescent: self.is_quiescent(),
            message: None,
        };

        if let Some(error) = self.fetch_build_data_error() {
            status.health = lsp_ext::Health::Warning;
            status.message = Some(error)
        }
        if !self.config.cargo_autoreload()
            && self.is_quiescent()
            && self.fetch_workspaces_queue.op_requested()
        {
            status.health = lsp_ext::Health::Warning;
            status.message = Some("Workspace reload required".to_string())
        }

        if let Some(error) = self.fetch_workspace_error() {
            status.health = lsp_ext::Health::Error;
            status.message = Some(error)
        }

        if self.last_reported_status.as_ref() != Some(&status) {
            self.last_reported_status = Some(status.clone());

            if let (lsp_ext::Health::Error, Some(message)) = (status.health, &status.message) {
                self.show_message(lsp_types::MessageType::Error, message.clone());
            }

            if self.config.server_status_notification() {
                self.send_notification::<lsp_ext::ServerStatusNotification>(status);
            }
        }
    }

    pub(crate) fn fetch_workspaces_request(&mut self) {
        self.fetch_workspaces_queue.request_op()
    }
    pub(crate) fn fetch_workspaces_if_needed(&mut self) {
        if !self.fetch_workspaces_queue.should_start_op() {
            return;
        }
        log::info!("will fetch workspaces");

        self.task_pool.handle.spawn_with_sender({
            let linked_projects = self.config.linked_projects();
            let detached_files = self.config.detached_files().to_vec();
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

                let mut workspaces = linked_projects
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

                if !detached_files.is_empty() {
                    workspaces
                        .push(project_model::ProjectWorkspace::load_detached_files(detached_files));
                }

                log::info!("did fetch workspaces {:?}", workspaces);
                sender
                    .send(Task::FetchWorkspace(ProjectWorkspaceProgress::End(workspaces)))
                    .unwrap();
            }
        });
    }
    pub(crate) fn fetch_workspaces_completed(
        &mut self,
        workspaces: Vec<anyhow::Result<ProjectWorkspace>>,
    ) {
        self.fetch_workspaces_queue.op_completed(workspaces)
    }

    pub(crate) fn fetch_build_data_request(&mut self) {
        self.fetch_build_data_queue.request_op();
    }
    pub(crate) fn fetch_build_data_if_needed(&mut self) {
        if !self.fetch_build_data_queue.should_start_op() {
            return;
        }

        let workspaces = Arc::clone(&self.workspaces);
        let config = self.config.cargo();
        self.task_pool.handle.spawn_with_sender(move |sender| {
            sender.send(Task::FetchBuildData(BuildDataProgress::Begin)).unwrap();

            let progress = {
                let sender = sender.clone();
                move |msg| {
                    sender.send(Task::FetchBuildData(BuildDataProgress::Report(msg))).unwrap()
                }
            };
            let mut res = Vec::new();
            for ws in workspaces.iter() {
                res.push(ws.run_build_scripts(&config, &progress));
                let ws = match ws {
                    ProjectWorkspace::Cargo { cargo, .. } => cargo,
                    ProjectWorkspace::DetachedFiles { .. } | ProjectWorkspace::Json { .. } => {
                        res.push(Ok(WorkspaceBuildScripts::default()));
                        continue;
                    }
                };
                res.push(WorkspaceBuildScripts::run(&config, ws, &progress))
            }
            sender.send(Task::FetchBuildData(BuildDataProgress::End((workspaces, res)))).unwrap();
        });
    }
    pub(crate) fn fetch_build_data_completed(
        &mut self,
        build_data: (Arc<Vec<ProjectWorkspace>>, Vec<anyhow::Result<WorkspaceBuildScripts>>),
    ) {
        self.fetch_build_data_queue.op_completed(build_data)
    }

    pub(crate) fn switch_workspaces(&mut self) {
        let _p = profile::span("GlobalState::switch_workspaces");
        log::info!("will switch workspaces");

        if let Some(error_message) = self.fetch_workspace_error() {
            log::error!("failed to switch workspaces: {}", error_message);
            if !self.workspaces.is_empty() {
                // It only makes sense to switch to a partially broken workspace
                // if we don't have any workspace at all yet.
                return;
            }
        }

        if let Some(error_message) = self.fetch_build_data_error() {
            log::error!("failed to switch build data: {}", error_message);
        }

        let workspaces = self
            .fetch_workspaces_queue
            .last_op_result()
            .iter()
            .filter_map(|res| res.as_ref().ok().cloned())
            .collect::<Vec<_>>();

        fn eq_ignore_build_data<'a>(
            left: &'a ProjectWorkspace,
            right: &'a ProjectWorkspace,
        ) -> bool {
            let key = |p: &'a ProjectWorkspace| match p {
                ProjectWorkspace::Cargo {
                    cargo,
                    sysroot,
                    rustc,
                    rustc_cfg,
                    cfg_overrides,

                    build_scripts: _,
                } => Some((cargo, sysroot, rustc, rustc_cfg, cfg_overrides)),
                _ => None,
            };
            match (key(left), key(right)) {
                (Some(lk), Some(rk)) => lk == rk,
                _ => left == right,
            }
        }

        let same_workspaces = workspaces.len() == self.workspaces.len()
            && workspaces
                .iter()
                .zip(self.workspaces.iter())
                .all(|(l, r)| eq_ignore_build_data(l, r));

        if same_workspaces {
            let (workspaces, build_scripts) = self.fetch_build_data_queue.last_op_result();
            if Arc::ptr_eq(&workspaces, &self.workspaces) {
                let workspaces = workspaces
                    .iter()
                    .cloned()
                    .zip(build_scripts)
                    .map(|(mut ws, bs)| {
                        ws.set_build_scripts(bs.as_ref().ok().cloned().unwrap_or_default());
                        ws
                    })
                    .collect::<Vec<_>>();

                // Workspaces are the same, but we've updated build data.
                self.workspaces = Arc::new(workspaces);
            } else {
                // Current build scripts do not match the version of the active
                // workspace, so there's nothing for us to update.
                return;
            }
        } else {
            // Here, we completely changed the workspace (Cargo.toml edit), so
            // we don't care about build-script results, they are stale.
            self.workspaces = Arc::new(workspaces)
        }

        if let FilesWatcher::Client = self.config.files().watcher {
            if self.config.did_change_watched_files_dynamic_registration() {
                let registration_options = lsp_types::DidChangeWatchedFilesRegistrationOptions {
                    watchers: self
                        .workspaces
                        .iter()
                        .flat_map(|ws| ws.to_roots())
                        .filter(|it| it.is_member)
                        .flat_map(|root| {
                            root.include.into_iter().flat_map(|it| {
                                [
                                    format!("{}/**/*.rs", it.display()),
                                    format!("{}/**/Cargo.toml", it.display()),
                                    format!("{}/**/Cargo.lock", it.display()),
                                ]
                            })
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
        let project_folders = ProjectFolders::new(&self.workspaces, &files_config.exclude);

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
            let proc_macro_client = self.proc_macro_client.as_ref();
            let mut load_proc_macro =
                move |path: &AbsPath| load_proc_macro(proc_macro_client, path);

            let vfs = &mut self.vfs.write().0;
            let loader = &mut self.loader;
            let mem_docs = &self.mem_docs;
            let mut load = move |path: &AbsPath| {
                let _p = profile::span("GlobalState::load");
                let vfs_path = vfs::VfsPath::from(path.to_path_buf());
                if !mem_docs.contains(&vfs_path) {
                    let contents = loader.handle.load_sync(path);
                    vfs.set_file_contents(vfs_path.clone(), contents);
                }
                let res = vfs.file_id(&vfs_path);
                if res.is_none() {
                    log::warn!("failed to load {}", path.display())
                }
                res
            };

            let mut crate_graph = CrateGraph::default();
            for ws in self.workspaces.iter() {
                crate_graph.extend(ws.to_crate_graph(&mut load_proc_macro, &mut load));
            }
            crate_graph
        };
        change.set_crate_graph(crate_graph);

        self.source_root_config = project_folders.source_root_config;

        self.analysis_host.apply_change(change);
        self.process_changes();
        self.reload_flycheck();
        log::info!("did switch workspaces");
    }

    fn fetch_workspace_error(&self) -> Option<String> {
        let mut buf = String::new();

        for ws in self.fetch_workspaces_queue.last_op_result() {
            if let Err(err) = ws {
                stdx::format_to!(buf, "rust-analyzer failed to load workspace: {:#}\n", err);
            }
        }

        if buf.is_empty() {
            return None;
        }

        Some(buf)
    }

    fn fetch_build_data_error(&self) -> Option<String> {
        let mut buf = String::new();

        for ws in &self.fetch_build_data_queue.last_op_result().1 {
            if let Err(err) = ws {
                stdx::format_to!(buf, "rust-analyzer failed to run custom build: {:#}\n", err);
            }
        }

        if buf.is_empty() {
            return None;
        }

        Some(buf)
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
                ProjectWorkspace::DetachedFiles { .. } => None,
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
    ) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        for root in workspaces.iter().flat_map(|ws| ws.to_roots()) {
            let file_set_roots: Vec<VfsPath> =
                root.include.iter().cloned().map(VfsPath::from).collect();

            let entry = {
                let mut dirs = vfs::loader::Directories::default();
                dirs.extensions.push("rs".into());
                dirs.include.extend(root.include);
                dirs.exclude.extend(root.exclude);
                for excl in global_excludes {
                    if dirs
                        .include
                        .iter()
                        .any(|incl| incl.starts_with(excl) || excl.starts_with(incl))
                    {
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

pub(crate) fn load_proc_macro(client: Option<&ProcMacroClient>, path: &AbsPath) -> Vec<ProcMacro> {
    return client
        .map(|it| it.by_dylib_path(path))
        .unwrap_or_default()
        .into_iter()
        .map(expander_to_proc_macro)
        .collect();

    fn expander_to_proc_macro(expander: proc_macro_api::ProcMacroProcessExpander) -> ProcMacro {
        let name = expander.name().into();
        let kind = match expander.kind() {
            proc_macro_api::ProcMacroKind::CustomDerive => ProcMacroKind::CustomDerive,
            proc_macro_api::ProcMacroKind::FuncLike => ProcMacroKind::FuncLike,
            proc_macro_api::ProcMacroKind::Attr => ProcMacroKind::Attr,
        };
        let expander = Arc::new(Expander(expander));
        ProcMacro { name, kind, expander }
    }

    #[derive(Debug)]
    struct Expander(proc_macro_api::ProcMacroProcessExpander);

    impl ProcMacroExpander for Expander {
        fn expand(
            &self,
            subtree: &tt::Subtree,
            attrs: Option<&tt::Subtree>,
            env: &Env,
        ) -> Result<tt::Subtree, tt::ExpansionError> {
            let env = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            self.0.expand(subtree, attrs, env)
        }
    }
}
