//! The context or environment in which the language server functions. In our
//! server implementation this is know as the `WorldState`.
//!
//! Each tick provides an immutable snapshot of the state as `WorldSnapshot`.

use std::{convert::TryFrom, sync::Arc};

use crossbeam_channel::{unbounded, Receiver, Sender};
use flycheck::{FlycheckConfig, FlycheckHandle};
use lsp_types::{request::Request, Url};
use parking_lot::RwLock;
use ra_db::{CrateId, SourceRoot, VfsPath};
use ra_ide::{Analysis, AnalysisChange, AnalysisHost, CrateGraph, FileId};
use ra_project_model::{CargoWorkspace, PackageRoot, ProcMacroClient, ProjectWorkspace, Target};
use stdx::format_to;
use vfs::{file_set::FileSetConfig, loader::Handle, AbsPath, AbsPathBuf};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    diagnostics::{CheckFixes, DiagnosticCollection},
    from_proto,
    line_endings::LineEndings,
    main_loop::{ReqQueue, Task},
    request_metrics::{LatestRequests, RequestMetrics},
    show_message,
    thread_pool::TaskPool,
    to_proto::url_from_abs_path,
    Result,
};
use rustc_hash::{FxHashMap, FxHashSet};

fn create_flycheck(
    workspaces: &[ProjectWorkspace],
    config: &FlycheckConfig,
) -> Option<(FlycheckHandle, Receiver<flycheck::Message>)> {
    // FIXME: Figure out the multi-workspace situation
    workspaces.iter().find_map(move |w| match w {
        ProjectWorkspace::Cargo { cargo, .. } => {
            let (sender, receiver) = unbounded();
            let sender = Box::new(move |msg| sender.send(msg).unwrap());
            let cargo_project_root = cargo.workspace_root().to_path_buf();
            let flycheck = FlycheckHandle::spawn(sender, config.clone(), cargo_project_root.into());
            Some((flycheck, receiver))
        }
        ProjectWorkspace::Json { .. } => {
            log::warn!("Cargo check watching only supported for cargo workspaces, disabling");
            None
        }
    })
}

#[derive(Eq, PartialEq)]
pub(crate) enum Status {
    Loading,
    Ready,
}

impl Default for Status {
    fn default() -> Self {
        Status::Loading
    }
}

/// `GlobalState` is the primary mutable state of the language server
///
/// The most interesting components are `vfs`, which stores a consistent
/// snapshot of the file systems, and `analysis_host`, which stores our
/// incremental salsa database.
pub(crate) struct GlobalState {
    sender: Sender<lsp_server::Message>,
    pub(crate) config: Config,
    pub(crate) task_pool: (TaskPool<Task>, Receiver<Task>),
    pub(crate) analysis_host: AnalysisHost,
    pub(crate) loader: Box<dyn vfs::loader::Handle>,
    pub(crate) task_receiver: Receiver<vfs::loader::Message>,
    pub(crate) flycheck: Option<(FlycheckHandle, Receiver<flycheck::Message>)>,
    pub(crate) diagnostics: DiagnosticCollection,
    pub(crate) mem_docs: FxHashSet<VfsPath>,
    pub(crate) vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
    pub(crate) status: Status,
    pub(crate) req_queue: ReqQueue,
    latest_requests: Arc<RwLock<LatestRequests>>,
    source_root_config: SourceRootConfig,
    proc_macro_client: ProcMacroClient,
    workspaces: Arc<Vec<ProjectWorkspace>>,
}

/// An immutable snapshot of the world's state at a point in time.
pub(crate) struct GlobalStateSnapshot {
    pub(crate) config: Config,
    pub(crate) analysis: Analysis,
    pub(crate) check_fixes: CheckFixes,
    pub(crate) latest_requests: Arc<RwLock<LatestRequests>>,
    vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
    workspaces: Arc<Vec<ProjectWorkspace>>,
}

impl GlobalState {
    pub(crate) fn new(
        sender: Sender<lsp_server::Message>,
        lru_capacity: Option<usize>,
        config: Config,
    ) -> GlobalState {
        let (task_sender, task_receiver) = unbounded::<vfs::loader::Message>();

        let loader = {
            let loader = vfs_notify::NotifyHandle::spawn(Box::new(move |msg| {
                task_sender.send(msg).unwrap()
            }));
            Box::new(loader)
        };

        let task_pool = {
            let (sender, receiver) = unbounded();
            (TaskPool::new(sender), receiver)
        };

        GlobalState {
            sender,
            config,
            task_pool,
            analysis_host: AnalysisHost::new(lru_capacity),
            loader,
            task_receiver,
            flycheck: None,
            diagnostics: Default::default(),
            mem_docs: FxHashSet::default(),
            vfs: Arc::new(RwLock::new((vfs::Vfs::default(), FxHashMap::default()))),
            status: Status::default(),
            req_queue: ReqQueue::default(),
            latest_requests: Default::default(),
            source_root_config: SourceRootConfig::default(),
            proc_macro_client: ProcMacroClient::dummy(),
            workspaces: Arc::new(Vec::new()),
        }
    }

    pub(crate) fn reload(&mut self) {
        let workspaces = {
            if self.config.linked_projects.is_empty()
                && self.config.notifications.cargo_toml_not_found
            {
                self.show_message(
                    lsp_types::MessageType::Error,
                    "rust-analyzer failed to discover workspace".to_string(),
                );
            };

            self.config
                .linked_projects
                .iter()
                .filter_map(|project| match project {
                    LinkedProject::ProjectManifest(manifest) => {
                        ra_project_model::ProjectWorkspace::load(
                            manifest.clone(),
                            &self.config.cargo,
                            self.config.with_sysroot,
                        )
                        .map_err(|err| {
                            log::error!("failed to load workspace: {:#}", err);
                            self.show_message(
                                lsp_types::MessageType::Error,
                                format!("rust-analyzer failed to load workspace: {:#}", err),
                            );
                        })
                        .ok()
                    }
                    LinkedProject::InlineJsonProject(it) => {
                        Some(ra_project_model::ProjectWorkspace::Json { project: it.clone() })
                    }
                })
                .collect::<Vec<_>>()
        };

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
                id: "file-watcher".to_string(),
                method: "workspace/didChangeWatchedFiles".to_string(),
                register_options: Some(serde_json::to_value(registration_options).unwrap()),
            };
            let params = lsp_types::RegistrationParams { registrations: vec![registration] };
            let request = self.req_queue.outgoing.register(
                lsp_types::request::RegisterCapability::METHOD.to_string(),
                params,
                |_, _| (),
            );
            self.send(request.into());
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
        self.loader.set_config(vfs::loader::Config { load: project_folders.load, watch });

        // Create crate graph from all the workspaces
        let crate_graph = {
            let mut crate_graph = CrateGraph::default();
            let vfs = &mut self.vfs.write().0;
            let loader = &mut self.loader;
            let mut load = |path: &AbsPath| {
                let contents = loader.load_sync(path);
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

        self.flycheck = self.config.check.as_ref().and_then(|c| create_flycheck(&workspaces, c));
        self.source_root_config = project_folders.source_root_config;
        self.workspaces = Arc::new(workspaces);

        self.analysis_host.apply_change(change);
        self.process_changes();
    }

    pub(crate) fn update_configuration(&mut self, config: Config) {
        self.analysis_host.update_lru_capacity(config.lru_capacity);
        if config.check != self.config.check {
            self.flycheck =
                config.check.as_ref().and_then(|it| create_flycheck(&self.workspaces, it));
        }

        self.config = config;
    }

    pub(crate) fn process_changes(&mut self) -> bool {
        let change = {
            let mut change = AnalysisChange::new();
            let (vfs, line_endings_map) = &mut *self.vfs.write();
            let changed_files = vfs.take_changes();
            if changed_files.is_empty() {
                return false;
            }

            let fs_op = changed_files.iter().any(|it| it.is_created_or_deleted());
            if fs_op {
                let roots = self.source_root_config.partition(&vfs);
                change.set_roots(roots)
            }

            for file in changed_files {
                let text = if file.exists() {
                    let bytes = vfs.file_contents(file.file_id).to_vec();
                    match String::from_utf8(bytes).ok() {
                        Some(text) => {
                            let (text, line_endings) = LineEndings::normalize(text);
                            line_endings_map.insert(file.file_id, line_endings);
                            Some(Arc::new(text))
                        }
                        None => None,
                    }
                } else {
                    None
                };
                change.change_file(file.file_id, text);
            }
            change
        };

        self.analysis_host.apply_change(change);
        true
    }

    pub(crate) fn snapshot(&self) -> GlobalStateSnapshot {
        GlobalStateSnapshot {
            config: self.config.clone(),
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            vfs: Arc::clone(&self.vfs),
            latest_requests: Arc::clone(&self.latest_requests),
            check_fixes: Arc::clone(&self.diagnostics.check_fixes),
        }
    }

    pub(crate) fn maybe_collect_garbage(&mut self) {
        self.analysis_host.maybe_collect_garbage()
    }

    pub(crate) fn collect_garbage(&mut self) {
        self.analysis_host.collect_garbage()
    }

    pub(crate) fn send(&mut self, message: lsp_server::Message) {
        self.sender.send(message).unwrap()
    }
    pub(crate) fn respond(&mut self, response: lsp_server::Response) {
        if let Some((method, start)) = self.req_queue.incoming.complete(response.id.clone()) {
            let duration = start.elapsed();
            log::info!("handled req#{} in {:?}", response.id, duration);
            let metrics =
                RequestMetrics { id: response.id.clone(), method: method.to_string(), duration };
            self.latest_requests.write().record(metrics);
            self.send(response.into());
        }
    }
    pub(crate) fn show_message(&self, typ: lsp_types::MessageType, message: String) {
        show_message(typ, message, &self.sender)
    }
}

impl Drop for GlobalState {
    fn drop(&mut self) {
        self.analysis_host.request_cancellation()
    }
}

impl GlobalStateSnapshot {
    pub(crate) fn url_to_file_id(&self, url: &Url) -> Result<FileId> {
        let path = from_proto::abs_path(url)?;
        let path = path.into();
        let res =
            self.vfs.read().0.file_id(&path).ok_or_else(|| format!("file not found: {}", path))?;
        Ok(res)
    }

    pub(crate) fn file_id_to_url(&self, id: FileId) -> Url {
        file_id_to_url(&self.vfs.read().0, id)
    }

    pub(crate) fn file_line_endings(&self, id: FileId) -> LineEndings {
        self.vfs.read().1[&id]
    }

    pub(crate) fn anchored_path(&self, file_id: FileId, path: &str) -> Url {
        let mut base = self.vfs.read().0.file_path(file_id);
        base.pop();
        let path = base.join(path);
        let path = path.as_path().unwrap();
        url_from_abs_path(&path)
    }

    pub(crate) fn cargo_target_for_crate_root(
        &self,
        crate_id: CrateId,
    ) -> Option<(&CargoWorkspace, Target)> {
        let file_id = self.analysis.crate_root(crate_id).ok()?;
        let path = self.vfs.read().0.file_path(file_id);
        let path = path.as_path()?;
        self.workspaces.iter().find_map(|ws| match ws {
            ProjectWorkspace::Cargo { cargo, .. } => {
                cargo.target_by_root(&path).map(|it| (cargo, it))
            }
            ProjectWorkspace::Json { .. } => None,
        })
    }

    pub(crate) fn status(&self) -> String {
        let mut buf = String::new();
        if self.workspaces.is_empty() {
            buf.push_str("no workspaces\n")
        } else {
            buf.push_str("workspaces:\n");
            for w in self.workspaces.iter() {
                format_to!(buf, "{} packages loaded\n", w.n_packages());
            }
        }
        buf.push_str("\nanalysis:\n");
        buf.push_str(
            &self
                .analysis
                .status()
                .unwrap_or_else(|_| "Analysis retrieval was cancelled".to_owned()),
        );
        buf
    }
}

pub(crate) fn file_id_to_url(vfs: &vfs::Vfs, id: FileId) -> Url {
    let path = vfs.file_path(id);
    let path = path.as_path().unwrap();
    url_from_abs_path(&path)
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
                let out_dir = AbsPathBuf::try_from(out_dir.to_path_buf()).unwrap();
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
