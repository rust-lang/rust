//! The context or environment in which the language server functions. In our
//! server implementation this is know as the `WorldState`.
//!
//! Each tick provides an immutable snapshot of the state as `WorldSnapshot`.

use std::{sync::Arc, time::Instant};

use crossbeam_channel::{unbounded, Receiver, Sender};
use flycheck::FlycheckHandle;
use lsp_types::Url;
use parking_lot::RwLock;
use ra_db::{CrateId, VfsPath};
use ra_ide::{Analysis, AnalysisChange, AnalysisHost, FileId};
use ra_project_model::{CargoWorkspace, ProcMacroClient, ProjectWorkspace, Target};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    config::Config,
    diagnostics::{CheckFixes, DiagnosticCollection},
    from_proto,
    line_endings::LineEndings,
    main_loop::Task,
    reload::SourceRootConfig,
    request_metrics::{LatestRequests, RequestMetrics},
    thread_pool::TaskPool,
    to_proto::url_from_abs_path,
    Result,
};

#[derive(Eq, PartialEq, Copy, Clone)]
pub(crate) enum Status {
    Loading,
    Ready,
    Invalid,
    NeedsReload,
}

impl Default for Status {
    fn default() -> Self {
        Status::Loading
    }
}

// Enforces drop order
pub(crate) struct Handle<H, C> {
    pub(crate) handle: H,
    pub(crate) receiver: C,
}

pub(crate) type ReqHandler = fn(&mut GlobalState, lsp_server::Response);
pub(crate) type ReqQueue = lsp_server::ReqQueue<(String, Instant), ReqHandler>;

/// `GlobalState` is the primary mutable state of the language server
///
/// The most interesting components are `vfs`, which stores a consistent
/// snapshot of the file systems, and `analysis_host`, which stores our
/// incremental salsa database.
///
/// Note that this struct has more than on impl in various modules!
pub(crate) struct GlobalState {
    sender: Sender<lsp_server::Message>,
    req_queue: ReqQueue,
    pub(crate) task_pool: Handle<TaskPool<Task>, Receiver<Task>>,
    pub(crate) loader: Handle<Box<dyn vfs::loader::Handle>, Receiver<vfs::loader::Message>>,
    pub(crate) flycheck: Option<Handle<FlycheckHandle, Receiver<flycheck::Message>>>,
    pub(crate) config: Config,
    pub(crate) analysis_host: AnalysisHost,
    pub(crate) diagnostics: DiagnosticCollection,
    pub(crate) mem_docs: FxHashSet<VfsPath>,
    pub(crate) vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
    pub(crate) status: Status,
    pub(crate) source_root_config: SourceRootConfig,
    pub(crate) proc_macro_client: ProcMacroClient,
    pub(crate) workspaces: Arc<Vec<ProjectWorkspace>>,
    latest_requests: Arc<RwLock<LatestRequests>>,
}

/// An immutable snapshot of the world's state at a point in time.
pub(crate) struct GlobalStateSnapshot {
    pub(crate) config: Config,
    pub(crate) analysis: Analysis,
    pub(crate) check_fixes: CheckFixes,
    pub(crate) latest_requests: Arc<RwLock<LatestRequests>>,
    vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
    pub(crate) workspaces: Arc<Vec<ProjectWorkspace>>,
}

impl GlobalState {
    pub(crate) fn new(sender: Sender<lsp_server::Message>, config: Config) -> GlobalState {
        let loader = {
            let (sender, receiver) = unbounded::<vfs::loader::Message>();
            let handle: vfs_notify::NotifyHandle =
                vfs::loader::Handle::spawn(Box::new(move |msg| sender.send(msg).unwrap()));
            let handle = Box::new(handle) as Box<dyn vfs::loader::Handle>;
            Handle { handle, receiver }
        };

        let task_pool = {
            let (sender, receiver) = unbounded();
            let handle = TaskPool::new(sender);
            Handle { handle, receiver }
        };

        let analysis_host = AnalysisHost::new(config.lru_capacity);
        GlobalState {
            sender,
            req_queue: ReqQueue::default(),
            task_pool,
            loader,
            flycheck: None,
            config,
            analysis_host,
            diagnostics: Default::default(),
            mem_docs: FxHashSet::default(),
            vfs: Arc::new(RwLock::new((vfs::Vfs::default(), FxHashMap::default()))),
            status: Status::default(),
            source_root_config: SourceRootConfig::default(),
            proc_macro_client: ProcMacroClient::dummy(),
            workspaces: Arc::new(Vec::new()),
            latest_requests: Default::default(),
        }
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

    pub(crate) fn send_request<R: lsp_types::request::Request>(
        &mut self,
        params: R::Params,
        handler: ReqHandler,
    ) {
        let request = self.req_queue.outgoing.register(R::METHOD.to_string(), params, handler);
        self.send(request.into());
    }
    pub(crate) fn complete_request(&mut self, response: lsp_server::Response) {
        let handler = self.req_queue.outgoing.complete(response.id.clone());
        handler(self, response)
    }

    pub(crate) fn send_notification<N: lsp_types::notification::Notification>(
        &mut self,
        params: N::Params,
    ) {
        let not = lsp_server::Notification::new(N::METHOD.to_string(), params);
        self.send(not.into());
    }

    pub(crate) fn register_request(
        &mut self,
        request: &lsp_server::Request,
        request_received: Instant,
    ) {
        self.req_queue
            .incoming
            .register(request.id.clone(), (request.method.clone(), request_received));
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
    pub(crate) fn cancel(&mut self, request_id: lsp_server::RequestId) {
        if let Some(response) = self.req_queue.incoming.cancel(request_id) {
            self.send(response.into());
        }
    }

    fn send(&mut self, message: lsp_server::Message) {
        self.sender.send(message).unwrap()
    }
}

impl Drop for GlobalState {
    fn drop(&mut self) {
        self.analysis_host.request_cancellation()
    }
}

impl GlobalStateSnapshot {
    pub(crate) fn url_to_file_id(&self, url: &Url) -> Result<FileId> {
        url_to_file_id(&self.vfs.read().0, url)
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
        let path = base.join(path).unwrap();
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
}

pub(crate) fn file_id_to_url(vfs: &vfs::Vfs, id: FileId) -> Url {
    let path = vfs.file_path(id);
    let path = path.as_path().unwrap();
    url_from_abs_path(&path)
}

pub(crate) fn url_to_file_id(vfs: &vfs::Vfs, url: &Url) -> Result<FileId> {
    let path = from_proto::vfs_path(url)?;
    let res = vfs.file_id(&path).ok_or_else(|| format!("file not found: {}", path))?;
    Ok(res)
}
