//! The context or environment in which the language server functions. In our
//! server implementation this is know as the `WorldState`.
//!
//! Each tick provides an immutable snapshot of the state as `WorldSnapshot`.

use std::{
    ops::Not as _,
    panic::AssertUnwindSafe,
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, unbounded};
use hir::ChangeWithProcMacros;
use ide::{Analysis, AnalysisHost, Cancellable, FileId, SourceRootId};
use ide_db::base_db::{Crate, ProcMacroPaths, SourceDatabase};
use itertools::Itertools;
use load_cargo::SourceRootConfig;
use lsp_types::{SemanticTokens, Url};
use parking_lot::{
    MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard,
    RwLockWriteGuard,
};
use proc_macro_api::ProcMacroClient;
use project_model::{ManifestPath, ProjectWorkspace, ProjectWorkspaceKind, WorkspaceBuildScripts};
use rustc_hash::{FxHashMap, FxHashSet};
use stdx::thread;
use tracing::{Level, span, trace};
use triomphe::Arc;
use vfs::{AbsPathBuf, AnchoredPathBuf, ChangeKind, Vfs, VfsPath};

use crate::{
    config::{Config, ConfigChange, ConfigErrors, RatomlFileKind},
    diagnostics::{CheckFixes, DiagnosticCollection},
    discover,
    flycheck::{FlycheckHandle, FlycheckMessage},
    line_index::{LineEndings, LineIndex},
    lsp::{from_proto, to_proto::url_from_abs_path},
    lsp_ext,
    main_loop::Task,
    mem_docs::MemDocs,
    op_queue::{Cause, OpQueue},
    reload,
    target_spec::{CargoTargetSpec, ProjectJsonTargetSpec, TargetSpec},
    task_pool::{TaskPool, TaskQueue},
    test_runner::{CargoTestHandle, CargoTestMessage},
};

#[derive(Debug)]
pub(crate) struct FetchWorkspaceRequest {
    pub(crate) path: Option<AbsPathBuf>,
    pub(crate) force_crate_graph_reload: bool,
}

pub(crate) struct FetchWorkspaceResponse {
    pub(crate) workspaces: Vec<anyhow::Result<ProjectWorkspace>>,
    pub(crate) force_crate_graph_reload: bool,
}

pub(crate) struct FetchBuildDataResponse {
    pub(crate) workspaces: Arc<Vec<ProjectWorkspace>>,
    pub(crate) build_scripts: Vec<anyhow::Result<WorkspaceBuildScripts>>,
}

// Enforces drop order
pub(crate) struct Handle<H, C> {
    pub(crate) handle: H,
    pub(crate) receiver: C,
}

pub(crate) type ReqHandler = fn(&mut GlobalState, lsp_server::Response);
type ReqQueue = lsp_server::ReqQueue<(String, Instant), ReqHandler>;

/// `GlobalState` is the primary mutable state of the language server
///
/// The most interesting components are `vfs`, which stores a consistent
/// snapshot of the file systems, and `analysis_host`, which stores our
/// incremental salsa database.
///
/// Note that this struct has more than one impl in various modules!
#[doc(alias = "GlobalMess")]
pub(crate) struct GlobalState {
    sender: Sender<lsp_server::Message>,
    req_queue: ReqQueue,

    pub(crate) task_pool: Handle<TaskPool<Task>, Receiver<Task>>,
    pub(crate) fmt_pool: Handle<TaskPool<Task>, Receiver<Task>>,
    pub(crate) cancellation_pool: thread::Pool,

    pub(crate) config: Arc<Config>,
    pub(crate) config_errors: Option<ConfigErrors>,
    pub(crate) analysis_host: AnalysisHost,
    pub(crate) diagnostics: DiagnosticCollection,
    pub(crate) mem_docs: MemDocs,
    pub(crate) source_root_config: SourceRootConfig,
    /// A mapping that maps a local source root's `SourceRootId` to it parent's `SourceRootId`, if it has one.
    pub(crate) local_roots_parent_map: Arc<FxHashMap<SourceRootId, SourceRootId>>,
    pub(crate) semantic_tokens_cache: Arc<Mutex<FxHashMap<Url, SemanticTokens>>>,

    // status
    pub(crate) shutdown_requested: bool,
    pub(crate) last_reported_status: lsp_ext::ServerStatusParams,

    // proc macros
    pub(crate) proc_macro_clients: Arc<[Option<anyhow::Result<ProcMacroClient>>]>,
    pub(crate) build_deps_changed: bool,

    // Flycheck
    pub(crate) flycheck: Arc<[FlycheckHandle]>,
    pub(crate) flycheck_sender: Sender<FlycheckMessage>,
    pub(crate) flycheck_receiver: Receiver<FlycheckMessage>,
    pub(crate) last_flycheck_error: Option<String>,

    // Test explorer
    pub(crate) test_run_session: Option<Vec<CargoTestHandle>>,
    pub(crate) test_run_sender: Sender<CargoTestMessage>,
    pub(crate) test_run_receiver: Receiver<CargoTestMessage>,
    pub(crate) test_run_remaining_jobs: usize,

    // Project loading
    pub(crate) discover_handle: Option<discover::DiscoverHandle>,
    pub(crate) discover_sender: Sender<discover::DiscoverProjectMessage>,
    pub(crate) discover_receiver: Receiver<discover::DiscoverProjectMessage>,

    // Debouncing channel for fetching the workspace
    // we want to delay it until the VFS looks stable-ish (and thus is not currently in the middle
    // of a VCS operation like `git switch`)
    pub(crate) fetch_ws_receiver: Option<(Receiver<Instant>, FetchWorkspaceRequest)>,

    // VFS
    pub(crate) loader: Handle<Box<dyn vfs::loader::Handle>, Receiver<vfs::loader::Message>>,
    pub(crate) vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
    pub(crate) vfs_config_version: u32,
    pub(crate) vfs_progress_config_version: u32,
    pub(crate) vfs_done: bool,
    // used to track how long VFS loading takes. this can't be on `vfs::loader::Handle`,
    // as that handle's lifetime is the same as `GlobalState` itself.
    pub(crate) vfs_span: Option<tracing::span::EnteredSpan>,
    pub(crate) wants_to_switch: Option<Cause>,

    /// `workspaces` field stores the data we actually use, while the `OpQueue`
    /// stores the result of the last fetch.
    ///
    /// If the fetch (partially) fails, we do not update the current value.
    ///
    /// The handling of build data is subtle. We fetch workspace in two phases:
    ///
    /// *First*, we run `cargo metadata`, which gives us fast results for
    /// initial analysis.
    ///
    /// *Second*, we run `cargo check` which runs build scripts and compiles
    /// proc macros.
    ///
    /// We need both for the precise analysis, but we want rust-analyzer to be
    /// at least partially available just after the first phase. That's because
    /// first phase is much faster, and is much less likely to fail.
    ///
    /// This creates a complication -- by the time the second phase completes,
    /// the results of the first phase could be invalid. That is, while we run
    /// `cargo check`, the user edits `Cargo.toml`, we notice this, and the new
    /// `cargo metadata` completes before `cargo check`.
    ///
    /// An additional complication is that we want to avoid needless work. When
    /// the user just adds comments or whitespace to Cargo.toml, we do not want
    /// to invalidate any salsa caches.
    pub(crate) workspaces: Arc<Vec<ProjectWorkspace>>,
    pub(crate) crate_graph_file_dependencies: FxHashSet<vfs::VfsPath>,
    pub(crate) detached_files: FxHashSet<ManifestPath>,

    // op queues
    pub(crate) fetch_workspaces_queue: OpQueue<FetchWorkspaceRequest, FetchWorkspaceResponse>,
    pub(crate) fetch_build_data_queue: OpQueue<(), FetchBuildDataResponse>,
    pub(crate) fetch_proc_macros_queue: OpQueue<(ChangeWithProcMacros, Vec<ProcMacroPaths>), bool>,
    pub(crate) prime_caches_queue: OpQueue,
    pub(crate) discover_workspace_queue: OpQueue,

    /// A deferred task queue.
    ///
    /// This queue is used for doing database-dependent work inside of sync
    /// handlers, as accessing the database may block latency-sensitive
    /// interactions and should be moved away from the main thread.
    ///
    /// For certain features, such as [`GlobalState::handle_discover_msg`],
    /// this queue should run only *after* [`GlobalState::process_changes`] has
    /// been called.
    pub(crate) deferred_task_queue: TaskQueue,
}

/// An immutable snapshot of the world's state at a point in time.
pub(crate) struct GlobalStateSnapshot {
    pub(crate) config: Arc<Config>,
    pub(crate) analysis: Analysis,
    pub(crate) check_fixes: CheckFixes,
    mem_docs: MemDocs,
    pub(crate) semantic_tokens_cache: Arc<Mutex<FxHashMap<Url, SemanticTokens>>>,
    vfs: Arc<RwLock<(vfs::Vfs, FxHashMap<FileId, LineEndings>)>>,
    pub(crate) workspaces: Arc<Vec<ProjectWorkspace>>,
    // used to signal semantic highlighting to fall back to syntax based highlighting until
    // proc-macros have been loaded
    // FIXME: Can we derive this from somewhere else?
    pub(crate) proc_macros_loaded: bool,
    pub(crate) flycheck: Arc<[FlycheckHandle]>,
}

impl std::panic::UnwindSafe for GlobalStateSnapshot {}

impl GlobalState {
    pub(crate) fn new(sender: Sender<lsp_server::Message>, config: Config) -> GlobalState {
        let loader = {
            let (sender, receiver) = unbounded::<vfs::loader::Message>();
            let handle: vfs_notify::NotifyHandle = vfs::loader::Handle::spawn(sender);
            let handle = Box::new(handle) as Box<dyn vfs::loader::Handle>;
            Handle { handle, receiver }
        };

        let task_pool = {
            let (sender, receiver) = unbounded();
            let handle = TaskPool::new_with_threads(sender, config.main_loop_num_threads());
            Handle { handle, receiver }
        };
        let fmt_pool = {
            let (sender, receiver) = unbounded();
            let handle = TaskPool::new_with_threads(sender, 1);
            Handle { handle, receiver }
        };
        let cancellation_pool = thread::Pool::new(1);

        let task_queue = {
            let (sender, receiver) = unbounded();
            TaskQueue { sender, receiver }
        };

        let mut analysis_host = AnalysisHost::new(config.lru_parse_query_capacity());
        if let Some(capacities) = config.lru_query_capacities_config() {
            analysis_host.update_lru_capacities(capacities);
        }
        let (flycheck_sender, flycheck_receiver) = unbounded();
        let (test_run_sender, test_run_receiver) = unbounded();

        let (discover_sender, discover_receiver) = unbounded();

        let mut this = GlobalState {
            sender,
            req_queue: ReqQueue::default(),
            task_pool,
            fmt_pool,
            cancellation_pool,
            loader,
            config: Arc::new(config.clone()),
            analysis_host,
            diagnostics: Default::default(),
            mem_docs: MemDocs::default(),
            semantic_tokens_cache: Arc::new(Default::default()),
            shutdown_requested: false,
            last_reported_status: lsp_ext::ServerStatusParams {
                health: lsp_ext::Health::Ok,
                quiescent: true,
                message: None,
            },
            source_root_config: SourceRootConfig::default(),
            local_roots_parent_map: Arc::new(FxHashMap::default()),
            config_errors: Default::default(),

            proc_macro_clients: Arc::from_iter([]),

            build_deps_changed: false,

            flycheck: Arc::from_iter([]),
            flycheck_sender,
            flycheck_receiver,
            last_flycheck_error: None,

            test_run_session: None,
            test_run_sender,
            test_run_receiver,
            test_run_remaining_jobs: 0,

            discover_handle: None,
            discover_sender,
            discover_receiver,

            fetch_ws_receiver: None,

            vfs: Arc::new(RwLock::new((vfs::Vfs::default(), Default::default()))),
            vfs_config_version: 0,
            vfs_progress_config_version: 0,
            vfs_span: None,
            vfs_done: true,
            wants_to_switch: None,

            workspaces: Arc::from(Vec::new()),
            crate_graph_file_dependencies: FxHashSet::default(),
            detached_files: FxHashSet::default(),
            fetch_workspaces_queue: OpQueue::default(),
            fetch_build_data_queue: OpQueue::default(),
            fetch_proc_macros_queue: OpQueue::default(),

            prime_caches_queue: OpQueue::default(),
            discover_workspace_queue: OpQueue::default(),

            deferred_task_queue: task_queue,
        };
        // Apply any required database inputs from the config.
        this.update_configuration(config);
        this
    }

    pub(crate) fn process_changes(&mut self) -> bool {
        let _p = span!(Level::INFO, "GlobalState::process_changes").entered();
        // We cannot directly resolve a change in a ratoml file to a format
        // that can be used by the config module because config talks
        // in `SourceRootId`s instead of `FileId`s and `FileId` -> `SourceRootId`
        // mapping is not ready until `AnalysisHost::apply_changes` has been called.
        let mut modified_ratoml_files: FxHashMap<FileId, (ChangeKind, vfs::VfsPath)> =
            FxHashMap::default();

        let mut change = ChangeWithProcMacros::default();
        let mut guard = self.vfs.write();
        let changed_files = guard.0.take_changes();
        if changed_files.is_empty() {
            return false;
        }

        let (change, modified_rust_files, workspace_structure_change) =
            self.cancellation_pool.scoped(|s| {
                // start cancellation in parallel, this will kick off lru eviction
                // allowing us to do meaningful work while waiting
                let analysis_host = AssertUnwindSafe(&mut self.analysis_host);
                s.spawn(thread::ThreadIntent::LatencySensitive, || {
                    { analysis_host }.0.request_cancellation()
                });

                // downgrade to read lock to allow more readers while we are normalizing text
                let guard = RwLockWriteGuard::downgrade_to_upgradable(guard);
                let vfs: &Vfs = &guard.0;

                let mut workspace_structure_change = None;
                // A file was added or deleted
                let mut has_structure_changes = false;
                let mut bytes = vec![];
                let mut modified_rust_files = vec![];
                for file in changed_files.into_values() {
                    let vfs_path = vfs.file_path(file.file_id);
                    if let Some(("rust-analyzer", Some("toml"))) = vfs_path.name_and_extension() {
                        // Remember ids to use them after `apply_changes`
                        modified_ratoml_files.insert(file.file_id, (file.kind(), vfs_path.clone()));
                    }

                    if let Some(path) = vfs_path.as_path() {
                        has_structure_changes |= file.is_created_or_deleted();

                        if file.is_modified() && path.extension() == Some("rs") {
                            modified_rust_files.push(file.file_id);
                        }

                        let additional_files = self
                            .config
                            .discover_workspace_config()
                            .map(|cfg| {
                                cfg.files_to_watch.iter().map(String::as_str).collect::<Vec<&str>>()
                            })
                            .unwrap_or_default();

                        let path = path.to_path_buf();
                        if file.is_created_or_deleted() {
                            workspace_structure_change.get_or_insert((path, false)).1 |=
                                self.crate_graph_file_dependencies.contains(vfs_path);
                        } else if reload::should_refresh_for_change(
                            &path,
                            file.kind(),
                            &additional_files,
                        ) {
                            trace!(?path, kind = ?file.kind(), "refreshing for a change");
                            workspace_structure_change.get_or_insert((path.clone(), false));
                        }
                    }

                    // Clear native diagnostics when their file gets deleted
                    if !file.exists() {
                        self.diagnostics.clear_native_for(file.file_id);
                    }

                    let text = if let vfs::Change::Create(v, _) | vfs::Change::Modify(v, _) =
                        file.change
                    {
                        String::from_utf8(v).ok().map(|text| {
                            // FIXME: Consider doing normalization in the `vfs` instead? That allows
                            // getting rid of some locking
                            let (text, line_endings) = LineEndings::normalize(text);
                            (text, line_endings)
                        })
                    } else {
                        None
                    };
                    // delay `line_endings_map` changes until we are done normalizing the text
                    // this allows delaying the re-acquisition of the write lock
                    bytes.push((file.file_id, text));
                }
                let (vfs, line_endings_map) = &mut *RwLockUpgradableReadGuard::upgrade(guard);
                bytes.into_iter().for_each(|(file_id, text)| {
                    let text = match text {
                        None => None,
                        Some((text, line_endings)) => {
                            line_endings_map.insert(file_id, line_endings);
                            Some(text)
                        }
                    };
                    change.change_file(file_id, text);
                });
                if has_structure_changes {
                    let roots = self.source_root_config.partition(vfs);
                    change.set_roots(roots);
                }
                (change, modified_rust_files, workspace_structure_change)
            });

        self.analysis_host.apply_change(change);
        if !modified_ratoml_files.is_empty()
            || !self.config.same_source_root_parent_map(&self.local_roots_parent_map)
        {
            let config_change = {
                let _p = span!(Level::INFO, "GlobalState::process_changes/config_change").entered();
                let user_config_path = (|| {
                    let mut p = Config::user_config_dir_path()?;
                    p.push("rust-analyzer.toml");
                    Some(p)
                })();

                let user_config_abs_path = user_config_path.as_deref();

                let mut change = ConfigChange::default();
                let db = self.analysis_host.raw_database();

                // FIXME @alibektas : This is silly. There is no reason to use VfsPaths when there is SourceRoots. But how
                // do I resolve a "workspace_root" to its corresponding id without having to rely on a cargo.toml's ( or project json etc.) file id?
                let workspace_ratoml_paths = self
                    .workspaces
                    .iter()
                    .map(|ws| {
                        VfsPath::from({
                            let mut p = ws.workspace_root().to_owned();
                            p.push("rust-analyzer.toml");
                            p
                        })
                    })
                    .collect_vec();

                for (file_id, (change_kind, vfs_path)) in modified_ratoml_files {
                    tracing::info!(%vfs_path, ?change_kind, "Processing rust-analyzer.toml changes");
                    if vfs_path.as_path() == user_config_abs_path {
                        tracing::info!(%vfs_path, ?change_kind, "Use config rust-analyzer.toml changes");
                        change.change_user_config(Some(db.file_text(file_id).text(db).clone()));
                    }

                    // If change has been made to a ratoml file that
                    // belongs to a non-local source root, we will ignore it.
                    let source_root_id = db.file_source_root(file_id).source_root_id(db);
                    let source_root = db.source_root(source_root_id).source_root(db);

                    if !source_root.is_library {
                        let entry = if workspace_ratoml_paths.contains(&vfs_path) {
                            tracing::info!(%vfs_path, ?source_root_id, "workspace rust-analyzer.toml changes");
                            change.change_workspace_ratoml(
                                source_root_id,
                                vfs_path.clone(),
                                Some(db.file_text(file_id).text(db).clone()),
                            )
                        } else {
                            tracing::info!(%vfs_path, ?source_root_id, "crate rust-analyzer.toml changes");
                            change.change_ratoml(
                                source_root_id,
                                vfs_path.clone(),
                                Some(db.file_text(file_id).text(db).clone()),
                            )
                        };

                        if let Some((kind, old_path, old_text)) = entry {
                            // SourceRoot has more than 1 RATOML files. In this case lexicographically smaller wins.
                            if old_path < vfs_path {
                                tracing::error!(
                                    "Two `rust-analyzer.toml` files were found inside the same crate. {vfs_path} has no effect."
                                );
                                // Put the old one back in.
                                match kind {
                                    RatomlFileKind::Crate => {
                                        change.change_ratoml(source_root_id, old_path, old_text);
                                    }
                                    RatomlFileKind::Workspace => {
                                        change.change_workspace_ratoml(
                                            source_root_id,
                                            old_path,
                                            old_text,
                                        );
                                    }
                                }
                            }
                        }
                    } else {
                        tracing::info!(%vfs_path, "Ignoring library rust-analyzer.toml");
                    }
                }
                change.change_source_root_parent_map(self.local_roots_parent_map.clone());
                change
            };

            let (config, e, should_update) = self.config.apply_change(config_change);
            self.config_errors = e.is_empty().not().then_some(e);

            if should_update {
                self.update_configuration(config);
            } else {
                // No global or client level config was changed. So we can naively replace config.
                self.config = Arc::new(config);
            }
        }

        // FIXME: `workspace_structure_change` is computed from `should_refresh_for_change` which is
        // path syntax based. That is not sufficient for all cases so we should lift that check out
        // into a `QueuedTask`, see `handle_did_save_text_document`.
        // Or maybe instead of replacing that check, kick off a semantic one if the syntactic one
        // didn't find anything (to make up for the lack of precision).
        {
            if !matches!(&workspace_structure_change, Some((.., true))) {
                _ = self
                    .deferred_task_queue
                    .sender
                    .send(crate::main_loop::QueuedTask::CheckProcMacroSources(modified_rust_files));
            }
            // FIXME: ideally we should only trigger a workspace fetch for non-library changes
            // but something's going wrong with the source root business when we add a new local
            // crate see https://github.com/rust-lang/rust-analyzer/issues/13029
            if let Some((path, force_crate_graph_reload)) = workspace_structure_change {
                let _p = span!(Level::INFO, "GlobalState::process_changes/ws_structure_change")
                    .entered();
                self.enqueue_workspace_fetch(path, force_crate_graph_reload);
            }
        }

        true
    }

    pub(crate) fn snapshot(&self) -> GlobalStateSnapshot {
        GlobalStateSnapshot {
            config: Arc::clone(&self.config),
            workspaces: Arc::clone(&self.workspaces),
            analysis: self.analysis_host.analysis(),
            vfs: Arc::clone(&self.vfs),
            check_fixes: Arc::clone(&self.diagnostics.check_fixes),
            mem_docs: self.mem_docs.clone(),
            semantic_tokens_cache: Arc::clone(&self.semantic_tokens_cache),
            proc_macros_loaded: !self.config.expand_proc_macros()
                || self.fetch_proc_macros_queue.last_op_result().copied().unwrap_or(false),
            flycheck: self.flycheck.clone(),
        }
    }

    pub(crate) fn send_request<R: lsp_types::request::Request>(
        &mut self,
        params: R::Params,
        handler: ReqHandler,
    ) {
        let request = self.req_queue.outgoing.register(R::METHOD.to_owned(), params, handler);
        self.send(request.into());
    }

    pub(crate) fn complete_request(&mut self, response: lsp_server::Response) {
        let handler = self
            .req_queue
            .outgoing
            .complete(response.id.clone())
            .expect("received response for unknown request");
        handler(self, response)
    }

    pub(crate) fn send_notification<N: lsp_types::notification::Notification>(
        &self,
        params: N::Params,
    ) {
        let not = lsp_server::Notification::new(N::METHOD.to_owned(), params);
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
        if let Some((method, start)) = self.req_queue.incoming.complete(&response.id) {
            if let Some(err) = &response.error
                && err.message.starts_with("server panicked")
            {
                self.poke_rust_analyzer_developer(format!("{}, check the log", err.message));
            }

            let duration = start.elapsed();
            tracing::debug!(name: "message response", method, %response.id, duration = format_args!("{:0.2?}", duration));
            self.send(response.into());
        }
    }

    pub(crate) fn cancel(&mut self, request_id: lsp_server::RequestId) {
        if let Some(response) = self.req_queue.incoming.cancel(request_id) {
            self.send(response.into());
        }
    }

    pub(crate) fn is_completed(&self, request: &lsp_server::Request) -> bool {
        self.req_queue.incoming.is_completed(&request.id)
    }

    #[track_caller]
    fn send(&self, message: lsp_server::Message) {
        self.sender.send(message).unwrap();
    }

    pub(crate) fn publish_diagnostics(
        &mut self,
        uri: Url,
        version: Option<i32>,
        mut diagnostics: Vec<lsp_types::Diagnostic>,
    ) {
        // We put this on a separate thread to avoid blocking the main thread with serialization work
        self.task_pool.handle.spawn_with_sender(stdx::thread::ThreadIntent::Worker, {
            let sender = self.sender.clone();
            move |_| {
                // VSCode assumes diagnostic messages to be non-empty strings, so we need to patch
                // empty diagnostics. Neither the docs of VSCode nor the LSP spec say whether
                // diagnostic messages are actually allowed to be empty or not and patching this
                // in the VSCode client does not work as the assertion happens in the protocol
                // conversion. So this hack is here to stay, and will be considered a hack
                // until the LSP decides to state that empty messages are allowed.

                // See https://github.com/rust-lang/rust-analyzer/issues/11404
                // See https://github.com/rust-lang/rust-analyzer/issues/13130
                let patch_empty = |message: &mut String| {
                    if message.is_empty() {
                        " ".clone_into(message);
                    }
                };

                for d in &mut diagnostics {
                    patch_empty(&mut d.message);
                    if let Some(dri) = &mut d.related_information {
                        for dri in dri {
                            patch_empty(&mut dri.message);
                        }
                    }
                }

                let not = lsp_server::Notification::new(
                    <lsp_types::notification::PublishDiagnostics as lsp_types::notification::Notification>::METHOD.to_owned(),
                    lsp_types::PublishDiagnosticsParams { uri, diagnostics, version },
                );
                _ = sender.send(not.into());
            }
        });
    }

    pub(crate) fn check_workspaces_msrv(&self) -> impl Iterator<Item = String> + '_ {
        self.workspaces.iter().filter_map(|ws| {
            if let Some(toolchain) = &ws.toolchain
                && *toolchain < crate::MINIMUM_SUPPORTED_TOOLCHAIN_VERSION
            {
                return Some(format!(
                    "Workspace `{}` is using an outdated toolchain version `{}` but \
                        rust-analyzer only supports `{}` and higher.\n\
                        Consider using the rust-analyzer rustup component for your toolchain or
                        upgrade your toolchain to a supported version.\n\n",
                    ws.manifest_or_root(),
                    toolchain,
                    crate::MINIMUM_SUPPORTED_TOOLCHAIN_VERSION,
                ));
            }
            None
        })
    }

    fn enqueue_workspace_fetch(&mut self, path: AbsPathBuf, force_crate_graph_reload: bool) {
        let already_requested = self.fetch_workspaces_queue.op_requested()
            && !self.fetch_workspaces_queue.op_in_progress();
        if self.fetch_ws_receiver.is_none() && already_requested {
            // Don't queue up a new fetch request if we already have done so
            // Otherwise we will re-fetch in quick succession which is unnecessary
            // Note though, that if one is already in progress, we *want* to re-queue
            // as the in-progress fetch might not have the latest changes in it anymore
            // FIXME: We should cancel the in-progress fetch here
            return;
        }

        self.fetch_ws_receiver = Some((
            crossbeam_channel::after(Duration::from_millis(100)),
            FetchWorkspaceRequest { path: Some(path), force_crate_graph_reload },
        ));
    }

    pub(crate) fn debounce_workspace_fetch(&mut self) {
        if let Some((fetch_receiver, _)) = &mut self.fetch_ws_receiver {
            *fetch_receiver = crossbeam_channel::after(Duration::from_millis(100));
        }
    }
}

impl Drop for GlobalState {
    fn drop(&mut self) {
        self.analysis_host.request_cancellation();
    }
}

impl GlobalStateSnapshot {
    fn vfs_read(&self) -> MappedRwLockReadGuard<'_, vfs::Vfs> {
        RwLockReadGuard::map(self.vfs.read(), |(it, _)| it)
    }

    /// Returns `None` if the file was excluded.
    pub(crate) fn url_to_file_id(&self, url: &Url) -> anyhow::Result<Option<FileId>> {
        url_to_file_id(&self.vfs_read(), url)
    }

    pub(crate) fn file_id_to_url(&self, id: FileId) -> Url {
        file_id_to_url(&self.vfs_read(), id)
    }

    /// Returns `None` if the file was excluded.
    pub(crate) fn vfs_path_to_file_id(&self, vfs_path: &VfsPath) -> anyhow::Result<Option<FileId>> {
        vfs_path_to_file_id(&self.vfs_read(), vfs_path)
    }

    pub(crate) fn file_line_index(&self, file_id: FileId) -> Cancellable<LineIndex> {
        let endings = self.vfs.read().1[&file_id];
        let index = self.analysis.file_line_index(file_id)?;
        let res = LineIndex { index, endings, encoding: self.config.caps().negotiated_encoding() };
        Ok(res)
    }

    pub(crate) fn file_version(&self, file_id: FileId) -> Option<i32> {
        Some(self.mem_docs.get(self.vfs_read().file_path(file_id))?.version)
    }

    pub(crate) fn url_file_version(&self, url: &Url) -> Option<i32> {
        let path = from_proto::vfs_path(url).ok()?;
        Some(self.mem_docs.get(&path)?.version)
    }

    pub(crate) fn anchored_path(&self, path: &AnchoredPathBuf) -> Url {
        let mut base = self.vfs_read().file_path(path.anchor).clone();
        base.pop();
        let path = base.join(&path.path).unwrap();
        let path = path.as_path().unwrap();
        url_from_abs_path(path)
    }

    pub(crate) fn file_id_to_file_path(&self, file_id: FileId) -> vfs::VfsPath {
        self.vfs_read().file_path(file_id).clone()
    }

    pub(crate) fn target_spec_for_crate(&self, crate_id: Crate) -> Option<TargetSpec> {
        let file_id = self.analysis.crate_root(crate_id).ok()?;
        let path = self.vfs_read().file_path(file_id).clone();
        let path = path.as_path()?;

        for workspace in self.workspaces.iter() {
            match &workspace.kind {
                ProjectWorkspaceKind::Cargo { cargo, .. }
                | ProjectWorkspaceKind::DetachedFile { cargo: Some((cargo, _, _)), .. } => {
                    let Some(target_idx) = cargo.target_by_root(path) else {
                        continue;
                    };

                    let target_data = &cargo[target_idx];
                    let package_data = &cargo[target_data.package];

                    return Some(TargetSpec::Cargo(CargoTargetSpec {
                        workspace_root: cargo.workspace_root().to_path_buf(),
                        cargo_toml: package_data.manifest.clone(),
                        crate_id,
                        package: cargo.package_flag(package_data),
                        target: target_data.name.clone(),
                        target_kind: target_data.kind,
                        required_features: target_data.required_features.clone(),
                        features: package_data.features.keys().cloned().collect(),
                        sysroot_root: workspace.sysroot.root().map(ToOwned::to_owned),
                    }));
                }
                ProjectWorkspaceKind::Json(project) => {
                    let Some(krate) = project.crate_by_root(path) else {
                        continue;
                    };
                    let Some(build) = krate.build else {
                        continue;
                    };

                    return Some(TargetSpec::ProjectJson(ProjectJsonTargetSpec {
                        label: build.label,
                        target_kind: build.target_kind,
                        shell_runnables: project.runnables().to_owned(),
                    }));
                }
                ProjectWorkspaceKind::DetachedFile { .. } => {}
            };
        }

        None
    }

    pub(crate) fn file_exists(&self, file_id: FileId) -> bool {
        self.vfs.read().0.exists(file_id)
    }
}

pub(crate) fn file_id_to_url(vfs: &vfs::Vfs, id: FileId) -> Url {
    let path = vfs.file_path(id);
    let path = path.as_path().unwrap();
    url_from_abs_path(path)
}

/// Returns `None` if the file was excluded.
pub(crate) fn url_to_file_id(vfs: &vfs::Vfs, url: &Url) -> anyhow::Result<Option<FileId>> {
    let path = from_proto::vfs_path(url)?;
    vfs_path_to_file_id(vfs, &path)
}

/// Returns `None` if the file was excluded.
pub(crate) fn vfs_path_to_file_id(
    vfs: &vfs::Vfs,
    vfs_path: &VfsPath,
) -> anyhow::Result<Option<FileId>> {
    let (file_id, excluded) =
        vfs.file_id(vfs_path).ok_or_else(|| anyhow::format_err!("file not found: {vfs_path}"))?;
    match excluded {
        vfs::FileExcluded::Yes => Ok(None),
        vfs::FileExcluded::No => Ok(Some(file_id)),
    }
}
