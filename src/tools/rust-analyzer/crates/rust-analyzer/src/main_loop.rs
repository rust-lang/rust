//! The main loop of `rust-analyzer` responsible for dispatching LSP
//! requests/replies and notifications back to the client.

use std::{
    fmt,
    ops::Div as _,
    panic::AssertUnwindSafe,
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, never, select};
use ide_db::base_db::{SourceDatabase, VfsPath, salsa::Database as _};
use lsp_server::{Connection, Notification, Request};
use lsp_types::{TextDocumentIdentifier, notification::Notification as _};
use stdx::thread::ThreadIntent;
use tracing::{Level, error, span};
use vfs::{AbsPathBuf, FileId, loader::LoadingProgress};

use crate::{
    config::Config,
    diagnostics::{DiagnosticsGeneration, NativeDiagnosticsFetchKind, fetch_native_diagnostics},
    discover::{DiscoverArgument, DiscoverCommand, DiscoverProjectMessage},
    flycheck::{self, FlycheckMessage},
    global_state::{
        FetchBuildDataResponse, FetchWorkspaceRequest, FetchWorkspaceResponse, GlobalState,
        file_id_to_url, url_to_file_id,
    },
    handlers::{
        dispatch::{NotificationDispatcher, RequestDispatcher},
        request::empty_diagnostic_report,
    },
    lsp::{
        from_proto, to_proto,
        utils::{Progress, notification_is},
    },
    lsp_ext,
    reload::{BuildDataProgress, ProcMacroProgress, ProjectWorkspaceProgress},
    test_runner::{CargoTestMessage, CargoTestOutput, TestState},
};

pub fn main_loop(config: Config, connection: Connection) -> anyhow::Result<()> {
    tracing::info!("initial config: {:#?}", config);

    // Windows scheduler implements priority boosts: if thread waits for an
    // event (like a condvar), and event fires, priority of the thread is
    // temporary bumped. This optimization backfires in our case: each time the
    // `main_loop` schedules a task to run on a threadpool, the worker threads
    // gets a higher priority, and (on a machine with fewer cores) displaces the
    // main loop! We work around this by marking the main loop as a
    // higher-priority thread.
    //
    // https://docs.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
    // https://docs.microsoft.com/en-us/windows/win32/procthread/priority-boosts
    // https://github.com/rust-lang/rust-analyzer/issues/2835
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::*;
        let thread = GetCurrentThread();
        let thread_priority_above_normal = 1;
        SetThreadPriority(thread, thread_priority_above_normal);
    }

    GlobalState::new(connection.sender, config).run(connection.receiver)
}

enum Event {
    Lsp(lsp_server::Message),
    Task(Task),
    QueuedTask(QueuedTask),
    Vfs(vfs::loader::Message),
    Flycheck(FlycheckMessage),
    TestResult(CargoTestMessage),
    DiscoverProject(DiscoverProjectMessage),
    FetchWorkspaces(FetchWorkspaceRequest),
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Event::Lsp(_) => write!(f, "Event::Lsp"),
            Event::Task(_) => write!(f, "Event::Task"),
            Event::Vfs(_) => write!(f, "Event::Vfs"),
            Event::Flycheck(_) => write!(f, "Event::Flycheck"),
            Event::QueuedTask(_) => write!(f, "Event::QueuedTask"),
            Event::TestResult(_) => write!(f, "Event::TestResult"),
            Event::DiscoverProject(_) => write!(f, "Event::DiscoverProject"),
            Event::FetchWorkspaces(_) => write!(f, "Event::SwitchWorkspaces"),
        }
    }
}

#[derive(Debug)]
pub(crate) enum QueuedTask {
    CheckIfIndexed(lsp_types::Url),
    CheckProcMacroSources(Vec<FileId>),
}

#[derive(Debug)]
pub(crate) enum DiagnosticsTaskKind {
    Syntax(DiagnosticsGeneration, Vec<(FileId, Vec<lsp_types::Diagnostic>)>),
    Semantic(DiagnosticsGeneration, Vec<(FileId, Vec<lsp_types::Diagnostic>)>),
}

#[derive(Debug)]
pub(crate) enum Task {
    Response(lsp_server::Response),
    DiscoverLinkedProjects(DiscoverProjectParam),
    Retry(lsp_server::Request),
    Diagnostics(DiagnosticsTaskKind),
    DiscoverTest(lsp_ext::DiscoverTestResults),
    PrimeCaches(PrimeCachesProgress),
    FetchWorkspace(ProjectWorkspaceProgress),
    FetchBuildData(BuildDataProgress),
    LoadProcMacros(ProcMacroProgress),
    // FIXME: Remove this in favor of a more general QueuedTask, see `handle_did_save_text_document`
    BuildDepsHaveChanged,
}

#[derive(Debug)]
pub(crate) enum DiscoverProjectParam {
    Buildfile(AbsPathBuf),
    Path(AbsPathBuf),
}

#[derive(Debug)]
pub(crate) enum PrimeCachesProgress {
    Begin,
    Report(ide::ParallelPrimeCachesProgress),
    End { cancelled: bool },
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let debug_non_verbose = |not: &Notification, f: &mut fmt::Formatter<'_>| {
            f.debug_struct("Notification").field("method", &not.method).finish()
        };

        match self {
            Event::Lsp(lsp_server::Message::Notification(not)) => {
                if notification_is::<lsp_types::notification::DidOpenTextDocument>(not)
                    || notification_is::<lsp_types::notification::DidChangeTextDocument>(not)
                {
                    return debug_non_verbose(not, f);
                }
            }
            Event::Task(Task::Response(resp)) => {
                return f
                    .debug_struct("Response")
                    .field("id", &resp.id)
                    .field("error", &resp.error)
                    .finish();
            }
            _ => (),
        }

        match self {
            Event::Lsp(it) => fmt::Debug::fmt(it, f),
            Event::Task(it) => fmt::Debug::fmt(it, f),
            Event::QueuedTask(it) => fmt::Debug::fmt(it, f),
            Event::Vfs(it) => fmt::Debug::fmt(it, f),
            Event::Flycheck(it) => fmt::Debug::fmt(it, f),
            Event::TestResult(it) => fmt::Debug::fmt(it, f),
            Event::DiscoverProject(it) => fmt::Debug::fmt(it, f),
            Event::FetchWorkspaces(it) => fmt::Debug::fmt(it, f),
        }
    }
}

impl GlobalState {
    fn run(mut self, inbox: Receiver<lsp_server::Message>) -> anyhow::Result<()> {
        self.update_status_or_notify();

        if self.config.did_save_text_document_dynamic_registration() {
            let additional_patterns = self
                .config
                .discover_workspace_config()
                .map(|cfg| cfg.files_to_watch.clone().into_iter())
                .into_iter()
                .flatten()
                .map(|f| format!("**/{f}"));
            self.register_did_save_capability(additional_patterns);
        }

        if self.config.discover_workspace_config().is_none() {
            self.fetch_workspaces_queue.request_op(
                "startup".to_owned(),
                FetchWorkspaceRequest { path: None, force_crate_graph_reload: false },
            );
            if let Some((cause, FetchWorkspaceRequest { path, force_crate_graph_reload })) =
                self.fetch_workspaces_queue.should_start_op()
            {
                self.fetch_workspaces(cause, path, force_crate_graph_reload);
            }
        }

        while let Ok(event) = self.next_event(&inbox) {
            let Some(event) = event else {
                anyhow::bail!("client exited without proper shutdown sequence");
            };
            if matches!(
                &event,
                Event::Lsp(lsp_server::Message::Notification(Notification { method, .. }))
                if method == lsp_types::notification::Exit::METHOD
            ) {
                return Ok(());
            }
            self.handle_event(event);
        }

        Err(anyhow::anyhow!("A receiver has been dropped, something panicked!"))
    }

    fn register_did_save_capability(&mut self, additional_patterns: impl Iterator<Item = String>) {
        let additional_filters = additional_patterns.map(|pattern| lsp_types::DocumentFilter {
            language: None,
            scheme: None,
            pattern: (Some(pattern)),
        });

        let mut selectors = vec![
            lsp_types::DocumentFilter {
                language: None,
                scheme: None,
                pattern: Some("**/*.rs".into()),
            },
            lsp_types::DocumentFilter {
                language: None,
                scheme: None,
                pattern: Some("**/Cargo.toml".into()),
            },
            lsp_types::DocumentFilter {
                language: None,
                scheme: None,
                pattern: Some("**/Cargo.lock".into()),
            },
        ];
        selectors.extend(additional_filters);

        let save_registration_options = lsp_types::TextDocumentSaveRegistrationOptions {
            include_text: Some(false),
            text_document_registration_options: lsp_types::TextDocumentRegistrationOptions {
                document_selector: Some(selectors),
            },
        };

        let registration = lsp_types::Registration {
            id: "textDocument/didSave".to_owned(),
            method: "textDocument/didSave".to_owned(),
            register_options: Some(serde_json::to_value(save_registration_options).unwrap()),
        };
        self.send_request::<lsp_types::request::RegisterCapability>(
            lsp_types::RegistrationParams { registrations: vec![registration] },
            |_, _| (),
        );
    }

    fn next_event(
        &mut self,
        inbox: &Receiver<lsp_server::Message>,
    ) -> Result<Option<Event>, crossbeam_channel::RecvError> {
        // Make sure we reply to formatting requests ASAP so the editor doesn't block
        if let Ok(task) = self.fmt_pool.receiver.try_recv() {
            return Ok(Some(Event::Task(task)));
        }

        select! {
            recv(inbox) -> msg =>
                return Ok(msg.ok().map(Event::Lsp)),

            recv(self.task_pool.receiver) -> task =>
                task.map(Event::Task),

            recv(self.deferred_task_queue.receiver) -> task =>
                task.map(Event::QueuedTask),

            recv(self.fmt_pool.receiver) -> task =>
                task.map(Event::Task),

            recv(self.loader.receiver) -> task =>
                task.map(Event::Vfs),

            recv(self.flycheck_receiver) -> task =>
                task.map(Event::Flycheck),

            recv(self.test_run_receiver) -> task =>
                task.map(Event::TestResult),

            recv(self.discover_receiver) -> task =>
                task.map(Event::DiscoverProject),

            recv(self.fetch_ws_receiver.as_ref().map_or(&never(), |(chan, _)| chan)) -> _instant => {
                Ok(Event::FetchWorkspaces(self.fetch_ws_receiver.take().unwrap().1))
            },
        }
        .map(Some)
    }

    fn handle_event(&mut self, event: Event) {
        let loop_start = Instant::now();
        let _p = tracing::info_span!("GlobalState::handle_event", event = %event).entered();

        let event_dbg_msg = format!("{event:?}");
        tracing::debug!(?loop_start, ?event, "handle_event");
        if tracing::enabled!(tracing::Level::INFO) {
            let task_queue_len = self.task_pool.handle.len();
            if task_queue_len > 0 {
                tracing::info!("task queue len: {}", task_queue_len);
            }
        }

        let was_quiescent = self.is_quiescent();
        match event {
            Event::Lsp(msg) => match msg {
                lsp_server::Message::Request(req) => self.on_new_request(loop_start, req),
                lsp_server::Message::Notification(not) => self.on_notification(not),
                lsp_server::Message::Response(resp) => self.complete_request(resp),
            },
            Event::QueuedTask(task) => {
                let _p = tracing::info_span!("GlobalState::handle_event/queued_task").entered();
                self.handle_queued_task(task);
                // Coalesce multiple task events into one loop turn
                while let Ok(task) = self.deferred_task_queue.receiver.try_recv() {
                    self.handle_queued_task(task);
                }
            }
            Event::Task(task) => {
                let _p = tracing::info_span!("GlobalState::handle_event/task").entered();
                let mut prime_caches_progress = Vec::new();

                self.handle_task(&mut prime_caches_progress, task);
                // Coalesce multiple task events into one loop turn
                while let Ok(task) = self.task_pool.receiver.try_recv() {
                    self.handle_task(&mut prime_caches_progress, task);
                }

                for progress in prime_caches_progress {
                    let (state, message, fraction, title);
                    match progress {
                        PrimeCachesProgress::Begin => {
                            state = Progress::Begin;
                            message = None;
                            fraction = 0.0;
                            title = "Indexing";
                        }
                        PrimeCachesProgress::Report(report) => {
                            state = Progress::Report;
                            title = report.work_type;

                            message = match &*report.crates_currently_indexing {
                                [crate_name] => Some(format!(
                                    "{}/{} ({})",
                                    report.crates_done,
                                    report.crates_total,
                                    crate_name.as_str(),
                                )),
                                [crate_name, rest @ ..] => Some(format!(
                                    "{}/{} ({} + {} more)",
                                    report.crates_done,
                                    report.crates_total,
                                    crate_name.as_str(),
                                    rest.len()
                                )),
                                _ => None,
                            };

                            fraction = Progress::fraction(report.crates_done, report.crates_total);
                        }
                        PrimeCachesProgress::End { cancelled } => {
                            state = Progress::End;
                            message = None;
                            fraction = 1.0;
                            title = "Indexing";

                            self.analysis_host.raw_database_mut().trigger_lru_eviction();
                            self.prime_caches_queue.op_completed(());
                            if cancelled {
                                self.prime_caches_queue
                                    .request_op("restart after cancellation".to_owned(), ());
                            }
                        }
                    };

                    self.report_progress(
                        title,
                        state,
                        message,
                        Some(fraction),
                        Some("rustAnalyzer/cachePriming".to_owned()),
                    );
                }
            }
            Event::Vfs(message) => {
                let _p = tracing::info_span!("GlobalState::handle_event/vfs").entered();
                self.handle_vfs_msg(message);
                // Coalesce many VFS event into a single loop turn
                while let Ok(message) = self.loader.receiver.try_recv() {
                    self.handle_vfs_msg(message);
                }
            }
            Event::Flycheck(message) => {
                let _p = tracing::info_span!("GlobalState::handle_event/flycheck").entered();
                self.handle_flycheck_msg(message);
                // Coalesce many flycheck updates into a single loop turn
                while let Ok(message) = self.flycheck_receiver.try_recv() {
                    self.handle_flycheck_msg(message);
                }
            }
            Event::TestResult(message) => {
                let _p = tracing::info_span!("GlobalState::handle_event/test_result").entered();
                self.handle_cargo_test_msg(message);
                // Coalesce many test result event into a single loop turn
                while let Ok(message) = self.test_run_receiver.try_recv() {
                    self.handle_cargo_test_msg(message);
                }
            }
            Event::DiscoverProject(message) => {
                self.handle_discover_msg(message);
                // Coalesce many project discovery events into a single loop turn.
                while let Ok(message) = self.discover_receiver.try_recv() {
                    self.handle_discover_msg(message);
                }
            }
            Event::FetchWorkspaces(req) => {
                self.fetch_workspaces_queue.request_op("project structure change".to_owned(), req)
            }
        }
        let event_handling_duration = loop_start.elapsed();
        let (state_changed, memdocs_added_or_removed) = if self.vfs_done {
            if let Some(cause) = self.wants_to_switch.take() {
                self.switch_workspaces(cause);
            }
            (self.process_changes(), self.mem_docs.take_changes())
        } else {
            (false, false)
        };

        if self.is_quiescent() {
            let became_quiescent = !was_quiescent;
            if became_quiescent {
                if self.config.check_on_save(None)
                    && self.config.flycheck_workspace(None)
                    && !self.fetch_build_data_queue.op_requested()
                {
                    // Project has loaded properly, kick off initial flycheck
                    self.flycheck.iter().for_each(|flycheck| flycheck.restart_workspace(None));
                }
                if self.config.prefill_caches() {
                    self.prime_caches_queue.request_op("became quiescent".to_owned(), ());
                }
            }

            let client_refresh = became_quiescent || state_changed;
            if client_refresh {
                // Refresh semantic tokens if the client supports it.
                if self.config.semantic_tokens_refresh() {
                    self.semantic_tokens_cache.lock().clear();
                    self.send_request::<lsp_types::request::SemanticTokensRefresh>((), |_, _| ());
                }

                // Refresh code lens if the client supports it.
                if self.config.code_lens_refresh() {
                    self.send_request::<lsp_types::request::CodeLensRefresh>((), |_, _| ());
                }

                // Refresh inlay hints if the client supports it.
                if self.config.inlay_hints_refresh() {
                    self.send_request::<lsp_types::request::InlayHintRefreshRequest>((), |_, _| ());
                }

                if self.config.diagnostics_refresh() {
                    self.send_request::<lsp_types::request::WorkspaceDiagnosticRefresh>(
                        (),
                        |_, _| (),
                    );
                }
            }

            let project_or_mem_docs_changed =
                became_quiescent || state_changed || memdocs_added_or_removed;
            if project_or_mem_docs_changed
                && !self.config.text_document_diagnostic()
                && self.config.publish_diagnostics(None)
            {
                self.update_diagnostics();
            }
            if project_or_mem_docs_changed && self.config.test_explorer() {
                self.update_tests();
            }
        }

        if let Some(diagnostic_changes) = self.diagnostics.take_changes() {
            for file_id in diagnostic_changes {
                let uri = file_id_to_url(&self.vfs.read().0, file_id);
                let version = from_proto::vfs_path(&uri)
                    .ok()
                    .and_then(|path| self.mem_docs.get(&path).map(|it| it.version));

                let diagnostics =
                    self.diagnostics.diagnostics_for(file_id).cloned().collect::<Vec<_>>();
                self.publish_diagnostics(uri, version, diagnostics);
            }
        }

        if self.config.cargo_autoreload_config(None)
            || self.config.discover_workspace_config().is_some()
        {
            if let Some((cause, FetchWorkspaceRequest { path, force_crate_graph_reload })) =
                self.fetch_workspaces_queue.should_start_op()
            {
                self.fetch_workspaces(cause, path, force_crate_graph_reload);
            }
        }

        if !self.fetch_workspaces_queue.op_in_progress() {
            if let Some((cause, ())) = self.fetch_build_data_queue.should_start_op() {
                self.fetch_build_data(cause);
            } else if let Some((cause, (change, paths))) =
                self.fetch_proc_macros_queue.should_start_op()
            {
                self.fetch_proc_macros(cause, change, paths);
            }
        }

        if let Some((cause, ())) = self.prime_caches_queue.should_start_op() {
            self.prime_caches(cause);
        }

        self.update_status_or_notify();

        let loop_duration = loop_start.elapsed();
        if loop_duration > Duration::from_millis(100) && was_quiescent {
            tracing::warn!(
                "overly long loop turn took {loop_duration:?} (event handling took {event_handling_duration:?}): {event_dbg_msg}"
            );
            self.poke_rust_analyzer_developer(format!(
                "overly long loop turn took {loop_duration:?} (event handling took {event_handling_duration:?}): {event_dbg_msg}"
            ));
        }
    }

    fn prime_caches(&mut self, cause: String) {
        tracing::debug!(%cause, "will prime caches");
        let num_worker_threads = self.config.prime_caches_num_threads();

        self.task_pool.handle.spawn_with_sender(ThreadIntent::Worker, {
            let analysis = AssertUnwindSafe(self.snapshot().analysis);
            move |sender| {
                sender.send(Task::PrimeCaches(PrimeCachesProgress::Begin)).unwrap();
                let res = analysis.parallel_prime_caches(num_worker_threads, |progress| {
                    let report = PrimeCachesProgress::Report(progress);
                    sender.send(Task::PrimeCaches(report)).unwrap();
                });
                sender
                    .send(Task::PrimeCaches(PrimeCachesProgress::End { cancelled: res.is_err() }))
                    .unwrap();
            }
        });
    }

    fn update_diagnostics(&mut self) {
        let db = self.analysis_host.raw_database();
        let generation = self.diagnostics.next_generation();
        let subscriptions = {
            let vfs = &self.vfs.read().0;
            self.mem_docs
                .iter()
                .map(|path| vfs.file_id(path).unwrap())
                .filter_map(|(file_id, excluded)| {
                    (excluded == vfs::FileExcluded::No).then_some(file_id)
                })
                .filter(|&file_id| {
                    let source_root_id = db.file_source_root(file_id).source_root_id(db);
                    let source_root = db.source_root(source_root_id).source_root(db);
                    // Only publish diagnostics for files in the workspace, not from crates.io deps
                    // or the sysroot.
                    // While theoretically these should never have errors, we have quite a few false
                    // positives particularly in the stdlib, and those diagnostics would stay around
                    // forever if we emitted them here.
                    !source_root.is_library
                })
                .collect::<std::sync::Arc<_>>()
        };
        tracing::trace!("updating notifications for {:?}", subscriptions);
        // Split up the work on multiple threads, but we don't wanna fill the entire task pool with
        // diagnostic tasks, so we limit the number of tasks to a quarter of the total thread pool.
        let max_tasks = self.config.main_loop_num_threads().div(4).max(1);
        let chunk_length = subscriptions.len() / max_tasks;
        let remainder = subscriptions.len() % max_tasks;

        let mut start = 0;
        for task_idx in 0..max_tasks {
            let extra = if task_idx < remainder { 1 } else { 0 };
            let end = start + chunk_length + extra;
            let slice = start..end;
            if slice.is_empty() {
                break;
            }
            // Diagnostics are triggered by the user typing
            // so we run them on a latency sensitive thread.
            let snapshot = self.snapshot();
            self.task_pool.handle.spawn_with_sender(ThreadIntent::LatencySensitive, {
                let subscriptions = subscriptions.clone();
                // Do not fetch semantic diagnostics (and populate query results) if we haven't even
                // loaded the initial workspace yet.
                let fetch_semantic =
                    self.vfs_done && self.fetch_workspaces_queue.last_op_result().is_some();
                move |sender| {
                    // We aren't observing the semantics token cache here
                    let snapshot = AssertUnwindSafe(&snapshot);
                    let Ok(diags) = std::panic::catch_unwind(|| {
                        fetch_native_diagnostics(
                            &snapshot,
                            subscriptions.clone(),
                            slice.clone(),
                            NativeDiagnosticsFetchKind::Syntax,
                        )
                    }) else {
                        return;
                    };
                    sender
                        .send(Task::Diagnostics(DiagnosticsTaskKind::Syntax(generation, diags)))
                        .unwrap();

                    if fetch_semantic {
                        let Ok(diags) = std::panic::catch_unwind(|| {
                            fetch_native_diagnostics(
                                &snapshot,
                                subscriptions.clone(),
                                slice.clone(),
                                NativeDiagnosticsFetchKind::Semantic,
                            )
                        }) else {
                            return;
                        };
                        sender
                            .send(Task::Diagnostics(DiagnosticsTaskKind::Semantic(
                                generation, diags,
                            )))
                            .unwrap();
                    }
                }
            });
            start = end;
        }
    }

    fn update_tests(&mut self) {
        if !self.vfs_done {
            return;
        }
        let db = self.analysis_host.raw_database();
        let subscriptions = self
            .mem_docs
            .iter()
            .map(|path| self.vfs.read().0.file_id(path).unwrap())
            .filter_map(|(file_id, excluded)| {
                (excluded == vfs::FileExcluded::No).then_some(file_id)
            })
            .filter(|&file_id| {
                let source_root_id = db.file_source_root(file_id).source_root_id(db);
                let source_root = db.source_root(source_root_id).source_root(db);
                !source_root.is_library
            })
            .collect::<Vec<_>>();
        tracing::trace!("updating tests for {:?}", subscriptions);

        // Updating tests are triggered by the user typing
        // so we run them on a latency sensitive thread.
        self.task_pool.handle.spawn(ThreadIntent::LatencySensitive, {
            let snapshot = self.snapshot();
            move || {
                let tests = subscriptions
                    .iter()
                    .copied()
                    .filter_map(|f| snapshot.analysis.discover_tests_in_file(f).ok())
                    .flatten()
                    .collect::<Vec<_>>();

                Task::DiscoverTest(lsp_ext::DiscoverTestResults {
                    tests: tests
                        .into_iter()
                        .filter_map(|t| {
                            let line_index = t.file.and_then(|f| snapshot.file_line_index(f).ok());
                            to_proto::test_item(&snapshot, t, line_index.as_ref())
                        })
                        .collect(),
                    scope: None,
                    scope_file: Some(
                        subscriptions
                            .into_iter()
                            .map(|f| TextDocumentIdentifier { uri: to_proto::url(&snapshot, f) })
                            .collect(),
                    ),
                })
            }
        });
    }

    fn update_status_or_notify(&mut self) {
        let status = self.current_status();
        if self.last_reported_status != status {
            self.last_reported_status = status.clone();

            if self.config.server_status_notification() {
                self.send_notification::<lsp_ext::ServerStatusNotification>(status);
            } else if let (
                health @ (lsp_ext::Health::Warning | lsp_ext::Health::Error),
                Some(message),
            ) = (status.health, &status.message)
            {
                let open_log_button = tracing::enabled!(tracing::Level::ERROR)
                    && (self.fetch_build_data_error().is_err()
                        || self.fetch_workspace_error().is_err());
                self.show_message(
                    match health {
                        lsp_ext::Health::Ok => lsp_types::MessageType::INFO,
                        lsp_ext::Health::Warning => lsp_types::MessageType::WARNING,
                        lsp_ext::Health::Error => lsp_types::MessageType::ERROR,
                    },
                    message.clone(),
                    open_log_button,
                );
            }
        }
    }

    fn handle_task(&mut self, prime_caches_progress: &mut Vec<PrimeCachesProgress>, task: Task) {
        match task {
            Task::Response(response) => self.respond(response),
            // Only retry requests that haven't been cancelled. Otherwise we do unnecessary work.
            Task::Retry(req) if !self.is_completed(&req) => self.on_request(req),
            Task::Retry(_) => (),
            Task::Diagnostics(kind) => {
                self.diagnostics.set_native_diagnostics(kind);
            }
            Task::PrimeCaches(progress) => match progress {
                PrimeCachesProgress::Begin => prime_caches_progress.push(progress),
                PrimeCachesProgress::Report(_) => {
                    match prime_caches_progress.last_mut() {
                        Some(last @ PrimeCachesProgress::Report(_)) => {
                            // Coalesce subsequent update events.
                            *last = progress;
                        }
                        _ => prime_caches_progress.push(progress),
                    }
                }
                PrimeCachesProgress::End { .. } => prime_caches_progress.push(progress),
            },
            Task::FetchWorkspace(progress) => {
                let (state, msg) = match progress {
                    ProjectWorkspaceProgress::Begin => (Progress::Begin, None),
                    ProjectWorkspaceProgress::Report(msg) => (Progress::Report, Some(msg)),
                    ProjectWorkspaceProgress::End(workspaces, force_crate_graph_reload) => {
                        let resp = FetchWorkspaceResponse { workspaces, force_crate_graph_reload };
                        self.fetch_workspaces_queue.op_completed(resp);
                        if let Err(e) = self.fetch_workspace_error() {
                            error!("FetchWorkspaceError: {e}");
                        }
                        self.wants_to_switch = Some("fetched workspace".to_owned());
                        self.diagnostics.clear_check_all();
                        (Progress::End, None)
                    }
                };

                self.report_progress("Fetching", state, msg, None, None);
            }
            Task::DiscoverLinkedProjects(arg) => {
                if let Some(cfg) = self.config.discover_workspace_config() {
                    if !self.discover_workspace_queue.op_in_progress() {
                        // the clone is unfortunately necessary to avoid a borrowck error when
                        // `self.report_progress` is called later
                        let title = &cfg.progress_label.clone();
                        let command = cfg.command.clone();
                        let discover = DiscoverCommand::new(self.discover_sender.clone(), command);

                        self.report_progress(title, Progress::Begin, None, None, None);
                        self.discover_workspace_queue
                            .request_op("Discovering workspace".to_owned(), ());
                        let _ = self.discover_workspace_queue.should_start_op();

                        let arg = match arg {
                            DiscoverProjectParam::Buildfile(it) => DiscoverArgument::Buildfile(it),
                            DiscoverProjectParam::Path(it) => DiscoverArgument::Path(it),
                        };

                        let handle =
                            discover.spawn(arg, &std::env::current_dir().unwrap()).unwrap();
                        self.discover_handle = Some(handle);
                    }
                }
            }
            Task::FetchBuildData(progress) => {
                let (state, msg) = match progress {
                    BuildDataProgress::Begin => (Some(Progress::Begin), None),
                    BuildDataProgress::Report(msg) => (Some(Progress::Report), Some(msg)),
                    BuildDataProgress::End((workspaces, build_scripts)) => {
                        let resp = FetchBuildDataResponse { workspaces, build_scripts };
                        self.fetch_build_data_queue.op_completed(resp);

                        if let Err(e) = self.fetch_build_data_error() {
                            error!("FetchBuildDataError: {e}");
                        }

                        if self.wants_to_switch.is_none() {
                            self.wants_to_switch = Some("fetched build data".to_owned());
                        }
                        (Some(Progress::End), None)
                    }
                };

                if let Some(state) = state {
                    self.report_progress("Building build-artifacts", state, msg, None, None);
                }
            }
            Task::LoadProcMacros(progress) => {
                let (state, msg) = match progress {
                    ProcMacroProgress::Begin => (Some(Progress::Begin), None),
                    ProcMacroProgress::Report(msg) => (Some(Progress::Report), Some(msg)),
                    ProcMacroProgress::End(change) => {
                        self.fetch_proc_macros_queue.op_completed(true);
                        self.analysis_host.apply_change(change);
                        self.finish_loading_crate_graph();
                        (Some(Progress::End), None)
                    }
                };

                if let Some(state) = state {
                    self.report_progress("Loading proc-macros", state, msg, None, None);
                }
            }
            Task::BuildDepsHaveChanged => self.build_deps_changed = true,
            Task::DiscoverTest(tests) => {
                self.send_notification::<lsp_ext::DiscoveredTests>(tests);
            }
        }
    }

    fn handle_vfs_msg(&mut self, message: vfs::loader::Message) {
        let _p = tracing::info_span!("GlobalState::handle_vfs_msg").entered();
        let is_changed = matches!(message, vfs::loader::Message::Changed { .. });
        match message {
            vfs::loader::Message::Changed { files } | vfs::loader::Message::Loaded { files } => {
                let _p = tracing::info_span!("GlobalState::handle_vfs_msg{changed/load}").entered();
                self.debounce_workspace_fetch();
                let vfs = &mut self.vfs.write().0;
                for (path, contents) in files {
                    let path = VfsPath::from(path);
                    // if the file is in mem docs, it's managed by the client via notifications
                    // so only set it if its not in there
                    if !self.mem_docs.contains(&path)
                        && (is_changed || vfs.file_id(&path).is_none())
                    {
                        vfs.set_file_contents(path, contents);
                    }
                }
            }
            vfs::loader::Message::Progress { n_total, n_done, dir, config_version } => {
                let _p = span!(Level::INFO, "GlobalState::handle_vfs_msg/progress").entered();
                stdx::always!(config_version <= self.vfs_config_version);

                let (n_done, state) = match n_done {
                    LoadingProgress::Started => {
                        self.vfs_span =
                            Some(span!(Level::INFO, "vfs_load", total = n_total).entered());
                        (0, Progress::Begin)
                    }
                    LoadingProgress::Progress(n_done) => (n_done.min(n_total), Progress::Report),
                    LoadingProgress::Finished => {
                        self.vfs_span = None;
                        (n_total, Progress::End)
                    }
                };

                self.vfs_progress_config_version = config_version;
                self.vfs_done = state == Progress::End;

                let mut message = format!("{n_done}/{n_total}");
                if let Some(dir) = dir {
                    message += &format!(
                        ": {}",
                        match dir.strip_prefix(self.config.root_path()) {
                            Some(relative_path) => relative_path.as_utf8_path(),
                            None => dir.as_ref(),
                        }
                    );
                }

                self.report_progress(
                    "Roots Scanned",
                    state,
                    Some(message),
                    Some(Progress::fraction(n_done, n_total)),
                    None,
                );
            }
        }
    }

    fn handle_queued_task(&mut self, task: QueuedTask) {
        match task {
            QueuedTask::CheckIfIndexed(uri) => {
                let snap = self.snapshot();

                self.task_pool.handle.spawn_with_sender(ThreadIntent::Worker, move |sender| {
                    let _p = tracing::info_span!("GlobalState::check_if_indexed").entered();
                    tracing::debug!(?uri, "handling uri");
                    let Some(id) = from_proto::file_id(&snap, &uri).expect("unable to get FileId")
                    else {
                        return;
                    };
                    if let Ok(crates) = &snap.analysis.crates_for(id) {
                        if crates.is_empty() {
                            if snap.config.discover_workspace_config().is_some() {
                                let path =
                                    from_proto::abs_path(&uri).expect("Unable to get AbsPath");
                                let arg = DiscoverProjectParam::Path(path);
                                sender.send(Task::DiscoverLinkedProjects(arg)).unwrap();
                            }
                        } else {
                            tracing::debug!(?uri, "is indexed");
                        }
                    }
                });
            }
            QueuedTask::CheckProcMacroSources(modified_rust_files) => {
                let analysis = AssertUnwindSafe(self.snapshot().analysis);
                self.task_pool.handle.spawn_with_sender(stdx::thread::ThreadIntent::Worker, {
                    move |sender| {
                        if modified_rust_files.into_iter().any(|file_id| {
                            // FIXME: Check whether these files could be build script related
                            match analysis.crates_for(file_id) {
                                Ok(crates) => crates.iter().any(|&krate| {
                                    analysis.is_proc_macro_crate(krate).is_ok_and(|it| it)
                                }),
                                _ => false,
                            }
                        }) {
                            sender.send(Task::BuildDepsHaveChanged).unwrap();
                        }
                    }
                });
            }
        }
    }

    fn handle_discover_msg(&mut self, message: DiscoverProjectMessage) {
        let title = self
            .config
            .discover_workspace_config()
            .map(|cfg| cfg.progress_label.clone())
            .expect("No title could be found; this is a bug");
        match message {
            DiscoverProjectMessage::Finished { project, buildfile } => {
                self.discover_handle = None;
                self.report_progress(&title, Progress::End, None, None, None);
                self.discover_workspace_queue.op_completed(());

                let mut config = Config::clone(&*self.config);
                config.add_discovered_project_from_command(project, buildfile);
                self.update_configuration(config);
            }
            DiscoverProjectMessage::Progress { message } => {
                self.report_progress(&title, Progress::Report, Some(message), None, None)
            }
            DiscoverProjectMessage::Error { error, source } => {
                self.discover_handle = None;
                let message = format!("Project discovery failed: {error}");
                self.discover_workspace_queue.op_completed(());
                self.show_and_log_error(message.clone(), source);
                self.report_progress(&title, Progress::End, Some(message), None, None)
            }
        }
    }

    fn handle_cargo_test_msg(&mut self, message: CargoTestMessage) {
        match message.output {
            CargoTestOutput::Test { name, state } => {
                let state = match state {
                    TestState::Started => lsp_ext::TestState::Started,
                    TestState::Ignored => lsp_ext::TestState::Skipped,
                    TestState::Ok => lsp_ext::TestState::Passed,
                    TestState::Failed { stdout } => lsp_ext::TestState::Failed { message: stdout },
                };

                // The notification requires the namespace form (with underscores) of the target
                let test_id = format!("{}::{name}", message.target.target.replace('-', "_"));

                self.send_notification::<lsp_ext::ChangeTestState>(
                    lsp_ext::ChangeTestStateParams { test_id, state },
                );
            }
            CargoTestOutput::Suite => (),
            CargoTestOutput::Finished => {
                self.test_run_remaining_jobs = self.test_run_remaining_jobs.saturating_sub(1);
                if self.test_run_remaining_jobs == 0 {
                    self.send_notification::<lsp_ext::EndRunTest>(());
                    self.test_run_session = None;
                }
            }
            CargoTestOutput::Custom { text } => {
                self.send_notification::<lsp_ext::AppendOutputToRunTest>(text);
            }
        }
    }

    fn handle_flycheck_msg(&mut self, message: FlycheckMessage) {
        match message {
            FlycheckMessage::AddDiagnostic { id, workspace_root, diagnostic, package_id } => {
                let snap = self.snapshot();
                let diagnostics = crate::diagnostics::to_proto::map_rust_diagnostic_to_lsp(
                    &self.config.diagnostics_map(None),
                    &diagnostic,
                    &workspace_root,
                    &snap,
                );
                for diag in diagnostics {
                    match url_to_file_id(&self.vfs.read().0, &diag.url) {
                        Ok(Some(file_id)) => self.diagnostics.add_check_diagnostic(
                            id,
                            &package_id,
                            file_id,
                            diag.diagnostic,
                            diag.fix,
                        ),
                        Ok(None) => {}
                        Err(err) => {
                            error!(
                                "flycheck {id}: File with cargo diagnostic not found in VFS: {}",
                                err
                            );
                        }
                    };
                }
            }
            FlycheckMessage::ClearDiagnostics { id, package_id: None } => {
                self.diagnostics.clear_check(id)
            }
            FlycheckMessage::ClearDiagnostics { id, package_id: Some(package_id) } => {
                self.diagnostics.clear_check_for_package(id, package_id)
            }
            FlycheckMessage::Progress { id, progress } => {
                let (state, message) = match progress {
                    flycheck::Progress::DidStart => (Progress::Begin, None),
                    flycheck::Progress::DidCheckCrate(target) => (Progress::Report, Some(target)),
                    flycheck::Progress::DidCancel => {
                        self.last_flycheck_error = None;
                        (Progress::End, None)
                    }
                    flycheck::Progress::DidFailToRestart(err) => {
                        self.last_flycheck_error =
                            Some(format!("cargo check failed to start: {err}"));
                        return;
                    }
                    flycheck::Progress::DidFinish(result) => {
                        self.last_flycheck_error =
                            result.err().map(|err| format!("cargo check failed to start: {err}"));
                        (Progress::End, None)
                    }
                };

                // When we're running multiple flychecks, we have to include a disambiguator in
                // the title, or the editor complains. Note that this is a user-facing string.
                let title = if self.flycheck.len() == 1 {
                    format!("{}", self.config.flycheck(None))
                } else {
                    format!("{} (#{})", self.config.flycheck(None), id + 1)
                };
                self.report_progress(
                    &title,
                    state,
                    message,
                    None,
                    Some(format!("rust-analyzer/flycheck/{id}")),
                );
            }
        }
    }

    /// Registers and handles a request. This should only be called once per incoming request.
    fn on_new_request(&mut self, request_received: Instant, req: Request) {
        let _p =
            span!(Level::INFO, "GlobalState::on_new_request", req.method = ?req.method).entered();
        self.register_request(&req, request_received);
        self.on_request(req);
    }

    /// Handles a request.
    fn on_request(&mut self, req: Request) {
        let mut dispatcher = RequestDispatcher { req: Some(req), global_state: self };
        dispatcher.on_sync_mut::<lsp_types::request::Shutdown>(|s, ()| {
            s.shutdown_requested = true;
            Ok(())
        });

        match &mut dispatcher {
            RequestDispatcher { req: Some(req), global_state: this } if this.shutdown_requested => {
                this.respond(lsp_server::Response::new_err(
                    req.id.clone(),
                    lsp_server::ErrorCode::InvalidRequest as i32,
                    "Shutdown already requested.".to_owned(),
                ));
                return;
            }
            _ => (),
        }

        use crate::handlers::request as handlers;
        use lsp_types::request as lsp_request;

        const RETRY: bool = true;
        const NO_RETRY: bool = false;

        #[rustfmt::skip]
        dispatcher
            // Request handlers that must run on the main thread
            // because they mutate GlobalState:
            .on_sync_mut::<lsp_ext::ReloadWorkspace>(handlers::handle_workspace_reload)
            .on_sync_mut::<lsp_ext::RebuildProcMacros>(handlers::handle_proc_macros_rebuild)
            .on_sync_mut::<lsp_ext::MemoryUsage>(handlers::handle_memory_usage)
            .on_sync_mut::<lsp_ext::RunTest>(handlers::handle_run_test)
            // Request handlers which are related to the user typing
            // are run on the main thread to reduce latency:
            .on_sync::<lsp_ext::JoinLines>(handlers::handle_join_lines)
            .on_sync::<lsp_ext::OnEnter>(handlers::handle_on_enter)
            .on_sync::<lsp_request::SelectionRangeRequest>(handlers::handle_selection_range)
            .on_sync::<lsp_ext::MatchingBrace>(handlers::handle_matching_brace)
            .on_sync::<lsp_ext::OnTypeFormatting>(handlers::handle_on_type_formatting)
            // Formatting should be done immediately as the editor might wait on it, but we can't
            // put it on the main thread as we do not want the main thread to block on rustfmt.
            // So we have an extra thread just for formatting requests to make sure it gets handled
            // as fast as possible.
            .on_fmt_thread::<lsp_request::Formatting>(handlers::handle_formatting)
            .on_fmt_thread::<lsp_request::RangeFormatting>(handlers::handle_range_formatting)
            // We can’t run latency-sensitive request handlers which do semantic
            // analysis on the main thread because that would block other
            // requests. Instead, we run these request handlers on higher priority
            // threads in the threadpool.
            // FIXME: Retrying can make the result of this stale?
            .on_latency_sensitive::<RETRY, lsp_request::Completion>(handlers::handle_completion)
            // FIXME: Retrying can make the result of this stale
            .on_latency_sensitive::<RETRY, lsp_request::ResolveCompletionItem>(handlers::handle_completion_resolve)
            .on_latency_sensitive::<RETRY, lsp_request::SemanticTokensFullRequest>(handlers::handle_semantic_tokens_full)
            .on_latency_sensitive::<RETRY, lsp_request::SemanticTokensFullDeltaRequest>(handlers::handle_semantic_tokens_full_delta)
            .on_latency_sensitive::<NO_RETRY, lsp_request::SemanticTokensRangeRequest>(handlers::handle_semantic_tokens_range)
            // FIXME: Some of these NO_RETRY could be retries if the file they are interested didn't change.
            // All other request handlers
            .on_with_vfs_default::<lsp_request::DocumentDiagnosticRequest>(handlers::handle_document_diagnostics, empty_diagnostic_report, || lsp_server::ResponseError {
                code: lsp_server::ErrorCode::ServerCancelled as i32,
                message: "server cancelled the request".to_owned(),
                data: serde_json::to_value(lsp_types::DiagnosticServerCancellationData {
                    retrigger_request: true
                }).ok(),
            })
            .on::<RETRY, lsp_request::DocumentSymbolRequest>(handlers::handle_document_symbol)
            .on::<RETRY, lsp_request::FoldingRangeRequest>(handlers::handle_folding_range)
            .on::<NO_RETRY, lsp_request::SignatureHelpRequest>(handlers::handle_signature_help)
            .on::<RETRY, lsp_request::WillRenameFiles>(handlers::handle_will_rename_files)
            .on::<NO_RETRY, lsp_request::GotoDefinition>(handlers::handle_goto_definition)
            .on::<NO_RETRY, lsp_request::GotoDeclaration>(handlers::handle_goto_declaration)
            .on::<NO_RETRY, lsp_request::GotoImplementation>(handlers::handle_goto_implementation)
            .on::<NO_RETRY, lsp_request::GotoTypeDefinition>(handlers::handle_goto_type_definition)
            .on::<NO_RETRY, lsp_request::InlayHintRequest>(handlers::handle_inlay_hints)
            .on_identity::<NO_RETRY, lsp_request::InlayHintResolveRequest, _>(handlers::handle_inlay_hints_resolve)
            .on::<NO_RETRY, lsp_request::CodeLensRequest>(handlers::handle_code_lens)
            .on_identity::<NO_RETRY, lsp_request::CodeLensResolve, _>(handlers::handle_code_lens_resolve)
            .on::<NO_RETRY, lsp_request::PrepareRenameRequest>(handlers::handle_prepare_rename)
            .on::<NO_RETRY, lsp_request::Rename>(handlers::handle_rename)
            .on::<NO_RETRY, lsp_request::References>(handlers::handle_references)
            .on::<NO_RETRY, lsp_request::DocumentHighlightRequest>(handlers::handle_document_highlight)
            .on::<NO_RETRY, lsp_request::CallHierarchyPrepare>(handlers::handle_call_hierarchy_prepare)
            .on::<NO_RETRY, lsp_request::CallHierarchyIncomingCalls>(handlers::handle_call_hierarchy_incoming)
            .on::<NO_RETRY, lsp_request::CallHierarchyOutgoingCalls>(handlers::handle_call_hierarchy_outgoing)
            // All other request handlers (lsp extension)
            .on::<RETRY, lsp_ext::FetchDependencyList>(handlers::fetch_dependency_list)
            .on::<RETRY, lsp_ext::AnalyzerStatus>(handlers::handle_analyzer_status)
            .on::<RETRY, lsp_ext::ViewFileText>(handlers::handle_view_file_text)
            .on::<RETRY, lsp_ext::ViewCrateGraph>(handlers::handle_view_crate_graph)
            .on::<RETRY, lsp_ext::ViewItemTree>(handlers::handle_view_item_tree)
            .on::<RETRY, lsp_ext::DiscoverTest>(handlers::handle_discover_test)
            .on::<RETRY, lsp_ext::WorkspaceSymbol>(handlers::handle_workspace_symbol)
            .on::<NO_RETRY, lsp_ext::Ssr>(handlers::handle_ssr)
            .on::<NO_RETRY, lsp_ext::ViewRecursiveMemoryLayout>(handlers::handle_view_recursive_memory_layout)
            .on::<NO_RETRY, lsp_ext::ViewSyntaxTree>(handlers::handle_view_syntax_tree)
            .on::<NO_RETRY, lsp_ext::ViewHir>(handlers::handle_view_hir)
            .on::<NO_RETRY, lsp_ext::ViewMir>(handlers::handle_view_mir)
            .on::<NO_RETRY, lsp_ext::InterpretFunction>(handlers::handle_interpret_function)
            .on::<NO_RETRY, lsp_ext::ExpandMacro>(handlers::handle_expand_macro)
            .on::<NO_RETRY, lsp_ext::ParentModule>(handlers::handle_parent_module)
            .on::<NO_RETRY, lsp_ext::ChildModules>(handlers::handle_child_modules)
            .on::<NO_RETRY, lsp_ext::Runnables>(handlers::handle_runnables)
            .on::<NO_RETRY, lsp_ext::RelatedTests>(handlers::handle_related_tests)
            .on::<NO_RETRY, lsp_ext::CodeActionRequest>(handlers::handle_code_action)
            .on_identity::<RETRY, lsp_ext::CodeActionResolveRequest, _>(handlers::handle_code_action_resolve)
            .on::<NO_RETRY, lsp_ext::HoverRequest>(handlers::handle_hover)
            .on::<NO_RETRY, lsp_ext::ExternalDocs>(handlers::handle_open_docs)
            .on::<NO_RETRY, lsp_ext::OpenCargoToml>(handlers::handle_open_cargo_toml)
            .on::<NO_RETRY, lsp_ext::MoveItem>(handlers::handle_move_item)
            //
            .on::<NO_RETRY, lsp_ext::InternalTestingFetchConfig>(handlers::internal_testing_fetch_config)
            .finish();
    }

    /// Handles an incoming notification.
    fn on_notification(&mut self, not: Notification) {
        let _p =
            span!(Level::INFO, "GlobalState::on_notification", not.method = ?not.method).entered();
        use crate::handlers::notification as handlers;
        use lsp_types::notification as notifs;

        NotificationDispatcher { not: Some(not), global_state: self }
            .on_sync_mut::<notifs::Cancel>(handlers::handle_cancel)
            .on_sync_mut::<notifs::WorkDoneProgressCancel>(
                handlers::handle_work_done_progress_cancel,
            )
            .on_sync_mut::<notifs::DidOpenTextDocument>(handlers::handle_did_open_text_document)
            .on_sync_mut::<notifs::DidChangeTextDocument>(handlers::handle_did_change_text_document)
            .on_sync_mut::<notifs::DidCloseTextDocument>(handlers::handle_did_close_text_document)
            .on_sync_mut::<notifs::DidSaveTextDocument>(handlers::handle_did_save_text_document)
            .on_sync_mut::<notifs::DidChangeConfiguration>(
                handlers::handle_did_change_configuration,
            )
            .on_sync_mut::<notifs::DidChangeWorkspaceFolders>(
                handlers::handle_did_change_workspace_folders,
            )
            .on_sync_mut::<notifs::DidChangeWatchedFiles>(handlers::handle_did_change_watched_files)
            .on_sync_mut::<lsp_ext::CancelFlycheck>(handlers::handle_cancel_flycheck)
            .on_sync_mut::<lsp_ext::ClearFlycheck>(handlers::handle_clear_flycheck)
            .on_sync_mut::<lsp_ext::RunFlycheck>(handlers::handle_run_flycheck)
            .on_sync_mut::<lsp_ext::AbortRunTest>(handlers::handle_abort_run_test)
            .finish();
    }
}
