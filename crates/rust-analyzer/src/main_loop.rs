//! The main loop of `rust-analyzer` responsible for dispatching LSP
//! requests/replies and notifications back to the client.
use std::{
    fmt,
    time::{Duration, Instant},
};

use always_assert::always;
use crossbeam_channel::{select, Receiver};
use flycheck::FlycheckHandle;
use ide_db::base_db::{SourceDatabaseExt, VfsPath};
use lsp_server::{Connection, Notification, Request};
use lsp_types::notification::Notification as _;
use triomphe::Arc;
use vfs::FileId;

use crate::{
    config::Config,
    dispatch::{NotificationDispatcher, RequestDispatcher},
    from_proto,
    global_state::{file_id_to_url, url_to_file_id, GlobalState},
    lsp_ext,
    lsp_utils::{notification_is, Progress},
    reload::{BuildDataProgress, ProcMacroProgress, ProjectWorkspaceProgress},
    Result,
};

pub fn main_loop(config: Config, connection: Connection) -> Result<()> {
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
        use winapi::um::processthreadsapi::*;
        let thread = GetCurrentThread();
        let thread_priority_above_normal = 1;
        SetThreadPriority(thread, thread_priority_above_normal);
    }

    GlobalState::new(connection.sender, config).run(connection.receiver)
}

enum Event {
    Lsp(lsp_server::Message),
    Task(Task),
    Vfs(vfs::loader::Message),
    Flycheck(flycheck::Message),
}

#[derive(Debug)]
pub(crate) enum Task {
    Response(lsp_server::Response),
    Retry(lsp_server::Request),
    Diagnostics(Vec<(FileId, Vec<lsp_types::Diagnostic>)>),
    PrimeCaches(PrimeCachesProgress),
    FetchWorkspace(ProjectWorkspaceProgress),
    FetchBuildData(BuildDataProgress),
    LoadProcMacros(ProcMacroProgress),
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
            Event::Vfs(it) => fmt::Debug::fmt(it, f),
            Event::Flycheck(it) => fmt::Debug::fmt(it, f),
        }
    }
}

impl GlobalState {
    fn run(mut self, inbox: Receiver<lsp_server::Message>) -> Result<()> {
        self.update_status_or_notify();

        if self.config.did_save_text_document_dynamic_registration() {
            self.register_did_save_capability();
        }

        self.fetch_workspaces_queue.request_op("startup".to_string(), ());
        if let Some((cause, ())) = self.fetch_workspaces_queue.should_start_op() {
            self.fetch_workspaces(cause);
        }

        while let Some(event) = self.next_event(&inbox) {
            if matches!(
                &event,
                Event::Lsp(lsp_server::Message::Notification(Notification { method, .. }))
                if method == lsp_types::notification::Exit::METHOD
            ) {
                return Ok(());
            }
            self.handle_event(event)?;
        }

        Err("client exited without proper shutdown sequence".into())
    }

    fn register_did_save_capability(&mut self) {
        let save_registration_options = lsp_types::TextDocumentSaveRegistrationOptions {
            include_text: Some(false),
            text_document_registration_options: lsp_types::TextDocumentRegistrationOptions {
                document_selector: Some(vec![
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
                ]),
            },
        };

        let registration = lsp_types::Registration {
            id: "textDocument/didSave".to_string(),
            method: "textDocument/didSave".to_string(),
            register_options: Some(serde_json::to_value(save_registration_options).unwrap()),
        };
        self.send_request::<lsp_types::request::RegisterCapability>(
            lsp_types::RegistrationParams { registrations: vec![registration] },
            |_, _| (),
        );
    }

    fn next_event(&self, inbox: &Receiver<lsp_server::Message>) -> Option<Event> {
        select! {
            recv(inbox) -> msg =>
                msg.ok().map(Event::Lsp),

            recv(self.task_pool.receiver) -> task =>
                Some(Event::Task(task.unwrap())),

            recv(self.loader.receiver) -> task =>
                Some(Event::Vfs(task.unwrap())),

            recv(self.flycheck_receiver) -> task =>
                Some(Event::Flycheck(task.unwrap())),
        }
    }

    fn handle_event(&mut self, event: Event) -> Result<()> {
        let loop_start = Instant::now();
        // NOTE: don't count blocking select! call as a loop-turn time
        let _p = profile::span("GlobalState::handle_event");

        let event_dbg_msg = format!("{event:?}");
        tracing::debug!("{:?} handle_event({})", loop_start, event_dbg_msg);
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
                lsp_server::Message::Notification(not) => self.on_notification(not)?,
                lsp_server::Message::Response(resp) => self.complete_request(resp),
            },
            Event::Task(task) => {
                let _p = profile::span("GlobalState::handle_event/task");
                let mut prime_caches_progress = Vec::new();

                self.handle_task(&mut prime_caches_progress, task);
                // Coalesce multiple task events into one loop turn
                while let Ok(task) = self.task_pool.receiver.try_recv() {
                    self.handle_task(&mut prime_caches_progress, task);
                }

                for progress in prime_caches_progress {
                    let (state, message, fraction);
                    match progress {
                        PrimeCachesProgress::Begin => {
                            state = Progress::Begin;
                            message = None;
                            fraction = 0.0;
                        }
                        PrimeCachesProgress::Report(report) => {
                            state = Progress::Report;

                            message = match &report.crates_currently_indexing[..] {
                                [crate_name] => Some(format!(
                                    "{}/{} ({crate_name})",
                                    report.crates_done, report.crates_total
                                )),
                                [crate_name, rest @ ..] => Some(format!(
                                    "{}/{} ({} + {} more)",
                                    report.crates_done,
                                    report.crates_total,
                                    crate_name,
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

                            self.prime_caches_queue.op_completed(());
                            if cancelled {
                                self.prime_caches_queue
                                    .request_op("restart after cancellation".to_string(), ());
                            }
                        }
                    };

                    self.report_progress("Indexing", state, message, Some(fraction), None);
                }
            }
            Event::Vfs(message) => {
                let _p = profile::span("GlobalState::handle_event/vfs");
                self.handle_vfs_msg(message);
                // Coalesce many VFS event into a single loop turn
                while let Ok(message) = self.loader.receiver.try_recv() {
                    self.handle_vfs_msg(message);
                }
            }
            Event::Flycheck(message) => {
                let _p = profile::span("GlobalState::handle_event/flycheck");
                self.handle_flycheck_msg(message);
                // Coalesce many flycheck updates into a single loop turn
                while let Ok(message) = self.flycheck_receiver.try_recv() {
                    self.handle_flycheck_msg(message);
                }
            }
        }

        let state_changed = self.process_changes();
        let memdocs_added_or_removed = self.mem_docs.take_changes();

        if self.is_quiescent() {
            let became_quiescent = !(was_quiescent
                || self.fetch_workspaces_queue.op_requested()
                || self.fetch_build_data_queue.op_requested()
                || self.fetch_proc_macros_queue.op_requested());

            if became_quiescent {
                if self.config.check_on_save() {
                    // Project has loaded properly, kick off initial flycheck
                    self.flycheck.iter().for_each(FlycheckHandle::restart);
                }
                if self.config.prefill_caches() {
                    self.prime_caches_queue.request_op("became quiescent".to_string(), ());
                }
            }

            let client_refresh = !was_quiescent || state_changed;
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
            }

            let update_diagnostics = (!was_quiescent || state_changed || memdocs_added_or_removed)
                && self.config.publish_diagnostics();
            if update_diagnostics {
                self.update_diagnostics()
            }
        }

        if let Some(diagnostic_changes) = self.diagnostics.take_changes() {
            for file_id in diagnostic_changes {
                let uri = file_id_to_url(&self.vfs.read().0, file_id);
                let mut diagnostics =
                    self.diagnostics.diagnostics_for(file_id).cloned().collect::<Vec<_>>();

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
                        *message = " ".to_string();
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

                let version = from_proto::vfs_path(&uri)
                    .map(|path| self.mem_docs.get(&path).map(|it| it.version))
                    .unwrap_or_default();

                self.send_notification::<lsp_types::notification::PublishDiagnostics>(
                    lsp_types::PublishDiagnosticsParams { uri, diagnostics, version },
                );
            }
        }

        if self.config.cargo_autoreload() {
            if let Some((cause, ())) = self.fetch_workspaces_queue.should_start_op() {
                self.fetch_workspaces(cause);
            }
        }

        if !self.fetch_workspaces_queue.op_in_progress() {
            if let Some((cause, ())) = self.fetch_build_data_queue.should_start_op() {
                self.fetch_build_data(cause);
            } else if let Some((cause, paths)) = self.fetch_proc_macros_queue.should_start_op() {
                self.fetch_proc_macros(cause, paths);
            }
        }

        if let Some((cause, ())) = self.prime_caches_queue.should_start_op() {
            self.prime_caches(cause);
        }

        self.update_status_or_notify();

        let loop_duration = loop_start.elapsed();
        if loop_duration > Duration::from_millis(100) && was_quiescent {
            tracing::warn!("overly long loop turn took {loop_duration:?}: {event_dbg_msg}");
            self.poke_rust_analyzer_developer(format!(
                "overly long loop turn took {loop_duration:?}: {event_dbg_msg}"
            ));
        }
        Ok(())
    }

    fn prime_caches(&mut self, cause: String) {
        tracing::debug!(%cause, "will prime caches");
        let num_worker_threads = self.config.prime_caches_num_threads();

        self.task_pool.handle.spawn_with_sender(stdx::thread::ThreadIntent::Worker, {
            let analysis = self.snapshot().analysis;
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

    fn update_status_or_notify(&mut self) {
        let status = self.current_status();
        if self.last_reported_status.as_ref() != Some(&status) {
            self.last_reported_status = Some(status.clone());

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
            Task::Diagnostics(diagnostics_per_file) => {
                for (file_id, diagnostics) in diagnostics_per_file {
                    self.diagnostics.set_native_diagnostics(file_id, diagnostics)
                }
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
                    ProjectWorkspaceProgress::End(workspaces) => {
                        self.fetch_workspaces_queue.op_completed(Some(workspaces));
                        if let Err(e) = self.fetch_workspace_error() {
                            tracing::error!("FetchWorkspaceError:\n{e}");
                        }

                        let old = Arc::clone(&self.workspaces);
                        self.switch_workspaces("fetched workspace".to_string());
                        let workspaces_updated = !Arc::ptr_eq(&old, &self.workspaces);

                        if self.config.run_build_scripts() && workspaces_updated {
                            self.fetch_build_data_queue
                                .request_op(format!("workspace updated"), ());
                        }

                        (Progress::End, None)
                    }
                };

                self.report_progress("Fetching", state, msg, None, None);
            }
            Task::FetchBuildData(progress) => {
                let (state, msg) = match progress {
                    BuildDataProgress::Begin => (Some(Progress::Begin), None),
                    BuildDataProgress::Report(msg) => (Some(Progress::Report), Some(msg)),
                    BuildDataProgress::End(build_data_result) => {
                        self.fetch_build_data_queue.op_completed(build_data_result);
                        if let Err(e) = self.fetch_build_data_error() {
                            tracing::error!("FetchBuildDataError:\n{e}");
                        }

                        self.switch_workspaces("fetched build data".to_string());

                        (Some(Progress::End), None)
                    }
                };

                if let Some(state) = state {
                    self.report_progress("Building", state, msg, None, None);
                }
            }
            Task::LoadProcMacros(progress) => {
                let (state, msg) = match progress {
                    ProcMacroProgress::Begin => (Some(Progress::Begin), None),
                    ProcMacroProgress::Report(msg) => (Some(Progress::Report), Some(msg)),
                    ProcMacroProgress::End(proc_macro_load_result) => {
                        self.fetch_proc_macros_queue.op_completed(true);
                        self.set_proc_macros(proc_macro_load_result);

                        (Some(Progress::End), None)
                    }
                };

                if let Some(state) = state {
                    self.report_progress("Loading", state, msg, None, None);
                }
            }
        }
    }

    fn handle_vfs_msg(&mut self, message: vfs::loader::Message) {
        match message {
            vfs::loader::Message::Loaded { files } => {
                let vfs = &mut self.vfs.write().0;
                for (path, contents) in files {
                    let path = VfsPath::from(path);
                    if !self.mem_docs.contains(&path) {
                        vfs.set_file_contents(path, contents);
                    }
                }
            }
            vfs::loader::Message::Progress { n_total, n_done, config_version } => {
                always!(config_version <= self.vfs_config_version);

                self.vfs_progress_config_version = config_version;
                self.vfs_progress_n_total = n_total;
                self.vfs_progress_n_done = n_done;

                // if n_total != 0 {
                let state = if n_done == 0 {
                    Progress::Begin
                } else if n_done < n_total {
                    Progress::Report
                } else {
                    assert_eq!(n_done, n_total);
                    Progress::End
                };
                self.report_progress(
                    "Roots Scanned",
                    state,
                    Some(format!("{n_done}/{n_total}")),
                    Some(Progress::fraction(n_done, n_total)),
                    None,
                );
                // }
            }
        }
    }

    fn handle_flycheck_msg(&mut self, message: flycheck::Message) {
        match message {
            flycheck::Message::AddDiagnostic { id, workspace_root, diagnostic } => {
                let snap = self.snapshot();
                let diagnostics = crate::diagnostics::to_proto::map_rust_diagnostic_to_lsp(
                    &self.config.diagnostics_map(),
                    &diagnostic,
                    &workspace_root,
                    &snap,
                );
                for diag in diagnostics {
                    match url_to_file_id(&self.vfs.read().0, &diag.url) {
                        Ok(file_id) => self.diagnostics.add_check_diagnostic(
                            id,
                            file_id,
                            diag.diagnostic,
                            diag.fix,
                        ),
                        Err(err) => {
                            tracing::error!(
                                "flycheck {id}: File with cargo diagnostic not found in VFS: {}",
                                err
                            );
                        }
                    };
                }
            }

            flycheck::Message::Progress { id, progress } => {
                let (state, message) = match progress {
                    flycheck::Progress::DidStart => {
                        self.diagnostics.clear_check(id);
                        (Progress::Begin, None)
                    }
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
                    format!("{}", self.config.flycheck())
                } else {
                    format!("cargo check (#{})", id + 1)
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

        dispatcher
            // Request handlers that must run on the main thread
            // because they mutate GlobalState:
            .on_sync_mut::<lsp_ext::ReloadWorkspace>(handlers::handle_workspace_reload)
            .on_sync_mut::<lsp_ext::RebuildProcMacros>(handlers::handle_proc_macros_rebuild)
            .on_sync_mut::<lsp_ext::MemoryUsage>(handlers::handle_memory_usage)
            .on_sync_mut::<lsp_ext::ShuffleCrateGraph>(handlers::handle_shuffle_crate_graph)
            // Request handlers which are related to the user typing
            // are run on the main thread to reduce latency:
            .on_sync::<lsp_ext::JoinLines>(handlers::handle_join_lines)
            .on_sync::<lsp_ext::OnEnter>(handlers::handle_on_enter)
            .on_sync::<lsp_types::request::SelectionRangeRequest>(handlers::handle_selection_range)
            .on_sync::<lsp_ext::MatchingBrace>(handlers::handle_matching_brace)
            .on_sync::<lsp_ext::OnTypeFormatting>(handlers::handle_on_type_formatting)
            // We can’t run latency-sensitive request handlers which do semantic
            // analysis on the main thread because that would block other
            // requests. Instead, we run these request handlers on higher priority
            // threads in the threadpool.
            .on_latency_sensitive::<lsp_types::request::Completion>(handlers::handle_completion)
            .on_latency_sensitive::<lsp_types::request::ResolveCompletionItem>(
                handlers::handle_completion_resolve,
            )
            .on_latency_sensitive::<lsp_types::request::SemanticTokensFullRequest>(
                handlers::handle_semantic_tokens_full,
            )
            .on_latency_sensitive::<lsp_types::request::SemanticTokensFullDeltaRequest>(
                handlers::handle_semantic_tokens_full_delta,
            )
            .on_latency_sensitive::<lsp_types::request::SemanticTokensRangeRequest>(
                handlers::handle_semantic_tokens_range,
            )
            // Formatting is not caused by the user typing,
            // but it does qualify as latency-sensitive
            // because a delay before formatting is applied
            // can be confusing for the user.
            .on_latency_sensitive::<lsp_types::request::Formatting>(handlers::handle_formatting)
            .on_latency_sensitive::<lsp_types::request::RangeFormatting>(
                handlers::handle_range_formatting,
            )
            // All other request handlers
            .on::<lsp_ext::FetchDependencyList>(handlers::fetch_dependency_list)
            .on::<lsp_ext::AnalyzerStatus>(handlers::handle_analyzer_status)
            .on::<lsp_ext::SyntaxTree>(handlers::handle_syntax_tree)
            .on::<lsp_ext::ViewHir>(handlers::handle_view_hir)
            .on::<lsp_ext::ViewMir>(handlers::handle_view_mir)
            .on::<lsp_ext::InterpretFunction>(handlers::handle_interpret_function)
            .on::<lsp_ext::ViewFileText>(handlers::handle_view_file_text)
            .on::<lsp_ext::ViewCrateGraph>(handlers::handle_view_crate_graph)
            .on::<lsp_ext::ViewItemTree>(handlers::handle_view_item_tree)
            .on::<lsp_ext::ExpandMacro>(handlers::handle_expand_macro)
            .on::<lsp_ext::ParentModule>(handlers::handle_parent_module)
            .on::<lsp_ext::Runnables>(handlers::handle_runnables)
            .on::<lsp_ext::RelatedTests>(handlers::handle_related_tests)
            .on::<lsp_ext::CodeActionRequest>(handlers::handle_code_action)
            .on::<lsp_ext::CodeActionResolveRequest>(handlers::handle_code_action_resolve)
            .on::<lsp_ext::HoverRequest>(handlers::handle_hover)
            .on::<lsp_ext::ExternalDocs>(handlers::handle_open_docs)
            .on::<lsp_ext::OpenCargoToml>(handlers::handle_open_cargo_toml)
            .on::<lsp_ext::MoveItem>(handlers::handle_move_item)
            .on::<lsp_ext::WorkspaceSymbol>(handlers::handle_workspace_symbol)
            .on::<lsp_types::request::DocumentSymbolRequest>(handlers::handle_document_symbol)
            .on::<lsp_types::request::GotoDefinition>(handlers::handle_goto_definition)
            .on::<lsp_types::request::GotoDeclaration>(handlers::handle_goto_declaration)
            .on::<lsp_types::request::GotoImplementation>(handlers::handle_goto_implementation)
            .on::<lsp_types::request::GotoTypeDefinition>(handlers::handle_goto_type_definition)
            .on_no_retry::<lsp_types::request::InlayHintRequest>(handlers::handle_inlay_hints)
            .on::<lsp_types::request::InlayHintResolveRequest>(handlers::handle_inlay_hints_resolve)
            .on::<lsp_types::request::CodeLensRequest>(handlers::handle_code_lens)
            .on::<lsp_types::request::CodeLensResolve>(handlers::handle_code_lens_resolve)
            .on::<lsp_types::request::FoldingRangeRequest>(handlers::handle_folding_range)
            .on::<lsp_types::request::SignatureHelpRequest>(handlers::handle_signature_help)
            .on::<lsp_types::request::PrepareRenameRequest>(handlers::handle_prepare_rename)
            .on::<lsp_types::request::Rename>(handlers::handle_rename)
            .on::<lsp_types::request::References>(handlers::handle_references)
            .on::<lsp_types::request::DocumentHighlightRequest>(handlers::handle_document_highlight)
            .on::<lsp_types::request::CallHierarchyPrepare>(handlers::handle_call_hierarchy_prepare)
            .on::<lsp_types::request::CallHierarchyIncomingCalls>(
                handlers::handle_call_hierarchy_incoming,
            )
            .on::<lsp_types::request::CallHierarchyOutgoingCalls>(
                handlers::handle_call_hierarchy_outgoing,
            )
            .on::<lsp_types::request::WillRenameFiles>(handlers::handle_will_rename_files)
            .on::<lsp_ext::Ssr>(handlers::handle_ssr)
            .finish();
    }

    /// Handles an incoming notification.
    fn on_notification(&mut self, not: Notification) -> Result<()> {
        use crate::handlers::notification as handlers;
        use lsp_types::notification as notifs;

        NotificationDispatcher { not: Some(not), global_state: self }
            .on::<notifs::Cancel>(handlers::handle_cancel)?
            .on::<notifs::WorkDoneProgressCancel>(handlers::handle_work_done_progress_cancel)?
            .on::<notifs::DidOpenTextDocument>(handlers::handle_did_open_text_document)?
            .on::<notifs::DidChangeTextDocument>(handlers::handle_did_change_text_document)?
            .on::<notifs::DidCloseTextDocument>(handlers::handle_did_close_text_document)?
            .on::<notifs::DidSaveTextDocument>(handlers::handle_did_save_text_document)?
            .on::<notifs::DidChangeConfiguration>(handlers::handle_did_change_configuration)?
            .on::<notifs::DidChangeWorkspaceFolders>(handlers::handle_did_change_workspace_folders)?
            .on::<notifs::DidChangeWatchedFiles>(handlers::handle_did_change_watched_files)?
            .on::<lsp_ext::CancelFlycheck>(handlers::handle_cancel_flycheck)?
            .on::<lsp_ext::ClearFlycheck>(handlers::handle_clear_flycheck)?
            .on::<lsp_ext::RunFlycheck>(handlers::handle_run_flycheck)?
            .finish();
        Ok(())
    }

    fn update_diagnostics(&mut self) {
        let db = self.analysis_host.raw_database();
        let subscriptions = self
            .mem_docs
            .iter()
            .map(|path| self.vfs.read().0.file_id(path).unwrap())
            .filter(|&file_id| {
                let source_root = db.file_source_root(file_id);
                // Only publish diagnostics for files in the workspace, not from crates.io deps
                // or the sysroot.
                // While theoretically these should never have errors, we have quite a few false
                // positives particularly in the stdlib, and those diagnostics would stay around
                // forever if we emitted them here.
                !db.source_root(source_root).is_library
            })
            .collect::<Vec<_>>();

        tracing::trace!("updating notifications for {:?}", subscriptions);

        let snapshot = self.snapshot();

        // Diagnostics are triggered by the user typing
        // so we run them on a latency sensitive thread.
        self.task_pool.handle.spawn(stdx::thread::ThreadIntent::LatencySensitive, move || {
            let _p = profile::span("publish_diagnostics");
            let _ctx = stdx::panic_context::enter("publish_diagnostics".to_owned());
            let diagnostics = subscriptions
                .into_iter()
                .filter_map(|file_id| {
                    let line_index = snapshot.file_line_index(file_id).ok()?;
                    Some((
                        file_id,
                        line_index,
                        snapshot
                            .analysis
                            .diagnostics(
                                &snapshot.config.diagnostics(),
                                ide::AssistResolveStrategy::None,
                                file_id,
                            )
                            .ok()?,
                    ))
                })
                .map(|(file_id, line_index, it)| {
                    (
                        file_id,
                        it.into_iter()
                            .map(move |d| lsp_types::Diagnostic {
                                range: crate::to_proto::range(&line_index, d.range),
                                severity: Some(crate::to_proto::diagnostic_severity(d.severity)),
                                code: Some(lsp_types::NumberOrString::String(
                                    d.code.as_str().to_string(),
                                )),
                                code_description: Some(lsp_types::CodeDescription {
                                    href: lsp_types::Url::parse(&format!(
                                        "https://rust-analyzer.github.io/manual.html#{}",
                                        d.code.as_str()
                                    ))
                                    .unwrap(),
                                }),
                                source: Some("rust-analyzer".to_string()),
                                message: d.message,
                                related_information: None,
                                tags: if d.unused {
                                    Some(vec![lsp_types::DiagnosticTag::UNNECESSARY])
                                } else {
                                    None
                                },
                                data: None,
                            })
                            .collect::<Vec<_>>(),
                    )
                });
            Task::Diagnostics(diagnostics.collect())
        });
    }
}
