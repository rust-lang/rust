//! The main loop of `rust-analyzer` responsible for dispatching LSP
//! requests/replies and notifications back to the client.
use std::{
    env, fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use always_assert::always;
use crossbeam_channel::{select, Receiver};
use ide::PrimeCachesProgress;
use ide::{Canceled, FileId};
use ide_db::base_db::VfsPath;
use lsp_server::{Connection, Notification, Request, Response};
use lsp_types::notification::Notification as _;
use project_model::BuildDataCollector;
use vfs::ChangeKind;

use crate::{
    config::Config,
    dispatch::{NotificationDispatcher, RequestDispatcher},
    document::DocumentData,
    from_proto,
    global_state::{file_id_to_url, url_to_file_id, GlobalState, Status},
    handlers, lsp_ext,
    lsp_utils::{apply_document_changes, is_canceled, notification_is, Progress},
    reload::{BuildDataProgress, ProjectWorkspaceProgress},
    Result,
};

pub fn main_loop(config: Config, connection: Connection) -> Result<()> {
    log::info!("initial config: {:#?}", config);

    // Windows scheduler implements priority boosts: if thread waits for an
    // event (like a condvar), and event fires, priority of the thread is
    // temporary bumped. This optimization backfires in our case: each time the
    // `main_loop` schedules a task to run on a threadpool, the worker threads
    // gets a higher priority, and (on a machine with fewer cores) displaces the
    // main loop! We work-around this by marking the main loop as a
    // higher-priority thread.
    //
    // https://docs.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities
    // https://docs.microsoft.com/en-us/windows/win32/procthread/priority-boosts
    // https://github.com/rust-analyzer/rust-analyzer/issues/2835
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
    Response(Response),
    Diagnostics(Vec<(FileId, Vec<lsp_types::Diagnostic>)>),
    PrimeCaches(PrimeCachesProgress),
    FetchWorkspace(ProjectWorkspaceProgress),
    FetchBuildData(BuildDataProgress),
}

impl fmt::Debug for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let debug_verbose_not = |not: &Notification, f: &mut fmt::Formatter| {
            f.debug_struct("Notification").field("method", &not.method).finish()
        };

        match self {
            Event::Lsp(lsp_server::Message::Notification(not)) => {
                if notification_is::<lsp_types::notification::DidOpenTextDocument>(not)
                    || notification_is::<lsp_types::notification::DidChangeTextDocument>(not)
                {
                    return debug_verbose_not(not, f);
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
        if self.config.linked_projects().is_empty()
            && self.config.notifications().cargo_toml_not_found
        {
            self.show_message(
                lsp_types::MessageType::Error,
                "rust-analyzer failed to discover workspace".to_string(),
            );
        };

        if self.config.did_save_text_document_dynamic_registration() {
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

        self.fetch_workspaces_request();
        self.fetch_workspaces_if_needed();

        while let Some(event) = self.next_event(&inbox) {
            if let Event::Lsp(lsp_server::Message::Notification(not)) = &event {
                if not.method == lsp_types::notification::Exit::METHOD {
                    return Ok(());
                }
            }
            self.handle_event(event)?
        }

        Err("client exited without proper shutdown sequence")?
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

        log::info!("handle_event({:?})", event);
        let task_queue_len = self.task_pool.handle.len();
        if task_queue_len > 0 {
            log::info!("task queue len: {}", task_queue_len);
        }

        let mut new_status = self.status;
        match event {
            Event::Lsp(msg) => match msg {
                lsp_server::Message::Request(req) => self.on_request(loop_start, req)?,
                lsp_server::Message::Notification(not) => {
                    self.on_notification(not)?;
                }
                lsp_server::Message::Response(resp) => self.complete_request(resp),
            },
            Event::Task(mut task) => {
                let _p = profile::span("GlobalState::handle_event/task");
                let mut prime_caches_progress = Vec::new();
                loop {
                    match task {
                        Task::Response(response) => self.respond(response),
                        Task::Diagnostics(diagnostics_per_file) => {
                            for (file_id, diagnostics) in diagnostics_per_file {
                                self.diagnostics.set_native_diagnostics(file_id, diagnostics)
                            }
                        }
                        Task::PrimeCaches(progress) => match progress {
                            PrimeCachesProgress::Started => prime_caches_progress.push(progress),
                            PrimeCachesProgress::StartedOnCrate { .. } => {
                                match prime_caches_progress.last_mut() {
                                    Some(last @ PrimeCachesProgress::StartedOnCrate { .. }) => {
                                        // Coalesce subsequent update events.
                                        *last = progress;
                                    }
                                    _ => prime_caches_progress.push(progress),
                                }
                            }
                            PrimeCachesProgress::Finished => prime_caches_progress.push(progress),
                        },
                        Task::FetchWorkspace(progress) => {
                            let (state, msg) = match progress {
                                ProjectWorkspaceProgress::Begin => (Progress::Begin, None),
                                ProjectWorkspaceProgress::Report(msg) => {
                                    (Progress::Report, Some(msg))
                                }
                                ProjectWorkspaceProgress::End(workspaces) => {
                                    self.fetch_workspaces_completed(workspaces);

                                    let old = Arc::clone(&self.workspaces);
                                    self.switch_workspaces();
                                    let workspaces_updated = !Arc::ptr_eq(&old, &self.workspaces);

                                    if self.config.run_build_scripts() && workspaces_updated {
                                        let mut collector = BuildDataCollector::default();
                                        for ws in self.workspaces.iter() {
                                            ws.collect_build_data_configs(&mut collector);
                                        }
                                        self.fetch_build_data_request(collector)
                                    }

                                    (Progress::End, None)
                                }
                            };

                            self.report_progress("fetching", state, msg, None);
                        }
                        Task::FetchBuildData(progress) => {
                            let (state, msg) = match progress {
                                BuildDataProgress::Begin => (Some(Progress::Begin), None),
                                BuildDataProgress::Report(msg) => {
                                    (Some(Progress::Report), Some(msg))
                                }
                                BuildDataProgress::End(build_data_result) => {
                                    self.fetch_build_data_completed(build_data_result);

                                    self.switch_workspaces();

                                    (Some(Progress::End), None)
                                }
                            };

                            if let Some(state) = state {
                                self.report_progress("loading", state, msg, None);
                            }
                        }
                    }

                    // Coalesce multiple task events into one loop turn
                    task = match self.task_pool.receiver.try_recv() {
                        Ok(task) => task,
                        Err(_) => break,
                    };
                }

                for progress in prime_caches_progress {
                    let (state, message, fraction);
                    match progress {
                        PrimeCachesProgress::Started => {
                            state = Progress::Begin;
                            message = None;
                            fraction = 0.0;
                        }
                        PrimeCachesProgress::StartedOnCrate { on_crate, n_done, n_total } => {
                            state = Progress::Report;
                            message = Some(format!("{}/{} ({})", n_done, n_total, on_crate));
                            fraction = Progress::fraction(n_done, n_total);
                        }
                        PrimeCachesProgress::Finished => {
                            state = Progress::End;
                            message = None;
                            fraction = 1.0;
                        }
                    };

                    self.report_progress("indexing", state, message, Some(fraction));
                }
            }
            Event::Vfs(mut task) => {
                let _p = profile::span("GlobalState::handle_event/vfs");
                loop {
                    match task {
                        vfs::loader::Message::Loaded { files } => {
                            let vfs = &mut self.vfs.write().0;
                            for (path, contents) in files {
                                let path = VfsPath::from(path);
                                if !self.mem_docs.contains_key(&path) {
                                    vfs.set_file_contents(path, contents);
                                }
                            }
                        }
                        vfs::loader::Message::Progress { n_total, n_done, config_version } => {
                            always!(config_version <= self.vfs_config_version);
                            if n_total == 0 {
                                new_status = Status::Invalid;
                            } else {
                                let state = if n_done == 0 {
                                    new_status = Status::Loading;
                                    Progress::Begin
                                } else if n_done < n_total {
                                    Progress::Report
                                } else {
                                    assert_eq!(n_done, n_total);
                                    new_status = Status::Ready {
                                        partial: self.config.run_build_scripts()
                                            && self.workspace_build_data.is_none()
                                            || config_version < self.vfs_config_version,
                                    };
                                    Progress::End
                                };
                                self.report_progress(
                                    "roots scanned",
                                    state,
                                    Some(format!("{}/{}", n_done, n_total)),
                                    Some(Progress::fraction(n_done, n_total)),
                                )
                            }
                        }
                    }
                    // Coalesce many VFS event into a single loop turn
                    task = match self.loader.receiver.try_recv() {
                        Ok(task) => task,
                        Err(_) => break,
                    }
                }
            }
            Event::Flycheck(mut task) => {
                let _p = profile::span("GlobalState::handle_event/flycheck");
                loop {
                    match task {
                        flycheck::Message::AddDiagnostic { workspace_root, diagnostic } => {
                            let diagnostics =
                                crate::diagnostics::to_proto::map_rust_diagnostic_to_lsp(
                                    &self.config.diagnostics_map(),
                                    &diagnostic,
                                    &workspace_root,
                                );
                            for diag in diagnostics {
                                match url_to_file_id(&self.vfs.read().0, &diag.url) {
                                    Ok(file_id) => self.diagnostics.add_check_diagnostic(
                                        file_id,
                                        diag.diagnostic,
                                        diag.fixes,
                                    ),
                                    Err(err) => {
                                        log::error!(
                                            "File with cargo diagnostic not found in VFS: {}",
                                            err
                                        );
                                    }
                                };
                            }
                        }

                        flycheck::Message::Progress { id, progress } => {
                            let (state, message) = match progress {
                                flycheck::Progress::DidStart => {
                                    self.diagnostics.clear_check();
                                    (Progress::Begin, None)
                                }
                                flycheck::Progress::DidCheckCrate(target) => {
                                    (Progress::Report, Some(target))
                                }
                                flycheck::Progress::DidCancel => (Progress::End, None),
                                flycheck::Progress::DidFinish(result) => {
                                    if let Err(err) = result {
                                        log::error!("cargo check failed: {}", err)
                                    }
                                    (Progress::End, None)
                                }
                            };

                            // When we're running multiple flychecks, we have to include a disambiguator in
                            // the title, or the editor complains. Note that this is a user-facing string.
                            let title = if self.flycheck.len() == 1 {
                                "cargo check".to_string()
                            } else {
                                format!("cargo check (#{})", id + 1)
                            };
                            self.report_progress(&title, state, message, None);
                        }
                    }
                    // Coalesce many flycheck updates into a single loop turn
                    task = match self.flycheck_receiver.try_recv() {
                        Ok(task) => task,
                        Err(_) => break,
                    }
                }
            }
        }

        let state_changed = self.process_changes();
        let prev_status = self.status;
        if prev_status != new_status {
            self.transition(new_status);
        }
        let is_ready = matches!(self.status, Status::Ready { .. });
        if prev_status == Status::Loading && is_ready {
            for flycheck in &self.flycheck {
                flycheck.update();
            }
        }

        if is_ready && (state_changed || prev_status == Status::Loading) {
            self.update_file_notifications_on_threadpool();

            // Refresh semantic tokens if the client supports it.
            if self.config.semantic_tokens_refresh() {
                self.semantic_tokens_cache.lock().clear();
                self.send_request::<lsp_types::request::SemanticTokensRefesh>((), |_, _| ());
            }

            // Refresh code lens if the client supports it.
            if self.config.code_lens_refresh() {
                self.send_request::<lsp_types::request::CodeLensRefresh>((), |_, _| ());
            }
        }

        if let Some(diagnostic_changes) = self.diagnostics.take_changes() {
            for file_id in diagnostic_changes {
                let url = file_id_to_url(&self.vfs.read().0, file_id);
                let diagnostics = self.diagnostics.diagnostics_for(file_id).cloned().collect();
                let version = from_proto::vfs_path(&url)
                    .map(|path| self.mem_docs.get(&path).map(|it| it.version))
                    .unwrap_or_default();

                self.send_notification::<lsp_types::notification::PublishDiagnostics>(
                    lsp_types::PublishDiagnosticsParams { uri: url, diagnostics, version },
                );
            }
        }

        if self.config.cargo_autoreload() {
            self.fetch_workspaces_if_needed();
        }
        self.fetch_build_data_if_needed();

        let loop_duration = loop_start.elapsed();
        if loop_duration > Duration::from_millis(100) {
            log::warn!("overly long loop turn: {:?}", loop_duration);
            if env::var("RA_PROFILE").is_ok() {
                self.show_message(
                    lsp_types::MessageType::Error,
                    format!("overly long loop turn: {:?}", loop_duration),
                )
            }
        }
        Ok(())
    }

    fn on_request(&mut self, request_received: Instant, req: Request) -> Result<()> {
        self.register_request(&req, request_received);

        if self.shutdown_requested {
            self.respond(Response::new_err(
                req.id,
                lsp_server::ErrorCode::InvalidRequest as i32,
                "Shutdown already requested.".to_owned(),
            ));

            return Ok(());
        }

        if self.status == Status::Loading && req.method != "shutdown" {
            self.respond(lsp_server::Response::new_err(
                req.id,
                // FIXME: i32 should impl From<ErrorCode> (from() guarantees lossless conversion)
                lsp_server::ErrorCode::ContentModified as i32,
                "Rust Analyzer is still loading...".to_owned(),
            ));
            return Ok(());
        }

        RequestDispatcher { req: Some(req), global_state: self }
            .on_sync::<lsp_ext::ReloadWorkspace>(|s, ()| {
                self.fetch_workspaces_request();
                self.fetch_workspaces_if_needed();
            })?
            .on_sync::<lsp_ext::JoinLines>(|s, p| handlers::handle_join_lines(s.snapshot(), p))?
            .on_sync::<lsp_ext::OnEnter>(|s, p| handlers::handle_on_enter(s.snapshot(), p))?
            .on_sync::<lsp_types::request::Shutdown>(|s, ()| {
                s.shutdown_requested = true;
                Ok(())
            })?
            .on_sync::<lsp_types::request::SelectionRangeRequest>(|s, p| {
                handlers::handle_selection_range(s.snapshot(), p)
            })?
            .on_sync::<lsp_ext::MatchingBrace>(|s, p| {
                handlers::handle_matching_brace(s.snapshot(), p)
            })?
            .on_sync::<lsp_ext::MemoryUsage>(|s, p| handlers::handle_memory_usage(s, p))?
            .on::<lsp_ext::AnalyzerStatus>(handlers::handle_analyzer_status)
            .on::<lsp_ext::SyntaxTree>(handlers::handle_syntax_tree)
            .on::<lsp_ext::ViewHir>(handlers::handle_view_hir)
            .on::<lsp_ext::ExpandMacro>(handlers::handle_expand_macro)
            .on::<lsp_ext::ParentModule>(handlers::handle_parent_module)
            .on::<lsp_ext::Runnables>(handlers::handle_runnables)
            .on::<lsp_ext::RelatedTests>(handlers::handle_related_tests)
            .on::<lsp_ext::InlayHints>(handlers::handle_inlay_hints)
            .on::<lsp_ext::CodeActionRequest>(handlers::handle_code_action)
            .on::<lsp_ext::CodeActionResolveRequest>(handlers::handle_code_action_resolve)
            .on::<lsp_ext::HoverRequest>(handlers::handle_hover)
            .on::<lsp_ext::ExternalDocs>(handlers::handle_open_docs)
            .on::<lsp_ext::OpenCargoToml>(handlers::handle_open_cargo_toml)
            .on::<lsp_ext::MoveItem>(handlers::handle_move_item)
            .on::<lsp_types::request::OnTypeFormatting>(handlers::handle_on_type_formatting)
            .on::<lsp_types::request::DocumentSymbolRequest>(handlers::handle_document_symbol)
            .on::<lsp_types::request::WorkspaceSymbol>(handlers::handle_workspace_symbol)
            .on::<lsp_types::request::GotoDefinition>(handlers::handle_goto_definition)
            .on::<lsp_types::request::GotoImplementation>(handlers::handle_goto_implementation)
            .on::<lsp_types::request::GotoTypeDefinition>(handlers::handle_goto_type_definition)
            .on::<lsp_types::request::Completion>(handlers::handle_completion)
            .on::<lsp_types::request::ResolveCompletionItem>(handlers::handle_completion_resolve)
            .on::<lsp_types::request::CodeLensRequest>(handlers::handle_code_lens)
            .on::<lsp_types::request::CodeLensResolve>(handlers::handle_code_lens_resolve)
            .on::<lsp_types::request::FoldingRangeRequest>(handlers::handle_folding_range)
            .on::<lsp_types::request::SignatureHelpRequest>(handlers::handle_signature_help)
            .on::<lsp_types::request::PrepareRenameRequest>(handlers::handle_prepare_rename)
            .on::<lsp_types::request::Rename>(handlers::handle_rename)
            .on::<lsp_types::request::References>(handlers::handle_references)
            .on::<lsp_types::request::Formatting>(handlers::handle_formatting)
            .on::<lsp_types::request::DocumentHighlightRequest>(handlers::handle_document_highlight)
            .on::<lsp_types::request::CallHierarchyPrepare>(handlers::handle_call_hierarchy_prepare)
            .on::<lsp_types::request::CallHierarchyIncomingCalls>(
                handlers::handle_call_hierarchy_incoming,
            )
            .on::<lsp_types::request::CallHierarchyOutgoingCalls>(
                handlers::handle_call_hierarchy_outgoing,
            )
            .on::<lsp_types::request::SemanticTokensFullRequest>(
                handlers::handle_semantic_tokens_full,
            )
            .on::<lsp_types::request::SemanticTokensFullDeltaRequest>(
                handlers::handle_semantic_tokens_full_delta,
            )
            .on::<lsp_types::request::SemanticTokensRangeRequest>(
                handlers::handle_semantic_tokens_range,
            )
            .on::<lsp_types::request::WillRenameFiles>(handlers::handle_will_rename_files)
            .on::<lsp_ext::Ssr>(handlers::handle_ssr)
            .finish();
        Ok(())
    }
    fn on_notification(&mut self, not: Notification) -> Result<()> {
        NotificationDispatcher { not: Some(not), global_state: self }
            .on::<lsp_types::notification::Cancel>(|this, params| {
                let id: lsp_server::RequestId = match params.id {
                    lsp_types::NumberOrString::Number(id) => id.into(),
                    lsp_types::NumberOrString::String(id) => id.into(),
                };
                this.cancel(id);
                Ok(())
            })?
            .on::<lsp_types::notification::DidOpenTextDocument>(|this, params| {
                if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                    if this
                        .mem_docs
                        .insert(path.clone(), DocumentData::new(params.text_document.version))
                        .is_some()
                    {
                        log::error!("duplicate DidOpenTextDocument: {}", path)
                    }
                    let changed = this
                        .vfs
                        .write()
                        .0
                        .set_file_contents(path, Some(params.text_document.text.into_bytes()));

                    // If the VFS contents are unchanged, update diagnostics, since `handle_event`
                    // won't see any changes. This avoids missing diagnostics when opening a file.
                    //
                    // If the file *was* changed, `handle_event` will already recompute and send
                    // diagnostics. We can't do it here, since the *current* file contents might be
                    // unset in salsa, since the VFS change hasn't been applied to the database yet.
                    if !changed {
                        this.maybe_update_diagnostics();
                    }
                }
                Ok(())
            })?
            .on::<lsp_types::notification::DidChangeTextDocument>(|this, params| {
                if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                    let doc = match this.mem_docs.get_mut(&path) {
                        Some(doc) => doc,
                        None => {
                            log::error!("expected DidChangeTextDocument: {}", path);
                            return Ok(());
                        }
                    };
                    let vfs = &mut this.vfs.write().0;
                    let file_id = vfs.file_id(&path).unwrap();
                    let mut text = String::from_utf8(vfs.file_contents(file_id).to_vec()).unwrap();
                    apply_document_changes(&mut text, params.content_changes);

                    // The version passed in DidChangeTextDocument is the version after all edits are applied
                    // so we should apply it before the vfs is notified.
                    doc.version = params.text_document.version;

                    vfs.set_file_contents(path.clone(), Some(text.into_bytes()));
                }
                Ok(())
            })?
            .on::<lsp_types::notification::DidCloseTextDocument>(|this, params| {
                let mut version = None;
                if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                    match this.mem_docs.remove(&path) {
                        Some(doc) => version = Some(doc.version),
                        None => log::error!("orphan DidCloseTextDocument: {}", path),
                    }

                    this.semantic_tokens_cache.lock().remove(&params.text_document.uri);

                    if let Some(path) = path.as_path() {
                        this.loader.handle.invalidate(path.to_path_buf());
                    }
                }

                // Clear the diagnostics for the previously known version of the file.
                // This prevents stale "cargo check" diagnostics if the file is
                // closed, "cargo check" is run and then the file is reopened.
                this.send_notification::<lsp_types::notification::PublishDiagnostics>(
                    lsp_types::PublishDiagnosticsParams {
                        uri: params.text_document.uri,
                        diagnostics: Vec::new(),
                        version,
                    },
                );
                Ok(())
            })?
            .on::<lsp_types::notification::DidSaveTextDocument>(|this, params| {
                for flycheck in &this.flycheck {
                    flycheck.update();
                }
                if let Ok(abs_path) = from_proto::abs_path(&params.text_document.uri) {
                    this.maybe_refresh(&[(abs_path, ChangeKind::Modify)]);
                }
                Ok(())
            })?
            .on::<lsp_types::notification::DidChangeConfiguration>(|this, _params| {
                // As stated in https://github.com/microsoft/language-server-protocol/issues/676,
                // this notification's parameters should be ignored and the actual config queried separately.
                this.send_request::<lsp_types::request::WorkspaceConfiguration>(
                    lsp_types::ConfigurationParams {
                        items: vec![lsp_types::ConfigurationItem {
                            scope_uri: None,
                            section: Some("rust-analyzer".to_string()),
                        }],
                    },
                    |this, resp| {
                        log::debug!("config update response: '{:?}", resp);
                        let Response { error, result, .. } = resp;

                        match (error, result) {
                            (Some(err), _) => {
                                log::error!("failed to fetch the server settings: {:?}", err)
                            }
                            (None, Some(mut configs)) => {
                                if let Some(json) = configs.get_mut(0) {
                                    // Note that json can be null according to the spec if the client can't
                                    // provide a configuration. This is handled in Config::update below.
                                    let mut config = Config::clone(&*this.config);
                                    config.update(json.take());
                                    this.update_configuration(config);
                                }
                            }
                            (None, None) => log::error!(
                                "received empty server settings response from the client"
                            ),
                        }
                    },
                );

                return Ok(());
            })?
            .on::<lsp_types::notification::DidChangeWatchedFiles>(|this, params| {
                for change in params.changes {
                    if let Ok(path) = from_proto::abs_path(&change.uri) {
                        this.loader.handle.invalidate(path);
                    }
                }
                Ok(())
            })?
            .finish();
        Ok(())
    }
    fn update_file_notifications_on_threadpool(&mut self) {
        self.maybe_update_diagnostics();
        self.task_pool.handle.spawn_with_sender({
            let snap = self.snapshot();
            move |sender| {
                snap.analysis
                    .prime_caches(|progress| {
                        sender.send(Task::PrimeCaches(progress)).unwrap();
                    })
                    .unwrap_or_else(|_: Canceled| {
                        // Pretend that we're done, so that the progress bar is removed. Otherwise
                        // the editor may complain about it already existing.
                        sender.send(Task::PrimeCaches(PrimeCachesProgress::Finished)).unwrap()
                    });
            }
        });
    }
    fn maybe_update_diagnostics(&mut self) {
        let subscriptions = self
            .mem_docs
            .keys()
            .map(|path| self.vfs.read().0.file_id(&path).unwrap())
            .collect::<Vec<_>>();

        log::trace!("updating notifications for {:?}", subscriptions);
        if self.config.publish_diagnostics() {
            let snapshot = self.snapshot();
            self.task_pool.handle.spawn(move || {
                let diagnostics = subscriptions
                    .into_iter()
                    .filter_map(|file_id| {
                        handlers::publish_diagnostics(&snapshot, file_id)
                            .map_err(|err| {
                                if !is_canceled(&*err) {
                                    log::error!("failed to compute diagnostics: {:?}", err);
                                }
                                ()
                            })
                            .ok()
                            .map(|diags| (file_id, diags))
                    })
                    .collect::<Vec<_>>();
                Task::Diagnostics(diagnostics)
            })
        }
    }
}
