//! The main loop of `rust-analyzer` responsible for dispatching LSP
//! requests/replies and notifications back to the client.
use std::{
    env, fmt, panic,
    time::{Duration, Instant},
};

use crossbeam_channel::{never, select, Receiver};
use lsp_server::{Connection, Notification, Request, Response};
use lsp_types::{notification::Notification as _, request::Request as _};
use ra_db::VfsPath;
use ra_ide::{Canceled, FileId};
use ra_prof::profile;
use ra_project_model::{PackageRoot, ProjectWorkspace};

use crate::{
    config::{Config, FilesWatcher, LinkedProject},
    diagnostics::DiagnosticTask,
    dispatch::{NotificationDispatcher, RequestDispatcher},
    from_proto,
    global_state::{file_id_to_url, GlobalState, Status},
    handlers, lsp_ext,
    lsp_utils::{
        apply_document_changes, is_canceled, notification_is, notification_new, show_message,
    },
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

    let global_state = {
        let workspaces = {
            if config.linked_projects.is_empty() && config.notifications.cargo_toml_not_found {
                show_message(
                    lsp_types::MessageType::Error,
                    "rust-analyzer failed to discover workspace".to_string(),
                    &connection.sender,
                );
            };

            config
                .linked_projects
                .iter()
                .filter_map(|project| match project {
                    LinkedProject::ProjectManifest(manifest) => {
                        ra_project_model::ProjectWorkspace::load(
                            manifest.clone(),
                            &config.cargo,
                            config.with_sysroot,
                        )
                        .map_err(|err| {
                            log::error!("failed to load workspace: {:#}", err);
                            show_message(
                                lsp_types::MessageType::Error,
                                format!("rust-analyzer failed to load workspace: {:#}", err),
                                &connection.sender,
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

        let mut req_queue = ReqQueue::default();

        if let FilesWatcher::Client = config.files.watcher {
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
            let request = req_queue.outgoing.register(
                lsp_types::request::RegisterCapability::METHOD.to_string(),
                params,
                DO_NOTHING,
            );
            connection.sender.send(request.into()).unwrap();
        }

        GlobalState::new(
            connection.sender.clone(),
            workspaces,
            config.lru_capacity,
            config,
            req_queue,
        )
    };

    log::info!("server initialized, serving requests");
    global_state.run(connection.receiver)?;
    Ok(())
}

enum Event {
    Lsp(lsp_server::Message),
    Task(Task),
    Vfs(vfs::loader::Message),
    Flycheck(flycheck::Message),
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
    fn next_event(&self, inbox: &Receiver<lsp_server::Message>) -> Option<Event> {
        select! {
            recv(inbox) -> msg =>
                msg.ok().map(Event::Lsp),

            recv(self.task_pool.1) -> task =>
                Some(Event::Task(task.unwrap())),

            recv(self.task_receiver) -> task =>
                Some(Event::Vfs(task.unwrap())),

            recv(self.flycheck.as_ref().map_or(&never(), |it| &it.1)) -> task =>
                Some(Event::Flycheck(task.unwrap())),
        }
    }

    fn run(mut self, inbox: Receiver<lsp_server::Message>) -> Result<()> {
        while let Some(event) = self.next_event(&inbox) {
            if let Event::Lsp(lsp_server::Message::Notification(not)) = &event {
                if not.method == lsp_types::notification::Exit::METHOD {
                    return Ok(());
                }
            }
            self.loop_turn(event)?
        }
        Err("client exited without proper shutdown sequence")?
    }

    fn loop_turn(&mut self, event: Event) -> Result<()> {
        let loop_start = Instant::now();
        // NOTE: don't count blocking select! call as a loop-turn time
        let _p = profile("main_loop_inner/loop-turn");

        log::info!("loop turn = {:?}", event);
        let queue_count = self.task_pool.0.len();
        if queue_count > 0 {
            log::info!("queued count = {}", queue_count);
        }

        let mut became_ready = false;
        match event {
            Event::Lsp(msg) => match msg {
                lsp_server::Message::Request(req) => self.on_request(loop_start, req)?,
                lsp_server::Message::Notification(not) => {
                    self.on_notification(not)?;
                }
                lsp_server::Message::Response(resp) => {
                    let handler = self.req_queue.outgoing.complete(resp.id.clone());
                    handler(self, resp)
                }
            },
            Event::Task(task) => {
                match task {
                    Task::Response(response) => self.respond(response),
                    Task::Diagnostics(tasks) => {
                        tasks.into_iter().for_each(|task| on_diagnostic_task(task, self))
                    }
                    Task::Unit => (),
                }
                self.maybe_collect_garbage();
            }
            Event::Vfs(task) => match task {
                vfs::loader::Message::Loaded { files } => {
                    let vfs = &mut self.vfs.write().0;
                    for (path, contents) in files {
                        let path = VfsPath::from(path);
                        if !self.mem_docs.contains(&path) {
                            vfs.set_file_contents(path, contents)
                        }
                    }
                }
                vfs::loader::Message::Progress { n_total, n_done } => {
                    let state = if n_done == 0 {
                        Progress::Begin
                    } else if n_done < n_total {
                        Progress::Report
                    } else {
                        assert_eq!(n_done, n_total);
                        self.status = Status::Ready;
                        became_ready = true;
                        Progress::End
                    };
                    report_progress(
                        self,
                        "roots scanned",
                        state,
                        Some(format!("{}/{}", n_done, n_total)),
                        Some(percentage(n_done, n_total)),
                    )
                }
            },
            Event::Flycheck(task) => match task {
                flycheck::Message::ClearDiagnostics => {
                    on_diagnostic_task(DiagnosticTask::ClearCheck, self)
                }

                flycheck::Message::AddDiagnostic { workspace_root, diagnostic } => {
                    let diagnostics = crate::diagnostics::to_proto::map_rust_diagnostic_to_lsp(
                        &self.config.diagnostics,
                        &diagnostic,
                        &workspace_root,
                    );
                    for diag in diagnostics {
                        let path = from_proto::vfs_path(&diag.location.uri)?;
                        let file_id = match self.vfs.read().0.file_id(&path) {
                            Some(file) => FileId(file.0),
                            None => {
                                log::error!(
                                    "File with cargo diagnostic not found in VFS: {}",
                                    path
                                );
                                return Ok(());
                            }
                        };

                        on_diagnostic_task(
                            DiagnosticTask::AddCheck(
                                file_id,
                                diag.diagnostic,
                                diag.fixes.into_iter().map(|it| it.into()).collect(),
                            ),
                            self,
                        )
                    }
                }

                flycheck::Message::Progress(status) => {
                    let (state, message) = match status {
                        flycheck::Progress::Being => (Progress::Begin, None),
                        flycheck::Progress::DidCheckCrate(target) => {
                            (Progress::Report, Some(target))
                        }
                        flycheck::Progress::End => (Progress::End, None),
                    };

                    report_progress(self, "cargo check", state, message, None);
                }
            },
        }

        let state_changed = self.process_changes();
        if became_ready {
            if let Some(flycheck) = &self.flycheck {
                flycheck.0.update();
            }
        }

        if self.status == Status::Ready && (state_changed || became_ready) {
            let subscriptions = self
                .mem_docs
                .iter()
                .map(|path| self.vfs.read().0.file_id(&path).unwrap())
                .collect::<Vec<_>>();

            self.update_file_notifications_on_threadpool(subscriptions);
        }

        let loop_duration = loop_start.elapsed();
        if loop_duration > Duration::from_millis(100) {
            log::error!("overly long loop turn: {:?}", loop_duration);
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
        self.req_queue.incoming.register(req.id.clone(), (req.method.clone(), request_received));

        RequestDispatcher { req: Some(req), global_state: self }
            .on_sync::<lsp_ext::CollectGarbage>(|s, ()| Ok(s.collect_garbage()))?
            .on_sync::<lsp_ext::JoinLines>(|s, p| handlers::handle_join_lines(s.snapshot(), p))?
            .on_sync::<lsp_ext::OnEnter>(|s, p| handlers::handle_on_enter(s.snapshot(), p))?
            .on_sync::<lsp_types::request::Shutdown>(|_, ()| Ok(()))?
            .on_sync::<lsp_types::request::SelectionRangeRequest>(|s, p| {
                handlers::handle_selection_range(s.snapshot(), p)
            })?
            .on_sync::<lsp_ext::MatchingBrace>(|s, p| {
                handlers::handle_matching_brace(s.snapshot(), p)
            })?
            .on::<lsp_ext::AnalyzerStatus>(handlers::handle_analyzer_status)?
            .on::<lsp_ext::SyntaxTree>(handlers::handle_syntax_tree)?
            .on::<lsp_ext::ExpandMacro>(handlers::handle_expand_macro)?
            .on::<lsp_ext::ParentModule>(handlers::handle_parent_module)?
            .on::<lsp_ext::Runnables>(handlers::handle_runnables)?
            .on::<lsp_ext::InlayHints>(handlers::handle_inlay_hints)?
            .on::<lsp_ext::CodeActionRequest>(handlers::handle_code_action)?
            .on::<lsp_ext::ResolveCodeActionRequest>(handlers::handle_resolve_code_action)?
            .on::<lsp_ext::HoverRequest>(handlers::handle_hover)?
            .on::<lsp_types::request::OnTypeFormatting>(handlers::handle_on_type_formatting)?
            .on::<lsp_types::request::DocumentSymbolRequest>(handlers::handle_document_symbol)?
            .on::<lsp_types::request::WorkspaceSymbol>(handlers::handle_workspace_symbol)?
            .on::<lsp_types::request::GotoDefinition>(handlers::handle_goto_definition)?
            .on::<lsp_types::request::GotoImplementation>(handlers::handle_goto_implementation)?
            .on::<lsp_types::request::GotoTypeDefinition>(handlers::handle_goto_type_definition)?
            .on::<lsp_types::request::Completion>(handlers::handle_completion)?
            .on::<lsp_types::request::CodeLensRequest>(handlers::handle_code_lens)?
            .on::<lsp_types::request::CodeLensResolve>(handlers::handle_code_lens_resolve)?
            .on::<lsp_types::request::FoldingRangeRequest>(handlers::handle_folding_range)?
            .on::<lsp_types::request::SignatureHelpRequest>(handlers::handle_signature_help)?
            .on::<lsp_types::request::PrepareRenameRequest>(handlers::handle_prepare_rename)?
            .on::<lsp_types::request::Rename>(handlers::handle_rename)?
            .on::<lsp_types::request::References>(handlers::handle_references)?
            .on::<lsp_types::request::Formatting>(handlers::handle_formatting)?
            .on::<lsp_types::request::DocumentHighlightRequest>(
                handlers::handle_document_highlight,
            )?
            .on::<lsp_types::request::CallHierarchyPrepare>(
                handlers::handle_call_hierarchy_prepare,
            )?
            .on::<lsp_types::request::CallHierarchyIncomingCalls>(
                handlers::handle_call_hierarchy_incoming,
            )?
            .on::<lsp_types::request::CallHierarchyOutgoingCalls>(
                handlers::handle_call_hierarchy_outgoing,
            )?
            .on::<lsp_types::request::SemanticTokensRequest>(handlers::handle_semantic_tokens)?
            .on::<lsp_types::request::SemanticTokensRangeRequest>(
                handlers::handle_semantic_tokens_range,
            )?
            .on::<lsp_ext::Ssr>(handlers::handle_ssr)?
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
                if let Some(response) = this.req_queue.incoming.cancel(id) {
                    this.send(response.into());
                }
                Ok(())
            })?
            .on::<lsp_types::notification::DidOpenTextDocument>(|this, params| {
                if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                    if !this.mem_docs.insert(path.clone()) {
                        log::error!("duplicate DidOpenTextDocument: {}", path)
                    }
                    this.vfs
                        .write()
                        .0
                        .set_file_contents(path, Some(params.text_document.text.into_bytes()));
                }
                Ok(())
            })?
            .on::<lsp_types::notification::DidChangeTextDocument>(|this, params| {
                if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                    assert!(this.mem_docs.contains(&path));
                    let vfs = &mut this.vfs.write().0;
                    let file_id = vfs.file_id(&path).unwrap();
                    let mut text = String::from_utf8(vfs.file_contents(file_id).to_vec()).unwrap();
                    apply_document_changes(&mut text, params.content_changes);
                    vfs.set_file_contents(path, Some(text.into_bytes()))
                }
                Ok(())
            })?
            .on::<lsp_types::notification::DidCloseTextDocument>(|this, params| {
                if let Ok(path) = from_proto::vfs_path(&params.text_document.uri) {
                    if !this.mem_docs.remove(&path) {
                        log::error!("orphan DidCloseTextDocument: {}", path)
                    }
                    if let Some(path) = path.as_path() {
                        this.loader.invalidate(path.to_path_buf());
                    }
                }
                let params = lsp_types::PublishDiagnosticsParams {
                    uri: params.text_document.uri,
                    diagnostics: Vec::new(),
                    version: None,
                };
                let not = notification_new::<lsp_types::notification::PublishDiagnostics>(params);
                this.send(not.into());
                Ok(())
            })?
            .on::<lsp_types::notification::DidSaveTextDocument>(|this, _params| {
                if let Some(flycheck) = &this.flycheck {
                    flycheck.0.update();
                }
                Ok(())
            })?
            .on::<lsp_types::notification::DidChangeConfiguration>(|this, _params| {
                // As stated in https://github.com/microsoft/language-server-protocol/issues/676,
                // this notification's parameters should be ignored and the actual config queried separately.
                let request = this.req_queue.outgoing.register(
                    lsp_types::request::WorkspaceConfiguration::METHOD.to_string(),
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
                            (None, Some(configs)) => {
                                if let Some(new_config) = configs.get(0) {
                                    let mut config = this.config.clone();
                                    config.update(&new_config);
                                    this.update_configuration(config);
                                }
                            }
                            (None, None) => log::error!(
                                "received empty server settings response from the client"
                            ),
                        }
                    },
                );
                this.send(request.into());

                return Ok(());
            })?
            .on::<lsp_types::notification::DidChangeWatchedFiles>(|this, params| {
                for change in params.changes {
                    if let Ok(path) = from_proto::abs_path(&change.uri) {
                        this.loader.invalidate(path);
                    }
                }
                Ok(())
            })?
            .finish();
        Ok(())
    }
    fn update_file_notifications_on_threadpool(&mut self, subscriptions: Vec<FileId>) {
        log::trace!("updating notifications for {:?}", subscriptions);
        if self.config.publish_diagnostics {
            let snapshot = self.snapshot();
            let subscriptions = subscriptions.clone();
            self.task_pool.0.spawn(move || {
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
                    })
                    .collect::<Vec<_>>();
                Task::Diagnostics(diagnostics)
            })
        }
        self.task_pool.0.spawn({
            let subs = subscriptions;
            let snap = self.snapshot();
            move || {
                snap.analysis.prime_caches(subs).unwrap_or_else(|_: Canceled| ());
                Task::Unit
            }
        });
    }
}

#[derive(Debug)]
pub(crate) enum Task {
    Response(Response),
    Diagnostics(()),
    Unit,
}

pub(crate) type ReqHandler = fn(&mut GlobalState, Response);
pub(crate) type ReqQueue = lsp_server::ReqQueue<(String, Instant), ReqHandler>;
const DO_NOTHING: ReqHandler = |_, _| ();

fn on_diagnostic_task(task: DiagnosticTask, global_state: &mut GlobalState) {
    let subscriptions = global_state.diagnostics.handle_task(task);

    for file_id in subscriptions {
        let url = file_id_to_url(&global_state.vfs.read().0, file_id);
        let diagnostics = global_state.diagnostics.diagnostics_for(file_id).cloned().collect();
        let params = lsp_types::PublishDiagnosticsParams { uri: url, diagnostics, version: None };
        let not = notification_new::<lsp_types::notification::PublishDiagnostics>(params);
        global_state.send(not.into());
    }
}

#[derive(Eq, PartialEq)]
enum Progress {
    Begin,
    Report,
    End,
}

fn percentage(done: usize, total: usize) -> f64 {
    (done as f64 / total.max(1) as f64) * 100.0
}

fn report_progress(
    global_state: &mut GlobalState,
    title: &str,
    state: Progress,
    message: Option<String>,
    percentage: Option<f64>,
) {
    if !global_state.config.client_caps.work_done_progress {
        return;
    }
    let token = lsp_types::ProgressToken::String(format!("rustAnalyzer/{}", title));
    let work_done_progress = match state {
        Progress::Begin => {
            let work_done_progress_create = global_state.req_queue.outgoing.register(
                lsp_types::request::WorkDoneProgressCreate::METHOD.to_string(),
                lsp_types::WorkDoneProgressCreateParams { token: token.clone() },
                DO_NOTHING,
            );
            global_state.send(work_done_progress_create.into());

            lsp_types::WorkDoneProgress::Begin(lsp_types::WorkDoneProgressBegin {
                title: title.into(),
                cancellable: None,
                message,
                percentage,
            })
        }
        Progress::Report => {
            lsp_types::WorkDoneProgress::Report(lsp_types::WorkDoneProgressReport {
                cancellable: None,
                message,
                percentage,
            })
        }
        Progress::End => {
            lsp_types::WorkDoneProgress::End(lsp_types::WorkDoneProgressEnd { message })
        }
    };
    let notification =
        notification_new::<lsp_types::notification::Progress>(lsp_types::ProgressParams {
            token,
            value: lsp_types::ProgressParamsValue::WorkDone(work_done_progress),
        });
    global_state.send(notification.into());
}
