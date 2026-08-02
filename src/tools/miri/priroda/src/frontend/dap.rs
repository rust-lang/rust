use std::io::{self, BufReader, BufWriter};

use emmy_dap_types::prelude::events::StoppedEventBody;
use emmy_dap_types::prelude::responses::{
    ScopesResponse, StackTraceResponse, ThreadsResponse, VariablesResponse,
};
use emmy_dap_types::prelude::types::{
    Capabilities, Scope, ScopePresentationhint, Source, StackFrame, StoppedEventReason, Thread,
    Variable,
};
use emmy_dap_types::prelude::{Command, Event, Request, ResponseBody, Server};
use miri::{InterpResult, bug, interp_ok};

use crate::debugger::{LocalDesc, PrirodaContext, StepResult};

// Priroda still exposes one interpreted thread and one selected frame to DAP.
// Keep the ids stable so editor follow-up requests can address the stopped state.
const THREAD_ID: i64 = 1;
const STACK_FRAME_ID: i64 = 1;
const LOCALS_VARIABLES_REFERENCE: i64 = 1;
const MAX_REQUEST_COUNT: usize = 128;
type ServerResult<T = ()> = Result<T, emmy_dap_types::errors::ServerError>;

enum DispatchOutcome {
    Continue,
    Exit,
}

/// Debug Adapter Protocol frontend.
pub(crate) struct Dap;

impl Dap {
    /// Serve DAP requests on stdin/stdout.
    pub(crate) fn run_dap_loop<'tcx>(
        &self,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx> {
        // FIXME: make this unbounded once Priroda has a full session lifecycle.
        if let Err(err) = DapSession::stdio().run_requests(session)? {
            eprintln!("priroda dap error: {err}");
        }

        interp_ok(())
    }
}

type DapServer = Server<io::StdinLock<'static>, io::StdoutLock<'static>>;

/// Owns the DAP stdio transport and dispatches requests into Priroda handlers.
struct DapSession {
    server: DapServer,
    initialized: bool,
}

impl DapSession {
    fn stdio() -> Self {
        Self {
            server: Server::new(
                BufReader::new(io::stdin().lock()),
                BufWriter::new(io::stdout().lock()),
            ),
            initialized: false,
        }
    }

    fn run_requests<'tcx>(
        &mut self,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult> {
        for _ in 0..MAX_REQUEST_COUNT {
            let request = match self.server.poll_request() {
                Ok(Some(request)) => request,
                Ok(None) => return interp_ok(Ok(())),
                Err(err) => return interp_ok(Err(err)),
            };

            match self.dispatch_request(request, session)? {
                Ok(DispatchOutcome::Continue) => {}
                Ok(DispatchOutcome::Exit) => return interp_ok(Ok(())),
                Err(err) => return interp_ok(Err(err)),
            }
        }

        interp_ok(Ok(()))
    }

    fn dispatch_request<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult<DispatchOutcome>> {
        // Reject non-initialize requests before the handshake completes so the
        // client gets a framed error.
        if !self.initialized && !matches!(&request.command, Command::Initialize(_)) {
            return interp_ok(
                self.handle_unsupported_request(request).map(|()| DispatchOutcome::Continue),
            );
        }

        match &request.command {
            Command::Initialize(_) =>
                interp_ok(self.handle_initialize(request).map(|()| DispatchOutcome::Continue)),
            Command::Launch(_) =>
                interp_ok(self.handle_launch(request).map(|()| DispatchOutcome::Continue)),
            Command::ConfigurationDone => {
                let res = self.handle_configuration_done(request, session)?;
                interp_ok(res.map(|()| DispatchOutcome::Continue))
            }
            Command::Threads =>
                interp_ok(self.handle_threads(request).map(|()| DispatchOutcome::Continue)),
            Command::StackTrace(_) =>
                interp_ok(
                    self.handle_stack_trace(request, session).map(|()| DispatchOutcome::Continue),
                ),
            Command::Scopes(_) =>
                interp_ok(self.handle_scopes(request, session).map(|()| DispatchOutcome::Continue)),
            Command::Variables(_) =>
                interp_ok(
                    self.handle_variables(request, session).map(|()| DispatchOutcome::Continue),
                ),
            Command::Next(_) | Command::StepIn(_) => {
                let body = match &request.command {
                    Command::Next(_) => ResponseBody::Next,
                    Command::StepIn(_) => ResponseBody::StepIn,
                    _ => unreachable!(),
                };
                let res = self.handle_step(request, body, session)?;
                interp_ok(res.map(|()| DispatchOutcome::Continue))
            }
            Command::Disconnect(_) =>
                interp_ok(self.handle_disconnect(request).map(|()| DispatchOutcome::Exit)),
            _ =>
                interp_ok(
                    self.handle_unsupported_request(request).map(|()| DispatchOutcome::Continue),
                ),
        }
    }

    /// FIXME: connect launch arguments to Priroda's session model.
    fn handle_launch(&mut self, request: Request) -> ServerResult {
        let response = request.success(ResponseBody::Launch);
        self.server.respond(response)
    }

    fn handle_scopes<'tcx>(
        &mut self,
        request: Request,
        _session: &PrirodaContext<'tcx>,
    ) -> ServerResult {
        let response = request.success(ResponseBody::Scopes(ScopesResponse {
            scopes: vec![Scope {
                name: "Locals".to_string(),
                presentation_hint: Some(ScopePresentationhint::Locals),
                variables_reference: LOCALS_VARIABLES_REFERENCE,
                named_variables: None,
                indexed_variables: Some(0),
                expensive: false,
                source: None,
                line: None,
                column: None,
                end_line: None,
                end_column: None,
            }],
        }));
        self.server.respond(response)
    }

    fn handle_variables<'tcx>(
        &mut self,
        request: Request,
        session: &PrirodaContext<'tcx>,
    ) -> ServerResult {
        let variables = match &request.command {
            Command::Variables(args) if args.variables_reference == LOCALS_VARIABLES_REFERENCE =>
                session.list_locals().into_iter().map(Self::local_to_variable).collect(),
            Command::Variables(_) => Vec::new(),
            _ => unreachable!(),
        };

        let response = request.success(ResponseBody::Variables(VariablesResponse { variables }));
        self.server.respond(response)
    }

    fn handle_configuration_done<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult> {
        session.stop_at_first_user_location()?;
        let response = request.success(ResponseBody::ConfigurationDone);
        interp_ok(
            self.server
                .respond(response)
                .and_then(|()| self.send_stopped_event(StoppedEventReason::Entry)),
        )
    }

    /// FIXME: replace this with Miri thread state once Priroda exposes a
    /// frontend-facing thread model.
    fn handle_threads(&mut self, request: Request) -> ServerResult {
        let response = request.success(ResponseBody::Threads(ThreadsResponse {
            threads: vec![Thread { id: THREAD_ID, name: "main".to_string() }],
        }));
        self.server.respond(response)
    }

    /// FIXME: report all frames once Priroda exposes a frontend-facing stack model.
    fn handle_stack_trace<'tcx>(
        &mut self,
        request: Request,
        session: &PrirodaContext<'tcx>,
    ) -> ServerResult {
        let stack_frames = match &session.current_location {
            Some(location) => {
                let path = session.local_path(location);
                vec![StackFrame {
                    id: STACK_FRAME_ID,
                    name: session.current_frame_name().unwrap_or_else(|| "<unknown>".to_string()),
                    source: path.as_ref().map(|path| {
                        Source {
                            name: path.file_name().map(|name| name.to_string_lossy().into_owned()),
                            path: Some(path.display().to_string()),
                            source_reference: None,
                            presentation_hint: None,
                            origin: None,
                            sources: None,
                            checksums: None,
                        }
                    }),
                    line: location
                        .line
                        .try_into()
                        .unwrap_or_else(|_| bug!("source line exceeds i64")),
                    column: location
                        .column
                        .try_into()
                        .unwrap_or_else(|_| bug!("source column exceeds i64")),
                    end_line: None,
                    end_column: None,
                    can_restart: None,
                    instruction_pointer_reference: None,
                    module_id: None,
                    presentation_hint: None,
                }]
            }
            None => Vec::new(),
        };
        let total_frames: i64 =
            stack_frames.len().try_into().unwrap_or_else(|_| bug!("frame count exceeds i64"));
        let response = request.success(ResponseBody::StackTrace(StackTraceResponse {
            stack_frames,
            total_frames: Some(total_frames),
        }));
        self.server.respond(response)
    }

    /// FIXME: grow capabilities as Priroda adds DAP features.
    fn handle_initialize(&mut self, request: Request) -> ServerResult {
        // Advertise configurationDone support ahead of its handler so VS Code
        // completes the full handshake; the handler arrives in a later commit.
        let response = request.success(ResponseBody::Initialize(Capabilities {
            supports_configuration_done_request: Some(true),
            ..Capabilities::default()
        }));
        self.server.respond(response)?;
        self.server.send_event(Event::Initialized)?;
        self.initialized = true;
        Ok(())
    }

    /// FIXME: distinguish step-over from step-in once Priroda has call-aware stepping.
    fn handle_step<'tcx>(
        &mut self,
        request: Request,
        body: ResponseBody,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult> {
        let result = session.step()?;
        interp_ok(
            self.server
                .respond(request.success(body))
                .and_then(|()| self.send_stopped_event(Self::stopped_reason(result))),
        )
    }

    fn handle_disconnect(&mut self, request: Request) -> ServerResult {
        self.server.respond(request.success(ResponseBody::Disconnect))?;
        self.server.send_event(Event::Terminated(None))
    }

    fn handle_unsupported_request(&mut self, request: Request) -> ServerResult {
        let message = format!(
            "unsupported request in Priroda DAP demo mode: {}",
            Self::display_command(&request.command)
        );
        let response = request.error(&message);
        self.server.respond(response)
    }

    fn send_stopped_event(&mut self, reason: StoppedEventReason) -> ServerResult {
        self.server.send_event(Event::Stopped(StoppedEventBody {
            reason,
            description: None,
            thread_id: Some(THREAD_ID),
            preserve_focus_hint: None,
            text: None,
            all_threads_stopped: Some(true),
            hit_breakpoint_ids: None,
        }))
    }

    fn stopped_reason(result: StepResult) -> StoppedEventReason {
        match result {
            StepResult::Step => StoppedEventReason::Step,
            StepResult::Breakpoint => StoppedEventReason::Breakpoint,
        }
    }

    fn display_command(command: &Command) -> &'static str {
        match command {
            Command::Initialize(_) => "initialize",
            Command::Launch(_) => "launch",
            Command::ConfigurationDone => "configurationDone",
            Command::Threads => "threads",
            Command::StackTrace(_) => "stackTrace",
            Command::Scopes(_) => "scopes",
            Command::Variables(_) => "variables",
            Command::Next(_) => "next",
            Command::StepIn(_) => "stepIn",
            Command::Disconnect(_) => "disconnect",
            _ => "unsupported",
        }
    }

    fn local_to_variable(local: LocalDesc) -> Variable {
        Variable {
            name: Self::local_name(&local),
            value: local.value,
            type_field: Some(local.ty),
            presentation_hint: None,
            evaluate_name: None,
            // FIXME: add child handles once Priroda can identify places across requests.
            variables_reference: 0,
            named_variables: None,
            indexed_variables: None,
            memory_reference: None,
        }
    }

    fn local_name(local: &LocalDesc) -> String {
        let source_projection = local.source_projection_str();

        // Prefer source names when debug info gives us one. If a local only has
        // MIR storage identity, keep that visible so the DAP Variables view
        // still has a stable row for every backing local.
        if let Some(source_name) = local.source_name {
            return format!("{source_name}{source_projection}");
        }

        let local_id = local
            .local
            .map_or_else(|| "<none>".to_string(), |local_idx| format!("_{}", local_idx.index()));
        format!("{local_id}{}", local.storage_projection_str())
    }
}
