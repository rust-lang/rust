use std::io::{self, BufReader, BufWriter};

use emmy_dap_types::errors::ServerError;
use emmy_dap_types::prelude::events::{ExitedEventBody, StoppedEventBody};
use emmy_dap_types::prelude::requests::SetBreakpointsArguments;
use emmy_dap_types::prelude::responses::{
    ContinueResponse, ScopesResponse, SetBreakpointsResponse, StackTraceResponse, ThreadsResponse,
    VariablesResponse,
};
use emmy_dap_types::prelude::types::{
    Breakpoint as DapBreakpoint, Capabilities, Scope, ScopePresentationhint, Source, StackFrame,
    StoppedEventReason, Thread, Variable,
};
use emmy_dap_types::prelude::{Command, Event, Request, ResponseBody, Server};
use miri::{InterpErrorInfo, InterpErrorKind, InterpResult, TerminationInfo, bug, interp_ok};

use crate::debugger::{LocalDesc, PrirodaContext, StepResult};

// Priroda still exposes one interpreted thread and one selected frame to DAP.
// Keep the ids stable so editor follow-up requests can address the stopped state.
const THREAD_ID: i64 = 1;
const STACK_FRAME_ID: i64 = 1;
const LOCALS_VARIABLES_REFERENCE: i64 = 1;
type ServerResult<T = ()> = Result<T, emmy_dap_types::errors::ServerError>;

enum HandlerOutcome {
    Continue,
    Exit,
}

enum HandlerError {
    Reject(&'static str),
    Transport(ServerError),
}

impl From<ServerError> for HandlerError {
    fn from(e: ServerError) -> Self {
        HandlerError::Transport(e)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DapState {
    Fresh,
    Initialized,
    Launched,
    Stopped,
    Terminated,
}

enum ExecutionOutcome {
    Stopped(StepResult),
    Terminated { code: i32 },
    Failed(String),
}

/// Debug Adapter Protocol frontend.
pub(crate) struct Dap;

impl Dap {
    /// Serve DAP requests on stdin/stdout.
    pub(crate) fn run_dap_loop<'tcx>(
        &self,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx> {
        if let Err(err) = DapSession::stdio().run_requests(session)? {
            eprintln!("priroda dap error: {err:?}");
        }

        interp_ok(())
    }
}

type DapServer = Server<io::StdinLock<'static>, io::StdoutLock<'static>>;

/// Owns the DAP stdio transport and dispatches requests into Priroda handlers.
struct DapSession {
    server: DapServer,
    state: DapState,
}

impl DapSession {
    fn stdio() -> Self {
        Self {
            server: Server::new(
                BufReader::new(io::stdin().lock()),
                BufWriter::new(io::stdout().lock()),
            ),
            state: DapState::Fresh,
        }
    }

    fn run_requests<'tcx>(
        &mut self,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult> {
        loop {
            let request = match self.server.poll_request() {
                Ok(Some(request)) => request,
                Ok(None) => return interp_ok(Ok(())),
                Err(err) => return interp_ok(Err(err)),
            };

            let request_for_dispatch = request.clone();

            match self.dispatch_request(request_for_dispatch, session)? {
                Ok(HandlerOutcome::Continue) => {}
                Ok(HandlerOutcome::Exit) => return interp_ok(Ok(())),
                Err(HandlerError::Reject(msg)) => {
                    let response = request.error(msg);
                    if let Err(err) = self.server.respond(response) {
                        return interp_ok(Err(err));
                    }
                }
                Err(HandlerError::Transport(e)) => return interp_ok(Err(e)),
            }
        }
    }

    fn dispatch_request<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, Result<HandlerOutcome, HandlerError>> {
        if let Err(msg) = self.require_initialized(&request) {
            return interp_ok(Err(HandlerError::Reject(msg)));
        }

        let outcome = match &request.command {
            Command::Initialize(_) => self.handle_initialize(request),
            Command::Launch(_) => self.handle_launch(request),
            Command::ConfigurationDone => return self.handle_configuration_done(request, session),
            Command::Threads => self.handle_threads(request),
            Command::StackTrace(_) => self.handle_stack_trace(request, session),
            Command::Scopes(args) => {
                let frame_id = args.frame_id;
                self.handle_scopes(request, frame_id, session)
            }
            Command::Variables(args) => {
                let variables_reference = args.variables_reference;
                self.handle_variables(request, variables_reference, session)
            }
            Command::Continue(_) => return self.handle_continue(request, session),
            Command::SetBreakpoints(args) => {
                let args = args.clone();
                self.handle_set_breakpoints(request, &args, session)
            }
            Command::Next(_) | Command::StepIn(_) => {
                let body = match &request.command {
                    Command::Next(_) => ResponseBody::Next,
                    Command::StepIn(_) => ResponseBody::StepIn,
                    _ => bug!("step body is selected by the outer Next/StepIn match"),
                };
                return self.handle_step(request, body, session);
            }
            Command::Disconnect(_) => self.handle_disconnect(request),
            Command::Attach(_)
            | Command::BreakpointLocations(_)
            | Command::Cancel(_)
            | Command::Completions(_)
            | Command::DataBreakpointInfo(_)
            | Command::Disassemble(_)
            | Command::Evaluate(_)
            | Command::ExceptionInfo(_)
            | Command::Goto(_)
            | Command::GotoTargets(_)
            | Command::LoadedSources
            | Command::Modules(_)
            | Command::Pause(_)
            | Command::ReadMemory(_)
            | Command::Restart(_)
            | Command::RestartFrame(_)
            | Command::ReverseContinue(_)
            | Command::SetDataBreakpoints(_)
            | Command::SetExceptionBreakpoints(_)
            | Command::SetExpression(_)
            | Command::SetFunctionBreakpoints(_)
            | Command::SetInstructionBreakpoints(_)
            | Command::SetVariable(_)
            | Command::Source(_)
            | Command::StepBack(_)
            | Command::StepInTargets(_)
            | Command::StepOut(_)
            | Command::Terminate(_)
            | Command::TerminateThreads(_)
            | Command::WriteMemory(_) => self.handle_unsupported_request(request),
        };
        interp_ok(outcome)
    }

    /// FIXME: connect launch arguments to Priroda's session model.
    fn handle_launch(&mut self, request: Request) -> Result<HandlerOutcome, HandlerError> {
        self.require_state(DapState::Initialized).map_err(HandlerError::Reject)?;

        let response = request.success(ResponseBody::Launch);
        self.server.respond(response)?;
        self.state = DapState::Launched;
        Ok(HandlerOutcome::Continue)
    }

    fn handle_scopes<'tcx>(
        &mut self,
        request: Request,
        frame_id: i64,
        session: &PrirodaContext<'tcx>,
    ) -> Result<HandlerOutcome, HandlerError> {
        self.require_stopped().map_err(HandlerError::Reject)?;
        Self::require_frame_id(frame_id).map_err(HandlerError::Reject)?;

        let (source, line, column) = match &session.current_location {
            Some(location) => {
                let source = session.local_path(location).as_ref().map(|path| {
                    Source {
                        name: path.file_name().map(|name| name.to_string_lossy().into_owned()),
                        path: Some(path.display().to_string()),
                        source_reference: Some(0),
                        presentation_hint: None,
                        origin: None,
                        sources: None,
                        checksums: None,
                    }
                });
                let line =
                    location.line.try_into().unwrap_or_else(|_| bug!("source line exceeds i64"));
                let column = location
                    .column
                    .try_into()
                    .unwrap_or_else(|_| bug!("source column exceeds i64"));
                (source, Some(line), Some(column))
            }
            None => (None, None, None),
        };
        let response = request.success(ResponseBody::Scopes(ScopesResponse {
            scopes: vec![Scope {
                name: "Locals".to_string(),
                presentation_hint: Some(ScopePresentationhint::Locals),
                variables_reference: LOCALS_VARIABLES_REFERENCE,
                named_variables: None,
                indexed_variables: Some(0),
                expensive: false,
                source,
                line,
                column,
                end_line: None,
                end_column: None,
            }],
        }));
        self.server.respond(response)?;
        Ok(HandlerOutcome::Continue)
    }

    fn handle_variables<'tcx>(
        &mut self,
        request: Request,
        variables_reference: i64,
        session: &PrirodaContext<'tcx>,
    ) -> Result<HandlerOutcome, HandlerError> {
        self.require_stopped().map_err(HandlerError::Reject)?;
        Self::require_variables_reference(variables_reference).map_err(HandlerError::Reject)?;

        let variables = if variables_reference == LOCALS_VARIABLES_REFERENCE {
            session.list_locals().into_iter().map(Self::local_to_variable).collect()
        } else {
            Vec::new()
        };

        let response = request.success(ResponseBody::Variables(VariablesResponse { variables }));
        self.server.respond(response)?;
        Ok(HandlerOutcome::Continue)
    }

    fn handle_configuration_done<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, Result<HandlerOutcome, HandlerError>> {
        if let Err(msg) = self.require_state(DapState::Launched) {
            return interp_ok(Err(HandlerError::Reject(msg)));
        }

        match Self::execution_outcome(session.stop_at_first_user_location()) {
            ExecutionOutcome::Stopped(_) => {
                let response = request.success(ResponseBody::ConfigurationDone);
                interp_ok(
                    self.server
                        .respond(response)
                        .and_then(|()| {
                            self.state = DapState::Stopped;
                            self.send_stopped_event(StoppedEventReason::Entry)
                        })
                        .map(|()| HandlerOutcome::Continue)
                        .map_err(HandlerError::Transport),
                )
            }
            ExecutionOutcome::Terminated { code } =>
                interp_ok(self.respond_terminated(request, ResponseBody::ConfigurationDone, code)),
            ExecutionOutcome::Failed(message) =>
                interp_ok(self.respond_execution_error(request, message)),
        }
    }

    /// FIXME: replace this with Miri thread state once Priroda exposes a
    /// frontend-facing thread model.
    fn handle_threads(&mut self, request: Request) -> Result<HandlerOutcome, HandlerError> {
        self.reject_after_termination().map_err(HandlerError::Reject)?;

        let response = request.success(ResponseBody::Threads(ThreadsResponse {
            threads: vec![Thread { id: THREAD_ID, name: "main".to_string() }],
        }));
        self.server.respond(response)?;
        Ok(HandlerOutcome::Continue)
    }

    /// FIXME: report all frames once Priroda exposes a frontend-facing stack model.
    fn handle_stack_trace<'tcx>(
        &mut self,
        request: Request,
        session: &PrirodaContext<'tcx>,
    ) -> Result<HandlerOutcome, HandlerError> {
        self.require_stopped().map_err(HandlerError::Reject)?;
        Self::require_thread_id(&request).map_err(HandlerError::Reject)?;

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
                            source_reference: Some(0),
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
        self.server.respond(response)?;
        Ok(HandlerOutcome::Continue)
    }

    /// FIXME: grow capabilities as Priroda adds DAP features.
    fn handle_initialize(&mut self, request: Request) -> Result<HandlerOutcome, HandlerError> {
        if self.state != DapState::Fresh {
            return Err(HandlerError::Reject("initialize may only be sent once"));
        }

        let response = request.success(ResponseBody::Initialize(Capabilities {
            supports_configuration_done_request: Some(true),
            supports_single_thread_execution_requests: Some(true),
            ..Capabilities::default()
        }));
        self.server.respond(response)?;
        self.server.send_event(Event::Initialized)?;
        self.state = DapState::Initialized;
        Ok(HandlerOutcome::Continue)
    }

    /// FIXME: distinguish step-over from step-in once Priroda has call-aware stepping.
    fn handle_step<'tcx>(
        &mut self,
        request: Request,
        body: ResponseBody,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, Result<HandlerOutcome, HandlerError>> {
        if let Err(msg) = self.require_stopped() {
            return interp_ok(Err(HandlerError::Reject(msg)));
        }
        if let Err(msg) = Self::require_thread_id(&request) {
            return interp_ok(Err(HandlerError::Reject(msg)));
        }

        match Self::execution_outcome(session.step()) {
            ExecutionOutcome::Stopped(result) =>
                interp_ok(
                    self.server
                        .respond(request.success(body))
                        .and_then(|()| {
                            self.state = DapState::Stopped;
                            self.send_stopped_event(Self::stopped_reason(result))
                        })
                        .map(|()| HandlerOutcome::Continue)
                        .map_err(HandlerError::Transport),
                ),
            ExecutionOutcome::Terminated { code } =>
                interp_ok(self.respond_terminated(request, body, code)),
            ExecutionOutcome::Failed(message) =>
                interp_ok(self.respond_execution_error(request, message)),
        }
    }

    fn handle_continue<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, Result<HandlerOutcome, HandlerError>> {
        if let Err(msg) = self.require_stopped() {
            return interp_ok(Err(HandlerError::Reject(msg)));
        }
        if let Err(msg) = Self::require_thread_id(&request) {
            return interp_ok(Err(HandlerError::Reject(msg)));
        }

        let body = ResponseBody::Continue(ContinueResponse { all_threads_continued: Some(true) });

        match Self::execution_outcome(session.continue_execution()) {
            ExecutionOutcome::Stopped(result) =>
                interp_ok(
                    self.server
                        .respond(request.success(body))
                        .and_then(|()| {
                            self.state = DapState::Stopped;
                            self.send_stopped_event(Self::stopped_reason(result))
                        })
                        .map(|()| HandlerOutcome::Continue)
                        .map_err(HandlerError::Transport),
                ),
            ExecutionOutcome::Terminated { code } =>
                interp_ok(self.respond_terminated(request, body, code)),
            ExecutionOutcome::Failed(message) =>
                interp_ok(self.respond_execution_error(request, message)),
        }
    }

    fn handle_set_breakpoints<'tcx>(
        &mut self,
        request: Request,
        args: &SetBreakpointsArguments,
        session: &mut PrirodaContext<'tcx>,
    ) -> Result<HandlerOutcome, HandlerError> {
        self.reject_after_termination().map_err(HandlerError::Reject)?;

        let Some(ref path_str) = args.source.path else {
            return Err(HandlerError::Reject(
                "setBreakpoints requires a source.path; sourceReference loads are not supported",
            ));
        };

        let path = std::path::PathBuf::from(path_str);
        let mut breakpoints = Vec::new();
        if let Some(ref req_bps) = args.breakpoints {
            for req_bp in req_bps {
                let line = req_bp.line as usize;
                session.set_breakpoint(path.clone(), line);
                breakpoints.push(DapBreakpoint {
                    verified: true,
                    message: None,
                    source: Some(args.source.clone()),
                    line: Some(req_bp.line),
                    column: req_bp.column,
                    end_line: None,
                    end_column: None,
                    id: None,
                    instruction_reference: None,
                    offset: None,
                });
            }
        }

        let response =
            request.success(ResponseBody::SetBreakpoints(SetBreakpointsResponse { breakpoints }));
        self.server.respond(response)?;
        Ok(HandlerOutcome::Continue)
    }

    fn handle_disconnect(&mut self, request: Request) -> Result<HandlerOutcome, HandlerError> {
        self.server.respond(request.success(ResponseBody::Disconnect))?;
        self.state = DapState::Terminated;
        self.server.send_event(Event::Terminated(None))?;
        Ok(HandlerOutcome::Exit)
    }

    fn handle_unsupported_request(
        &mut self,
        request: Request,
    ) -> Result<HandlerOutcome, HandlerError> {
        let message = format!(
            "unsupported request in Priroda DAP demo mode: {}",
            Self::display_command(&request.command)
        );
        let response = request.error(&message);
        self.server.respond(response)?;
        Ok(HandlerOutcome::Continue)
    }

    fn reject_after_termination(&self) -> Result<(), &'static str> {
        if self.state == DapState::Terminated {
            return Err("request received after termination");
        }
        Ok(())
    }

    fn require_state(&self, expected: DapState) -> Result<(), &'static str> {
        if self.state != expected {
            return Err(match expected {
                DapState::Initialized => "launch requires initialize",
                DapState::Launched => "configurationDone requires launch",
                _ => "invalid session state for request",
            });
        }
        Ok(())
    }

    fn require_initialized(&self, request: &Request) -> Result<(), &'static str> {
        if self.state == DapState::Fresh && !matches!(&request.command, Command::Initialize(_)) {
            return Err("initialize must be sent first");
        }
        Ok(())
    }

    fn require_stopped(&self) -> Result<(), &'static str> {
        if self.state != DapState::Stopped {
            return Err("request requires a stopped frame");
        }
        Ok(())
    }

    fn require_thread_id(request: &Request) -> Result<(), &'static str> {
        let valid = match &request.command {
            Command::StackTrace(args) => args.thread_id == THREAD_ID,
            Command::Next(args) => args.thread_id == THREAD_ID,
            Command::StepIn(args) => args.thread_id == THREAD_ID,
            Command::Continue(args) => args.thread_id == THREAD_ID,
            _ => true,
        };

        if !valid {
            return Err("unknown threadId");
        }
        Ok(())
    }

    fn require_frame_id(frame_id: i64) -> Result<(), &'static str> {
        if frame_id != STACK_FRAME_ID {
            return Err("unknown frameId");
        }
        Ok(())
    }

    fn require_variables_reference(variables_reference: i64) -> Result<(), &'static str> {
        if variables_reference != LOCALS_VARIABLES_REFERENCE {
            return Err("unknown variablesReference");
        }
        Ok(())
    }

    fn respond_execution_error(
        &mut self,
        request: Request,
        message: String,
    ) -> Result<HandlerOutcome, HandlerError> {
        self.state = DapState::Terminated;
        self.server.respond(request.error(&message))?;
        self.server.send_event(Event::Terminated(None))?;
        Ok(HandlerOutcome::Exit)
    }

    fn respond_terminated(
        &mut self,
        request: Request,
        body: ResponseBody,
        code: i32,
    ) -> Result<HandlerOutcome, HandlerError> {
        self.state = DapState::Terminated;
        self.server.respond(request.success(body))?;
        self.server.send_event(Event::Exited(ExitedEventBody { exit_code: code.into() }))?;
        self.server.send_event(Event::Terminated(None))?;
        Ok(HandlerOutcome::Exit)
    }

    fn execution_outcome<'tcx>(result: InterpResult<'tcx, StepResult>) -> ExecutionOutcome {
        match result.report_err() {
            Ok(step) => ExecutionOutcome::Stopped(step),
            Err(err) => Self::interp_error_outcome(err),
        }
    }

    fn interp_error_outcome<'tcx>(err: InterpErrorInfo<'tcx>) -> ExecutionOutcome {
        let kind = err.into_kind();
        if let InterpErrorKind::MachineStop(info) = &kind
            && let Some(TerminationInfo::Exit { code, .. }) = info.downcast_ref::<TerminationInfo>()
        {
            return ExecutionOutcome::Terminated { code: *code };
        }

        ExecutionOutcome::Failed(kind.to_string())
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
            Command::Attach(_) => "attach",
            Command::BreakpointLocations(_) => "breakpointLocations",
            Command::Cancel(_) => "cancel",
            Command::Completions(_) => "completions",
            Command::Continue(_) => "continue",
            Command::DataBreakpointInfo(_) => "dataBreakpointInfo",
            Command::Disassemble(_) => "disassemble",
            Command::Evaluate(_) => "evaluate",
            Command::ExceptionInfo(_) => "exceptionInfo",
            Command::Goto(_) => "goto",
            Command::GotoTargets(_) => "gotoTargets",
            Command::LoadedSources => "loadedSources",
            Command::Modules(_) => "modules",
            Command::Pause(_) => "pause",
            Command::ReadMemory(_) => "readMemory",
            Command::Restart(_) => "restart",
            Command::RestartFrame(_) => "restartFrame",
            Command::ReverseContinue(_) => "reverseContinue",
            Command::SetBreakpoints(_) => "setBreakpoints",
            Command::SetDataBreakpoints(_) => "setDataBreakpoints",
            Command::SetExceptionBreakpoints(_) => "setExceptionBreakpoints",
            Command::SetExpression(_) => "setExpression",
            Command::SetFunctionBreakpoints(_) => "setFunctionBreakpoints",
            Command::SetInstructionBreakpoints(_) => "setInstructionBreakpoints",
            Command::SetVariable(_) => "setVariable",
            Command::Source(_) => "source",
            Command::StepBack(_) => "stepBack",
            Command::StepInTargets(_) => "stepInTargets",
            Command::StepOut(_) => "stepOut",
            Command::Terminate(_) => "terminate",
            Command::TerminateThreads(_) => "terminateThreads",
            Command::WriteMemory(_) => "writeMemory",
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
