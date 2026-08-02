use std::io::{self, BufReader, BufWriter};

use emmy_dap_types::prelude::events::{ExitedEventBody, StoppedEventBody};
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

enum DispatchOutcome {
    Continue,
    Exit,
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

            match self.dispatch_request(request, session)? {
                Ok(DispatchOutcome::Continue) => {}
                Ok(DispatchOutcome::Exit) => return interp_ok(Ok(())),
                Err(err) => return interp_ok(Err(err)),
            }
        }
    }

    fn dispatch_request<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult<DispatchOutcome>> {
        if self.state == DapState::Fresh && !matches!(&request.command, Command::Initialize(_)) {
            return interp_ok(
                self.respond_error(request, "initialize must be sent first")
                    .map(|()| DispatchOutcome::Continue),
            );
        }

        match &request.command {
            Command::Initialize(_) =>
                interp_ok(self.handle_initialize(request).map(|()| DispatchOutcome::Continue)),
            Command::Launch(_) =>
                interp_ok(self.handle_launch(request).map(|()| DispatchOutcome::Continue)),
            Command::ConfigurationDone => {
                let res = self.handle_configuration_done(request, session)?;
                interp_ok(res.map(|()| self.dispatch_outcome()))
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
            Command::Continue(_) => {
                let res = self.handle_continue(request, session)?;
                interp_ok(res.map(|()| self.dispatch_outcome()))
            }
            Command::SetBreakpoints(_) =>
                interp_ok(
                    self.handle_set_breakpoints(request, session)
                        .map(|()| DispatchOutcome::Continue),
                ),
            Command::Next(_) | Command::StepIn(_) => {
                let body = match &request.command {
                    Command::Next(_) => ResponseBody::Next,
                    Command::StepIn(_) => ResponseBody::StepIn,
                    _ => bug!("step body is selected by the outer Next/StepIn match"),
                };
                let res = self.handle_step(request, body, session)?;
                interp_ok(res.map(|()| self.dispatch_outcome()))
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
        if self.reject_after_termination(&request)?
            || self.require_state(&request, DapState::Initialized, "launch requires initialize")?
        {
            return Ok(());
        }

        let response = request.success(ResponseBody::Launch);
        self.server.respond(response)?;
        self.state = DapState::Launched;
        Ok(())
    }

    fn handle_scopes<'tcx>(
        &mut self,
        request: Request,
        _session: &PrirodaContext<'tcx>,
    ) -> ServerResult {
        if self.reject_after_termination(&request)?
            || self.require_stopped(&request)?
            || self.require_frame_id(&request)?
        {
            return Ok(());
        }

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
        if self.reject_after_termination(&request)?
            || self.require_stopped(&request)?
            || self.require_variables_reference(&request)?
        {
            return Ok(());
        }

        let variables = match &request.command {
            Command::Variables(_) =>
                session.list_locals().into_iter().map(Self::local_to_variable).collect(),
            _ => bug!("dispatch routes only Variables to handle_variables"),
        };

        let response = request.success(ResponseBody::Variables(VariablesResponse { variables }));
        self.server.respond(response)
    }

    fn handle_configuration_done<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult> {
        let rejected = match self.check_configuration_done_request(&request) {
            Ok(rejected) => rejected,
            Err(err) => return interp_ok(Err(err)),
        };
        if rejected {
            return interp_ok(Ok(()));
        }

        match Self::execution_outcome(session.stop_at_first_user_location()) {
            ExecutionOutcome::Stopped(_) => {
                let response = request.success(ResponseBody::ConfigurationDone);
                interp_ok(self.server.respond(response).and_then(|()| {
                    self.state = DapState::Stopped;
                    self.send_stopped_event(StoppedEventReason::Entry)
                }))
            }
            ExecutionOutcome::Terminated { code } =>
                interp_ok(self.respond_terminated(request, ResponseBody::ConfigurationDone, code)),
            ExecutionOutcome::Failed(message) =>
                interp_ok(self.respond_execution_error(request, message)),
        }
    }

    /// FIXME: replace this with Miri thread state once Priroda exposes a
    /// frontend-facing thread model.
    fn handle_threads(&mut self, request: Request) -> ServerResult {
        if self.reject_after_termination(&request)? {
            return Ok(());
        }

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
        if self.reject_after_termination(&request)?
            || self.require_stopped(&request)?
            || self.require_thread_id(&request)?
        {
            return Ok(());
        }

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
        self.server.respond(response)
    }

    /// FIXME: grow capabilities as Priroda adds DAP features.
    fn handle_initialize(&mut self, request: Request) -> ServerResult {
        // Advertise configurationDone support ahead of its handler so VS Code
        // completes the full handshake; the handler arrives in a later commit.
        if self.reject_after_termination(&request)? {
            return Ok(());
        }
        if self.state != DapState::Fresh {
            return self.respond_error(request, "initialize may only be sent once");
        }

        let response = request.success(ResponseBody::Initialize(Capabilities {
            supports_configuration_done_request: Some(true),
            supports_single_thread_execution_requests: Some(true),
            ..Capabilities::default()
        }));
        self.server.respond(response)?;
        self.server.send_event(Event::Initialized)?;
        self.state = DapState::Initialized;
        Ok(())
    }

    /// FIXME: distinguish step-over from step-in once Priroda has call-aware stepping.
    fn handle_step<'tcx>(
        &mut self,
        request: Request,
        body: ResponseBody,
        session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx, ServerResult> {
        let rejected = match self.check_step_request(&request) {
            Ok(rejected) => rejected,
            Err(err) => return interp_ok(Err(err)),
        };
        if rejected {
            return interp_ok(Ok(()));
        }

        match Self::execution_outcome(session.step()) {
            ExecutionOutcome::Stopped(result) =>
                interp_ok(self.server.respond(request.success(body)).and_then(|()| {
                    self.state = DapState::Stopped;
                    self.send_stopped_event(Self::stopped_reason(result))
                })),
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
    ) -> InterpResult<'tcx, ServerResult> {
        let rejected = match self.check_step_request(&request) {
            Ok(rejected) => rejected,
            Err(err) => return interp_ok(Err(err)),
        };
        if rejected {
            return interp_ok(Ok(()));
        }

        let body = ResponseBody::Continue(ContinueResponse { all_threads_continued: Some(true) });

        match Self::execution_outcome(session.continue_execution()) {
            ExecutionOutcome::Stopped(result) =>
                interp_ok(self.server.respond(request.success(body)).and_then(|()| {
                    self.state = DapState::Stopped;
                    self.send_stopped_event(Self::stopped_reason(result))
                })),
            ExecutionOutcome::Terminated { code } =>
                interp_ok(self.respond_terminated(request, body, code)),
            ExecutionOutcome::Failed(message) =>
                interp_ok(self.respond_execution_error(request, message)),
        }
    }

    fn handle_set_breakpoints<'tcx>(
        &mut self,
        request: Request,
        session: &mut PrirodaContext<'tcx>,
    ) -> ServerResult {
        if self.reject_after_termination(&request)? {
            return Ok(());
        }

        let mut breakpoints = Vec::new();
        if let Command::SetBreakpoints(ref args) = request.command {
            if let Some(ref path_str) = args.source.path {
                let path = std::path::PathBuf::from(path_str);
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
            }
        }

        let response =
            request.success(ResponseBody::SetBreakpoints(SetBreakpointsResponse { breakpoints }));
        self.server.respond(response)
    }

    fn handle_disconnect(&mut self, request: Request) -> ServerResult {
        self.server.respond(request.success(ResponseBody::Disconnect))?;
        self.state = DapState::Terminated;
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

    fn reject_after_termination(&mut self, request: &Request) -> ServerResult<bool> {
        if self.state == DapState::Terminated {
            self.server.respond(request.clone().error("request received after termination"))?;
            return Ok(true);
        }

        Ok(false)
    }

    fn require_state(
        &mut self,
        request: &Request,
        expected: DapState,
        message: &'static str,
    ) -> ServerResult<bool> {
        if self.state != expected {
            self.server.respond(request.clone().error(message))?;
            return Ok(true);
        }

        Ok(false)
    }

    fn require_stopped(&mut self, request: &Request) -> ServerResult<bool> {
        if self.state != DapState::Stopped {
            self.server.respond(request.clone().error("request requires a stopped frame"))?;
            return Ok(true);
        }

        Ok(false)
    }

    fn require_thread_id(&mut self, request: &Request) -> ServerResult<bool> {
        let valid = match &request.command {
            Command::StackTrace(args) => args.thread_id == THREAD_ID,
            Command::Next(args) => args.thread_id == THREAD_ID,
            Command::StepIn(args) => args.thread_id == THREAD_ID,
            Command::Continue(args) => args.thread_id == THREAD_ID,
            _ => true,
        };

        if !valid {
            self.server.respond(request.clone().error("unknown threadId"))?;
            return Ok(true);
        }

        Ok(false)
    }

    fn require_frame_id(&mut self, request: &Request) -> ServerResult<bool> {
        let Command::Scopes(args) = &request.command else {
            bug!("dispatch routes only scopes to require_frame_id");
        };

        if args.frame_id != STACK_FRAME_ID {
            self.server.respond(request.clone().error("unknown frameId"))?;
            return Ok(true);
        }

        Ok(false)
    }

    fn require_variables_reference(&mut self, request: &Request) -> ServerResult<bool> {
        let Command::Variables(args) = &request.command else {
            bug!("dispatch routes only variables to require_variables_reference");
        };

        if args.variables_reference != LOCALS_VARIABLES_REFERENCE {
            self.server.respond(request.clone().error("unknown variablesReference"))?;
            return Ok(true);
        }

        Ok(false)
    }

    fn check_configuration_done_request(&mut self, request: &Request) -> ServerResult<bool> {
        if self.reject_after_termination(request)? {
            return Ok(true);
        }

        if self.state == DapState::Stopped {
            self.server
                .respond(request.clone().error("configurationDone may only be sent once"))?;
            return Ok(true);
        }

        if self.require_state(request, DapState::Launched, "configurationDone requires launch")? {
            return Ok(true);
        }

        Ok(false)
    }

    fn check_step_request(&mut self, request: &Request) -> ServerResult<bool> {
        if self.reject_after_termination(request)?
            || self.require_stopped(request)?
            || self.require_thread_id(request)?
        {
            return Ok(true);
        }

        Ok(false)
    }

    fn respond_error(&mut self, request: Request, message: &str) -> ServerResult {
        self.server.respond(request.error(message))
    }

    fn dispatch_outcome(&self) -> DispatchOutcome {
        if self.state == DapState::Terminated {
            DispatchOutcome::Exit
        } else {
            DispatchOutcome::Continue
        }
    }

    fn respond_execution_error(&mut self, request: Request, message: String) -> ServerResult {
        self.state = DapState::Terminated;
        self.server.respond(request.error(&message))?;
        self.server.send_event(Event::Terminated(None))
    }

    fn respond_terminated(
        &mut self,
        request: Request,
        body: ResponseBody,
        code: i32,
    ) -> ServerResult {
        self.state = DapState::Terminated;
        self.server.respond(request.success(body))?;
        self.server.send_event(Event::Exited(ExitedEventBody { exit_code: code.into() }))?;
        self.server.send_event(Event::Terminated(None))?;
        Ok(())
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
